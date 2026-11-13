#!/usr/bin/env python3
# Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Developed at the German Cancer Research Center (DKFZ).

"""Nested patient CV for a recovery head attached to a frozen existing detector."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from dkfz.live.cross_validation import (
    _validate_patient_id,
    ensure_output_outside_data,
    nested_lopo_folds,
    resolve_patient_groups,
    run_nested_lopo,
)
from dkfz.live.cv_paths import (
    NESTED_RECOVERY_LOPO_DIRNAME,
    cv_modelname,
    default_cv_output_dir,
    prepare_cv_output_dir,
)
from dkfz.live.io import repo_relative_path, resolve_input_paths
from dkfz.live.onnx_export import build_model_from_config
from dkfz.live.training_config import (
    load_training_config_values,
    training_kwargs_from_config_and_values,
)
from dkfz.live.training_pipeline import training_feature_columns, write_metadata


def recovery_patient_groups(config):
    """Group only the saved training files; never expand to benchmark recordings."""
    if not config.get("inputs"):
        raise ValueError(
            "Source config needs inputs, or provide explicit patient grouping"
        )
    groups = {}
    for path in resolve_input_paths(config["inputs"]):
        patient = _validate_patient_id(path.parent.name)
        groups.setdefault(patient, []).append(path)
    nested_lopo_folds(groups)
    return groups


def run_recovery_lopo(
    *,
    config,
    checkpoint,
    modelname,
    output_dir,
    values=None,
    patient_groups=None,
    overwrite=False,
):
    config, checkpoint = Path(config).resolve(), Path(checkpoint).resolve()
    cfg = load_training_config_values(str(config))
    if cfg.get("recovery_head"):
        raise ValueError("Start from the original single-output detector")
    if cfg.get("signal_quality_mode", "none") != "none" and not cfg.get(
        "signal_quality_stats"
    ):
        raise ValueError(
            "Frozen signal-quality preprocessing needs saved source calibration"
        )
    source_model = build_model_from_config(cfg, checkpoint)
    groups = recovery_patient_groups(cfg) if patient_groups is None else patient_groups
    nested_lopo_folds(groups)
    output = ensure_output_outside_data(output_dir)
    for source in (config, checkpoint):
        if source == output or output in source.parents:
            raise ValueError("Recovery output must not contain the source model")
    values = dict(values or {})
    allowed = {
        "num_epochs",
        "lr",
        "batch_size",
        "prep_workers",
        "seed",
        "use_amp",
        "save_prepped",
        "recovery_max_fp_increase",
        "recovery_min_extra_fp_per_minute",
        "cache_frozen_recovery",
        "plots",
    }
    if set(values) - allowed:
        raise ValueError(
            "Recovery CV overrides must not change the frozen model or preprocessing"
        )
    options = dict(
        config=str(config),
        modelname=cv_modelname(modelname),
        recovery_head=True,
        recovery_only=True,
        num_epochs=10,
        lr=1e-4,
        use_prepped=False,
        save_prepped=True,
    )
    options.update(values)
    kwargs = training_kwargs_from_config_and_values(options, config_values=cfg)
    # Null explicitly disables ECG3; the generic training resolver otherwise
    # falls back to today's default, changing an older detector's input shape.
    kwargs["ecg3_col"] = cfg.get("ecg3_col")
    kwargs["min_distance"] = int(
        cfg.get("holdoff_frames", cfg.get("min_distance", kwargs["min_distance"]))
    )
    features, input_dim = training_feature_columns(
        int(kwargs["frame_size_raw"]),
        bool(kwargs["not_use_puls"]),
        include_ecg3=kwargs["ecg3_col"] is not None,
        ecg_slope_mode=str(kwargs["ecg_slope_mode"]),
        signal_quality_mode=str(kwargs["signal_quality_mode"]),
    )
    if input_dim != cfg["input_dim"] or (
        cfg.get("feature_cols") is not None and features != cfg["feature_cols"]
    ):
        raise ValueError(
            "Source config does not match the frozen preprocessing feature contract"
        )
    kwargs.update(
        base_modelname=modelname,
        cv_method="nested_recovery_lopo",
        _recovery_checkpoint=str(checkpoint),
        _recovery_checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        _recovery_primary_policy=dict(
            threshold=(
                cfg.get("manual_threshold")
                if cfg.get("manual_threshold") is not None
                else cfg["threshold"]
            ),
            # Native early emission falls back to the automatic threshold,
            # even when a lower manual detection threshold is configured.
            fp_threshold=(
                cfg.get("fp_threshold")
                if cfg.get("fp_threshold") is not None
                else cfg["threshold"]
            ),
            der_threshold=cfg.get("der_threshold"),
        ),
    )
    output = prepare_cv_output_dir(output, overwrite=overwrite, require_empty=True)
    print(
        "Recovery-only nested LOPO: the pretrained base is frozen and may have seen the held-out patients."
    )
    summary = run_nested_lopo(
        patient_groups=groups,
        kwargs=kwargs,
        output_dir=output,
        seed=int(kwargs["seed"]),
    )

    # Assert the frozen-primary contract on every trained checkpoint before publishing.
    reference = source_model.state_dict()
    trained_checkpoints = sorted(output.rglob("*.pth"))
    for path in trained_checkpoints:
        state = torch.load(path, map_location="cpu", weights_only=True)
        if any(not torch.equal(value, state[key]) for key, value in reference.items()):
            raise RuntimeError(f"Frozen primary weights changed in {path}")
    if (
        hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        != kwargs["_recovery_checkpoint_sha256"]
    ):
        raise RuntimeError("Source checkpoint changed during recovery training")
    summary["primary_weights_verified_unchanged"] = True
    summary["verified_frozen_checkpoints"] = len(trained_checkpoints)
    write_metadata(output / "summary.json", summary)

    for path in [
        output / "train_config.json",
        *output.glob("outer/*/train_config.json"),
    ]:
        metadata = json.loads(path.read_text())
        for key in ("threshold", "manual_threshold", "fp_threshold", "der_threshold"):
            metadata[key] = cfg.get(key)
        metadata["primary_weights_verified_unchanged"] = True
        metadata["source_model_name"] = cfg.get("modelname")
        if path == output / "train_config.json":
            metadata["modelname"] = modelname
        write_metadata(path, metadata)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    from dkfz.live.training_config import add_training_plot_arguments

    add_training_plot_arguments(parser)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--modelname", required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--output-checkpoint", type=Path)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Restart this run by replacing its CV output directory.",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=10,
        help="Maximum recovery epochs; selected by nested CV.",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--cache-frozen-recovery",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Cache frozen representations for CPU head training (default: enabled).",
    )
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--recovery-max-fp-increase", type=float, default=0.10)
    parser.add_argument(
        "--recovery-min-extra-fp-per-minute",
        type=float,
        default=None,
        help="Minimum additional FP allowance per validation minute; default 0.5.",
    )
    parser.add_argument("--prep-workers", type=int)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--patient-root")
    source.add_argument("--patient-manifest")
    args = parser.parse_args()
    output = args.output_dir or default_cv_output_dir(
        args.modelname, NESTED_RECOVERY_LOPO_DIRNAME
    )
    groups = None
    if args.patient_root or args.patient_manifest:
        groups = resolve_patient_groups(
            patient_root=args.patient_root, patient_manifest=args.patient_manifest
        )
    if args.output_checkpoint and args.output_checkpoint.resolve() in (
        args.checkpoint.resolve(),
        args.config.resolve(),
    ):
        parser.error("Output checkpoint must not replace a source artifact")
    run_recovery_lopo(
        config=args.config,
        checkpoint=args.checkpoint,
        modelname=args.modelname,
        output_dir=output,
        patient_groups=groups,
        overwrite=args.overwrite,
        values=dict(
            num_epochs=args.num_epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            prep_workers=args.prep_workers,
            recovery_max_fp_increase=args.recovery_max_fp_increase,
            recovery_min_extra_fp_per_minute=args.recovery_min_extra_fp_per_minute,
            cache_frozen_recovery=args.cache_frozen_recovery,
            plots=args.plots,
        ),
    )
    if args.output_checkpoint:
        args.output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        if (output / "best.pth").resolve() != args.output_checkpoint.resolve():
            shutil.copy2(output / "best.pth", args.output_checkpoint)
        config_path = output / "train_config.json"
        metadata = json.loads(config_path.read_text())
        metadata["checkpoint"] = repo_relative_path(args.output_checkpoint)
        write_metadata(config_path, metadata)
    print(f"Recovery LOPO results: {repo_relative_path(output)}")


if __name__ == "__main__":
    main()
