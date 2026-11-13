#!/usr/bin/env python
# Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Developed at the German Cancer Research Center (DKFZ).

"""Update single-output FP/DER or two-output recovery policies from validation.

Default: report only. --apply updates configs; single-output updates keep a
one-time backup. For old single-output CV grids, --rebuild-cv-validation permits
retraining inner validation replicas. For two-output CV models it replays saved
inner checkpoints without training. Deployment weights and two-output primary
thresholds stay fixed. Outer test data are never used for selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dkfz.core.network_helpers import (
    evaluate_threshold_grid,
    select_threshold,
    select_threshold_from_search_rows,
)
from dkfz.live.event_matching import (
    outer_association_frames,
    event_tolerances,
    METRICS_SCHEMA_VERSION,
)
from post_reselect_thresholds import (
    pack_validation_sequences,
    read_validation_analysis_rows,
    write_json_atomic,
)


def digest(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def cv_archive(path, cfg):
    """Find only an explicitly linked archive or a verified identical checkpoint."""
    direct = cfg.get("selection_search_file")
    candidates = []
    if direct:
        linked = Path(direct)
        # Metadata copied from another workstation may contain foreign paths.
        if not linked.is_absolute() or linked.is_relative_to(Path.home()):
            candidates.append(linked.parent)
    candidates.append(path.parent)
    base = cfg.get("base_modelname")
    if base:
        candidates.append(
            ROOT / "models" / f"{base}_cv" / cfg.get("cv_method", "nested_lopo")
        )
    for folder in candidates:
        if not grid_path(folder).is_file():
            continue
        if folder.resolve() != path.parent.resolve():
            local = path.parent / "best.pth"
            archived = folder / "best.pth"
            if (
                not local.is_file()
                or not archived.is_file()
                or digest(local) != digest(archived)
            ):
                raise ValueError("CV archive checkpoint does not match this model")
        source_cfg_path = folder / "train_config.json"
        if source_cfg_path.is_file() and source_cfg_path.resolve() != path.resolve():
            source_cfg = json.loads(source_cfg_path.read_text())
            for key in (
                "feature_cols",
                "input_dim",
                "hidden_dim",
                "num_layers",
                "window_size",
                "raw_hz",
                "frame_size_raw",
                "holdoff_frames",
                "ecg_slope_mode",
                "signal_quality_mode",
                "ecg_rotation_angle",
                "ecg_quantile",
                "ecg_norm_window_samples",
                "puls_quantile_low",
                "puls_quantile_high",
            ):
                if source_cfg.get(key) != cfg.get(key):
                    raise ValueError(f"CV archive model/preprocessing mismatch: {key}")
        return folder
    raise FileNotFoundError("Cannot locate the model's verified CV threshold archive")


def grid_path(folder):
    calibrated = folder / "best_threshold_search.csv"
    if calibrated.is_file():
        return calibrated
    ordinary = folder / "best_epoch_threshold_search.csv"
    return (
        ordinary
        if ordinary.is_file()
        else folder / "best_hpo_epoch_threshold_search.csv"
    )


def eligible_rows(rows, cfg, sensitivity_floor=None):
    """Preserve the original global sensitivity floor and the deployed epoch."""
    frame = pd.DataFrame(rows).copy()
    if "trial_id" in frame:
        frame = frame.loc[frame["trial_id"].eq(int(cfg["hpo_trial_id"]))].copy()
    selectable = frame["selectable_threshold"].astype(bool) & frame["threshold"].gt(0)
    best = float(frame.loc[selectable, "exact_sensitivity"].max())
    floor = float(cfg.get("threshold_sensitivity_efficiency", 0.95)) * best
    if sensitivity_floor is not None:
        floor = float(sensitivity_floor)
    mask = selectable & frame["exact_sensitivity"].ge(floor)
    if "epoch" in frame:
        mask &= frame["epoch"].eq(int(cfg["selected_fixed_epoch"]))
    eligible = frame.loc[mask].copy()
    if eligible.empty:
        raise ValueError(
            "No valid candidate at the deployed epoch within the sensitivity floor"
        )
    return eligible, floor


def legacy_fp_threshold(rows, cfg):
    eligible, _ = eligible_rows(rows, cfg)
    if "outer_position_error" in eligible:
        ope = eligible["outer_position_error"].astype(int)
        for key in (
            "unmatched_false_positive",
            "unmatched_false_negative",
            "unique_error_count",
        ):
            eligible[key] = eligible[key] + ope
    return float(
        eligible.sort_values(
            [
                "unmatched_false_positive",
                "unique_error_count",
                "unmatched_false_negative",
                "position_error",
                "threshold",
            ],
            ascending=[True, True, True, True, False],
        ).iloc[0]["threshold"]
    )


def select_fixed(rows, cfg, sensitivity_floor=None):
    eligible, floor = eligible_rows(rows, cfg, sensitivity_floor)
    # Filtering has already enforced the original joint-search floor.
    # Use the equivalent efficiency on this fixed-epoch subset.
    efficiency = floor / float(eligible["exact_sensitivity"].max())
    return select_threshold_from_search_rows(
        eligible.to_dict("records"),
        objective=cfg.get("threshold_objective", "min_error"),
        sensitivity_efficiency=min(1.0, efficiency),
    )


def validation_rows(frames, cfg, thresholds):
    # Prefix every sequence by its fold, so identical patient/sequence labels
    # across independently fitted models never merge into a single signal.
    prepared = []
    for index, frame in enumerate(frames):
        frame = frame.copy()
        frame["sid"] = str(index) + ":" + frame["sid"].astype(str)
        prepared.append(frame)
    probabilities, targets = pack_validation_sequences(
        pd.concat(prepared, ignore_index=True),
        target_col=cfg.get("target_col", "target"),
        holdoff_frames=int(cfg.get("holdoff_frames", cfg.get("min_distance", 10))),
        error_association_frames=event_tolerances(cfg)["error_association_frames"],
    )
    return evaluate_threshold_grid(
        probabilities,
        np.flatnonzero(targets == 1),
        holdoff_frames=int(cfg.get("holdoff_frames", cfg.get("min_distance", 10))),
        **event_tolerances(cfg),
        thresholds=thresholds,
    )


def saved_cv_validation(archive, cfg):
    frames = []
    recorded = json.loads((archive / "cv_config.json").read_text())
    for fold in recorded["folds"]:
        outer = archive / "outer" / fold["test_patient_id"]
        trial = outer
        selection_path = outer / "inner_selection.json"
        if cfg.get("cv_method") == "nested_lopo_hpo":
            trial = outer / "search" / f"trial_{int(cfg['hpo_trial_id']):03d}"
            selection_path = trial / "trial.json"
        selection = json.loads(selection_path.read_text())
        if int(selection["selected_fixed_epoch"]) != int(cfg["selected_fixed_epoch"]):
            raise ValueError(
                "Saved CV probabilities mix epochs; need --rebuild-cv-validation"
            )
        for inner in sorted((trial / "inner").iterdir()):
            frames.append(
                read_validation_analysis_rows(
                    inner / "selected_epoch_validation_predictions.parquet",
                    cfg.get("target_col", "target"),
                )
            )
    if not frames:
        raise ValueError("No saved inner validation probabilities")
    if (
        cfg.get("cv_method") == "nested_lopo_hpo"
        and cfg.get("threshold_selection_source")
        != "all_nested_inner_out_of_fold_full_region"
    ):
        from dkfz.live.hpo_cross_validation import _common_validation_predictions

        frames, _ = _common_validation_predictions(
            frames,
            candidate_window_size=int(cfg["window_size"]),
            maximum_window_size=int(recorded["maximum_candidate_window_size"]),
        )
    return frames


def rebuild_cv_validation(archive, cfg, output, device):
    """Regenerate only training/validation folds, using the recorded seeds."""
    import torch
    from dkfz.live.cross_validation import (
        _train_inner_fold,
        _predict_inner_at_epoch,
        nested_lopo_folds,
        _validate_cached_patient_sources,
    )
    from dkfz.live.training_config import TRAINING_DEFAULTS

    recorded = json.loads((archive / "cv_config.json").read_text())
    kwargs = {**TRAINING_DEFAULTS, **recorded["training"]}
    is_hpo = cfg.get("cv_method") == "nested_lopo_hpo"
    if is_hpo:
        candidate = next(
            item
            for item in recorded["candidates"]
            if int(item["trial_id"]) == int(cfg["hpo_trial_id"])
        )
        kwargs.update(candidate["parameters"])
    # Use the final model's architecture/preprocessing, with the original base seed.
    for key in ("window_size", "hidden_dim", "num_layers", "lr", "batch_size"):
        if kwargs[key] != cfg[key]:
            raise ValueError(f"CV regeneration configuration mismatch: {key}")
    kwargs["num_epochs"] = int(cfg["selected_fixed_epoch"])
    cached = pd.read_parquet(archive / "patient_frames.parquet")
    _validate_cached_patient_sources(
        cached,
        {
            patient: [Path(path) for path in paths]
            for patient, paths in recorded["patient_inputs"].items()
        },
    )
    patient_frames = {
        str(key): group.copy()
        for key, group in cached.groupby("patient_id", sort=False)
    }
    if set(patient_frames) != set(recorded["patient_ids"]):
        raise ValueError("CV patient cache does not match the recorded training cohort")
    frames = []
    for outer in nested_lopo_folds(recorded["patient_ids"]):
        for inner in outer.inner_folds:
            print(
                f"  Rebuilding validation: outer={outer.test_patient_id}, inner={inner.validation_patient_id}, epoch={kwargs['num_epochs']}",
                flush=True,
            )
            result = _train_inner_fold(
                outer_fold=outer,
                inner_fold=inner,
                patient_frames=patient_frames,
                kwargs=kwargs,
                feature_cols=cfg["feature_cols"],
                input_dim=int(cfg["input_dim"]),
                device=torch.device(device),
                outer_dir=output / outer.test_patient_id,
                base_seed=int(recorded["seed"]),
            )
            frames.append(
                _predict_inner_at_epoch(
                    result,
                    selected_epoch=kwargs["num_epochs"],
                    patient_frames=patient_frames,
                    kwargs=kwargs,
                    feature_cols=cfg["feature_cols"],
                    input_dim=int(cfg["input_dim"]),
                    device=torch.device(device),
                )
            )
    if (
        is_hpo
        and cfg.get("threshold_selection_source")
        != "all_nested_inner_out_of_fold_full_region"
    ):
        from dkfz.live.hpo_cross_validation import _common_validation_predictions

        frames, _ = _common_validation_predictions(
            frames,
            candidate_window_size=int(cfg["window_size"]),
            maximum_window_size=int(recorded["maximum_candidate_window_size"]),
        )
    return frames


def process(path, args, rebuilt_cache):
    cfg = json.loads(path.read_text())
    if cfg.get("recovery_head"):
        import hashlib
        from dkfz.live.recovery_calibration import calibrate_config

        audit_id = hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:12]
        return calibrate_config(
            path,
            apply=args.apply,
            max_fp_increase=getattr(args, "recovery_max_fp_increase", None),
            min_extra_fp_per_minute=getattr(
                args, "recovery_min_extra_fp_per_minute", None
            ),
            output=args.output_dir / "recovery" / f"{audit_id}_threshold_search.csv",
            rebuild_cv_validation=getattr(args, "rebuild_cv_validation", False),
            device=getattr(args, "device", "cpu"),
        )
    previous = cfg["threshold"]
    selection = None
    status = "complete"
    reason = ""
    if cfg.get("cv_method"):
        archive = cv_archive(path, cfg)
        grid = pd.read_csv(grid_path(archive))
        fp = legacy_fp_threshold(grid, cfg)
        if "outer_position_error" in grid and grid.get(
            "metrics_schema_version", pd.Series([0])
        ).min() >= (4 if event_tolerances(cfg)["exact_tolerance_frames"] else 3):
            selection = select_fixed(grid, cfg)
        else:
            try:
                try:
                    frames = saved_cv_validation(archive, cfg)
                except (ValueError, FileNotFoundError, KeyError):
                    if not args.rebuild_cv_validation:
                        raise ValueError(
                            "Missing same-epoch oPE data; rerun with --rebuild-cv-validation"
                        )
                    key = (
                        str(archive.resolve()),
                        int(cfg["selected_fixed_epoch"]),
                        cfg.get("hpo_trial_id"),
                    )
                    if key not in rebuilt_cache:
                        rebuilt_cache[key] = rebuild_cv_validation(
                            archive,
                            cfg,
                            args.output_dir
                            / "cv_rebuild"
                            / archive.parent.name
                            / archive.name
                            / f"epoch_{key[1]}_trial_{key[2]}",
                            args.device,
                        )
                    frames = rebuilt_cache[key]
                rows = validation_rows(frames, cfg, np.sort(grid["threshold"].unique()))
                # The original cohort/fixed epoch is unchanged. Its raw validation
                # replay supplies new oPE counts; FP retains the archived calculation.
                selection = select_fixed(
                    rows,
                    cfg,
                    (
                        None
                        if event_tolerances(cfg)["exact_tolerance_frames"]
                        else eligible_rows(grid, cfg)[1]
                    ),
                )
            except (ValueError, FileNotFoundError, KeyError) as error:
                if args.rebuild_cv_validation:
                    raise
                status, reason = "needs_cv_validation_rebuild", str(error)
    else:
        name = (
            "tuned_all_events.parquet"
            if path.name == "tuned_config.json"
            else "train_all_events.parquet"
        )
        frames = [
            read_validation_analysis_rows(
                path.parent / name, cfg.get("target_col", "target")
            )
        ]
        rows = validation_rows(frames, cfg, np.linspace(0, 1, 1001, dtype=np.float32))
        selection = select_fixed(rows, cfg)
        fp = selection.fp_threshold
    der = None if selection is None else selection.der_threshold
    if args.apply:
        backup = path.with_suffix(path.suffix + ".before_threshold_policies")
        if not backup.exists():
            backup.write_bytes(path.read_bytes())
        cfg.update(fp_threshold=fp, der_threshold=der)
        cfg.setdefault(
            "manual_threshold",
            previous if cfg.get("threshold_selection_mode") == "manual" else None,
        )
        cfg["threshold_backfill_status"] = status
        if selection is not None:
            cfg.update(selection.policy_metadata())
            cfg["fp_threshold"] = fp
            cfg["threshold"] = selection.threshold
            cfg["threshold_policy_version"] = 3
            cfg["metrics_schema_version"] = METRICS_SCHEMA_VERSION
            cfg.update({key: selection.metrics[key] for key in event_tolerances(cfg)})
            cfg.update(
                {
                    f"threshold_validation_{key}": value
                    for key, value in selection.metrics.items()
                }
            )
            cfg["outer_error_association_frames"] = outer_association_frames(
                cfg.get("raw_hz", 400), cfg.get("frame_size_raw", 8)
            )
            cfg["outer_error_association_ms"] = (
                cfg["outer_error_association_frames"]
                * cfg.get("frame_size_raw", 8)
                / cfg.get("raw_hz", 400)
                * 1000
            )
            report = (
                path.parent / "metrics" / f"{path.stem}_backfilled_threshold_search.csv"
            )
            report.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(selection.search_rows).to_csv(report, index=False)
            cfg["threshold_backfill_search_file"] = str(report)
        write_json_atomic(path, cfg)
    return dict(
        config=str(path),
        previous_threshold=previous,
        fp_threshold=fp,
        der_threshold=der,
        status=status,
        reason=reason,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "roots",
        nargs="*",
        type=Path,
        default=[Path("models"), Path("models_noCV"), Path("finished_models")],
    )
    parser.add_argument(
        "--model", action="append", help="Restrict to a modelname (repeatable)"
    )
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--recovery-max-fp-increase",
        type=float,
        help="Override the relative validation FP allowance for two-output models; primary settings remain fixed.",
    )
    parser.add_argument(
        "--rebuild-cv-validation",
        action="store_true",
        help="Reconstruct missing CV validation: replay saved checkpoints for two-output models; retrain inner folds for old single-output grids lacking oPE",
    )
    parser.add_argument(
        "--recovery-min-extra-fp-per-minute",
        type=float,
        default=None,
        help="Override the minimum additional recovery FP allowance per validation minute; default 0.5.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for explicitly requested CV regeneration, e.g. cuda",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("dbg/threshold_backfill")
    )
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(
        {
            p
            for root in args.roots
            if root.exists()
            for name in ("train_config.json", "tuned_config.json")
            for p in root.rglob(name)
            if "outer" not in p.parts and "inner" not in p.parts
        }
    )
    reports, rebuilt = [], {}
    for path in paths:
        cfg = json.loads(path.read_text())
        if args.model and cfg.get("modelname") not in args.model:
            continue
        print(f"Checking {path}", flush=True)
        try:
            record = process(path, args, rebuilt)
        except (ValueError, FileNotFoundError, KeyError) as error:
            record = dict(config=str(path), status="error", reason=str(error))
        reports.append(record)
        print(record, flush=True)
    pd.DataFrame(reports).to_csv(args.output_dir / "report.csv", index=False)
    if not reports:
        parser.error("No matching model configurations found")
    return int(any(record["status"] != "complete" for record in reports))


if __name__ == "__main__":
    raise SystemExit(main())
