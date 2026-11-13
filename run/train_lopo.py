#!/usr/bin/env python3
# Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Developed at the German Cancer Research Center (DKFZ).

"""Run nested leave-one-patient-out training and evaluation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dkfz.live.cross_validation import (  # noqa: E402
    ensure_output_outside_data,
    resolve_patient_groups,
    run_nested_lopo,
)
from dkfz.live.cv_paths import (  # noqa: E402
    NESTED_LOPO_DIRNAME,
    cv_base_modelname,
    cv_modelname,
    default_cv_output_dir,
    prepare_cv_output_dir,
)
from dkfz.live.training_config import (  # noqa: E402
    TRAINING_DEFAULTS,
    load_training_config_values,
    training_kwargs_from_config_and_values,
)
from dkfz.live.training_reuse import (  # noqa: E402
    check_run_available,
    existing_run_status,
    lock_training_run,
    record_training_request,
    training_request,
)
from dkfz.live.training_seed import validate_training_seed  # noqa: E402


def _setting(value: str):
    """Parse JSON-like --set values, falling back to a plain string."""
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _setting_overrides(values: list[str]) -> dict:
    overrides = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"--set expects NAME=VALUE, got {item!r}.")
        key, raw_value = item.split("=", 1)
        if key not in TRAINING_DEFAULTS:
            raise ValueError(
                f"Unknown training setting {key!r}; expected a run/train.py setting."
            )
        if key in {"inputs", "threshold_inputs"}:
            raise ValueError(
                f"{key!r} is controlled by LOPO patient grouping and cannot be set."
            )
        overrides[key] = _setting(raw_value)
    return overrides


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Nested leave-one-patient-out validation. Each positional input is "
            "one patient directory. Alternatively use --patient-root or "
            "--patient-manifest."
        )
    )
    parser.add_argument(
        "patient_inputs",
        nargs="*",
        help="Patient directories; shell globs are accepted as one or more values.",
    )
    patient_source = parser.add_mutually_exclusive_group()
    patient_source.add_argument(
        "--patient-root",
        help="Directory whose immediate parquet-containing subdirectories are patients.",
    )
    patient_source.add_argument(
        "--patient-manifest",
        help="JSON mapping patient IDs to input paths or glob lists.",
    )
    parser.add_argument("--config", help="Existing train_config.json to reuse.")
    parser.add_argument(
        "--modelname",
        help="Base model name. CV artifacts use <modelname>_cv.",
    )
    parser.add_argument(
        "--output-dir",
        help="Output root. Defaults to models/<modelname>_cv/nested_lopo.",
    )
    parser.add_argument("--prepped-file", help="Patient-framed parquet cache path.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Force retraining even when an identical completed run exists.",
    )
    parser.add_argument(
        "--check-existing",
        action="store_true",
        help="Print a read-only JSON reuse/replacement decision without training.",
    )
    parser.add_argument("--run-lock-fd", type=int, help=argparse.SUPPRESS)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base fold seed. Defaults to the config value or 42.",
    )
    parser.add_argument("--lr", type=float)
    parser.add_argument("--hidden-dim", type=int)
    parser.add_argument("--num-layers", type=int)
    parser.add_argument("--window-size", type=int)
    parser.add_argument("--num-epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--prep-workers", type=int)
    parser.add_argument(
        "--threshold-objective",
        choices=("min_error", "min_fp"),
    )
    parser.add_argument("--threshold-sensitivity-efficiency", type=float)
    parser.add_argument("--amp", action="store_true", dest="use_amp")
    parser.add_argument("--no-amp", action="store_false", dest="use_amp")
    parser.set_defaults(use_amp=None)
    parser.add_argument(
        "--save-prepped",
        action="store_true",
        dest="save_prepped",
        help="Save patient frames under the LOPO output tree.",
    )
    parser.add_argument(
        "--no-save-prepped",
        action="store_false",
        dest="save_prepped",
    )
    parser.set_defaults(save_prepped=None)
    parser.add_argument(
        "--use-prepped",
        action="store_true",
        dest="use_prepped",
        help="Reuse and validate an existing LOPO patient-frame cache.",
    )
    parser.add_argument(
        "--no-use-prepped",
        action="store_false",
        dest="use_prepped",
    )
    parser.set_defaults(use_prepped=None)
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help=(
            "Override any other run/train.py setting. Values accept JSON syntax; "
            "unquoted non-JSON values are treated as strings. May be repeated."
        ),
    )
    from dkfz.live.training_config import add_recovery_training_arguments

    add_recovery_training_arguments(parser)
    from dkfz.live.training_config import add_prediction_training_arguments

    add_prediction_training_arguments(parser)
    return parser


def resolve_lopo_training_kwargs(args: argparse.Namespace) -> dict:
    explicit = {
        key: getattr(args, key, None)
        for key in (
            "plots",
            "prediction_mode",
            "prediction_delay_frames",
            "signal_quality_puls_flat_window_frames",
            "signal_quality_puls_median_window_frames",
        )
    }
    explicit.update(
        {
            "config": args.config,
            "recovery_head": getattr(args, "recovery_head", None),
            "recovery_max_fp_increase": getattr(args, "recovery_max_fp_increase", None),
            "recovery_min_extra_fp_per_minute": getattr(
                args, "recovery_min_extra_fp_per_minute", None
            ),
            "inputs": [],
            "modelname": args.modelname,
            "lr": args.lr,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "window_size": args.window_size,
            "num_epochs": args.num_epochs,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "prep_workers": args.prep_workers,
            "threshold_objective": args.threshold_objective,
            "threshold_sensitivity_efficiency": args.threshold_sensitivity_efficiency,
            "use_amp": args.use_amp,
            "save_prepped": args.save_prepped,
            "use_prepped": args.use_prepped,
            "prepped_file": args.prepped_file,
            "threshold_inputs": [],
        }
    )
    setting_overrides = _setting_overrides(args.set)
    explicit.update(setting_overrides)
    config_values = load_training_config_values(args.config)
    kwargs = training_kwargs_from_config_and_values(
        explicit,
        config_values=config_values,
    )
    # Parser defaults use None for "unspecified", but --set NAME=null is an
    # explicit request to clear a saved setting, such as a manual threshold.
    kwargs.update(
        {key: value for key, value in setting_overrides.items() if value is None}
    )
    kwargs["base_modelname"] = cv_base_modelname(str(kwargs["modelname"]))
    kwargs["modelname"] = cv_modelname(str(kwargs["modelname"]))
    kwargs["cv_method"] = "nested_lopo"
    kwargs["inputs"] = []
    kwargs["threshold_inputs"] = []
    # Saved fitted stats belong to a previous split. Every LOPO fold refits them
    # from its own training patients.
    kwargs["signal_quality_stats"] = None
    return kwargs


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        kwargs = resolve_lopo_training_kwargs(args)
        validate_training_seed(kwargs["seed"])
        for key in (
            "hidden_dim",
            "num_layers",
            "window_size",
            "num_epochs",
            "batch_size",
        ):
            value = kwargs[key]
            if isinstance(value, bool) or int(value) != float(value) or int(value) < 1:
                raise ValueError(f"{key} must be a positive integer")
        if not math.isfinite(float(kwargs["lr"])) or float(kwargs["lr"]) <= 0:
            raise ValueError("lr must be finite and positive")
        if kwargs["threshold_objective"] not in {"min_error", "min_fp"}:
            raise ValueError("Unsupported threshold_objective")
        if not 0 < float(kwargs["threshold_sensitivity_efficiency"]) <= 1:
            raise ValueError("threshold_sensitivity_efficiency must be in (0, 1]")
        if kwargs.get("recovery_only"):
            raise ValueError(
                "Frozen recovery training requires run/train_recovery_lopo.py"
            )
        patient_groups = resolve_patient_groups(
            patient_inputs=args.patient_inputs,
            patient_root=args.patient_root,
            patient_manifest=args.patient_manifest,
        )
        output_dir = Path(
            args.output_dir
            or default_cv_output_dir(kwargs["modelname"], NESTED_LOPO_DIRNAME)
        )
        output_dir = ensure_output_outside_data(output_dir)
        prepped_file = Path(args.prepped_file or output_dir / "patient_frames.parquet")
        ensure_output_outside_data(prepped_file)
        kwargs["prepped_file"] = str(prepped_file)
        if args.check_existing:
            check_run_available(output_dir)
            if (
                args.overwrite
                or not output_dir.exists()
                or not any(output_dir.iterdir())
            ):
                print(
                    json.dumps(
                        dict(
                            action="train",
                            reason=(
                                "forced by --overwrite" if args.overwrite else "new run"
                            ),
                        )
                    )
                )
                return 0
            request = training_request(kwargs, patient_groups)
            print(
                json.dumps(
                    existing_run_status(output_dir, request, force=args.overwrite)
                )
            )
            return 0
        with lock_training_run(output_dir, inherited_fd=args.run_lock_fd):
            request = training_request(kwargs, patient_groups)
            status = existing_run_status(output_dir, request, force=args.overwrite)
            if status["action"] == "reuse":
                print(
                    f"Skipping {kwargs['modelname']}: {status['reason']}; use --overwrite to retrain."
                )
                return 0
            print(f"Training {kwargs['modelname']}: {status['reason']}")
            output_dir = prepare_cv_output_dir(
                output_dir,
                overwrite=args.overwrite or status["action"] == "replace",
                require_empty=False,
            )
            summary = run_nested_lopo(
                patient_groups=patient_groups,
                kwargs=kwargs,
                output_dir=output_dir,
                prepped_file=prepped_file,
                seed=int(kwargs["seed"]),
            )
            record_training_request(output_dir, request)
    except (FileNotFoundError, KeyError, TypeError, ValueError, OverflowError) as exc:
        parser.error(str(exc))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
