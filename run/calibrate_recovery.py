#!/usr/bin/env python3
# Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Calibrate recovery from validation scores while keeping primary policy fixed."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dkfz.live.recovery_calibration import calibrate_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--validation-log",
        type=Path,
        action="append",
        help="Annotated validation log with prob and recovery_prob; repeatable. Defaults to saved training/CV validation artifacts.",
    )
    parser.add_argument("--max-fp-increase", type=float, default=None)
    parser.add_argument(
        "--min-extra-fp-per-minute",
        type=float,
        default=None,
        help="Minimum additional FP allowance per validation minute; config/default 0.5.",
    )
    parser.add_argument(
        "--rebuild-cv-validation",
        action="store_true",
        help="Replay existing inner checkpoints at the deployment epoch when logs are unavailable; no retraining.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for explicitly requested validation replay.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Threshold-search CSV; defaults alongside the config.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Save the selected recovery policy to config.",
    )
    args = parser.parse_args()
    try:
        report = calibrate_config(
            args.config,
            logs=args.validation_log,
            max_fp_increase=args.max_fp_increase,
            min_extra_fp_per_minute=args.min_extra_fp_per_minute,
            output=args.output,
            apply=args.apply,
            rebuild_cv_validation=args.rebuild_cv_validation,
            device=args.device,
        )
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))
    print(
        f"Recovery threshold: {report['recovery_threshold']}; audit: {report['audit']}; applied: {args.apply}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
