#!/usr/bin/env python3
# Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Developed at the German Cancer Research Center (DKFZ).

"""Create fixed-threshold dashboard and error PDFs for an offline prediction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dkfz.live.io import repo_relative_path  # noqa: E402
from dkfz.live.prediction_cache import read_prediction_frame  # noqa: E402
from dkfz.live.threshold_analysis import (  # noqa: E402
    classify_validation_events,
    fixed_threshold_analysis,
    write_error_examples,
    write_threshold_dashboard,
)


def threshold_grid(step: float, configured_threshold: float) -> np.ndarray:
    """Return an inclusive grid containing the exact configured threshold."""
    step = float(step)
    if not 0.0 < step <= 1.0:
        raise ValueError("threshold_step must be greater than 0 and at most 1")
    values = np.append(np.arange(0.0, 1.0, step, dtype=np.float64), 1.0)
    values = np.append(values, float(configured_threshold))
    return np.unique(np.clip(values, 0.0, 1.0)).astype(np.float32)


def pack_prediction_sequences(
    df: pd.DataFrame,
    *,
    target_col: str,
    holdoff_frames: int,
    error_association_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Pack independent live sequences with zero gaps for threshold sweeps."""
    required = {"prob", target_col}
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"Prediction log is missing column(s): {', '.join(missing)}")

    from dkfz.live.eval_metrics import independent_prediction_groups

    groups = independent_prediction_groups(df)
    gap_len = int(max(holdoff_frames + 1, 2 * error_association_frames + 1, 20))
    probability_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    for _sid, group in groups:
        values = group["prob"].to_numpy(dtype=np.float32)
        if "recovery_prob" in group:
            recovery = group["recovery_prob"].to_numpy(dtype=np.float32).copy()
            recovery[0] = 0.0
            values = np.column_stack((values, recovery))
        probability_parts.extend(
            (
                values,
                np.zeros(
                    (gap_len, 2) if values.ndim == 2 else gap_len, dtype=np.float32
                ),
            )
        )
        target_parts.extend(
            (
                group[target_col].to_numpy(dtype=np.float32),
                np.zeros(gap_len, dtype=np.float32),
            )
        )
    if not probability_parts:
        raise ValueError("Prediction log contains no rows")
    return np.concatenate(probability_parts), np.concatenate(target_parts)


def prediction_analysis_settings(config: dict, metrics: dict) -> dict:
    """Resolve the recorded threshold and event settings for a prediction."""
    threshold = metrics.get("threshold", config.get("threshold"))
    if threshold is None:
        raise ValueError("Prediction metrics and model config contain no threshold")
    from dkfz.live.event_matching import event_tolerances

    policy = event_tolerances(config)
    recorded = metrics.get("event_position", {})
    if (
        int(recorded.get("metrics_schema_version", 0)) >= 4
        or policy["exact_tolerance_frames"] == 0
    ):
        policy.update({k: int(recorded[k]) for k in policy if k in recorded})
    return {
        **(
            {
                "recovery_threshold": metrics.get(
                    "recovery_threshold",
                    (
                        config.get("recovery_threshold")
                        if config.get("recovery_enabled", True)
                        else None
                    ),
                )
            }
            if config.get("recovery_head")
            else {}
        ),
        "target_col": str(
            metrics.get("target_col") or config.get("target_col", "target")
        ),
        "threshold": float(threshold),
        "holdoff_frames": int(
            config.get("holdoff_frames", config.get("min_distance", 10))
        ),
        **policy,
        "sensitivity_efficiency": float(
            config.get("threshold_sensitivity_efficiency", 0.95)
        ),
    }


def create_pooled_prediction_dashboard(
    *,
    prediction_logs: list[Path],
    metrics_jsons: list[Path],
    config_path: Path,
    dashboard_path: Path,
    threshold_step: float = 0.001,
) -> None:
    """Write a pooled dashboard and three standalone plots, preserving boundaries."""
    if not prediction_logs or len(prediction_logs) != len(metrics_jsons):
        raise ValueError(
            "Provide one metrics JSON per prediction log, and at least one log"
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    settings = None
    probability_parts = []
    target_parts = []
    for prediction_log, metrics_json in zip(prediction_logs, metrics_jsons):
        metrics = json.loads(metrics_json.read_text(encoding="utf-8"))
        input_settings = prediction_analysis_settings(config, metrics)
        if settings is not None and input_settings != settings:
            raise ValueError(
                f"Cannot pool predictions with different analysis settings: {metrics_json}"
            )
        settings = input_settings
        # Pack each file separately: sid values can repeat across recordings.
        probabilities, targets = pack_prediction_sequences(
            read_prediction_frame(prediction_log),
            target_col=settings["target_col"],
            holdoff_frames=settings["holdoff_frames"],
            error_association_frames=settings["error_association_frames"],
        )
        probability_parts.append(probabilities)
        target_parts.append(targets)

    analysis_settings = {
        key: value for key, value in settings.items() if key != "target_col"
    }
    selection = fixed_threshold_analysis(
        probability_parts[0],
        np.flatnonzero(target_parts[0] == 1),
        additional_sequences=(
            (probabilities, np.flatnonzero(targets == 1))
            for probabilities, targets in zip(probability_parts[1:], target_parts[1:])
        ),
        **analysis_settings,
        thresholds=threshold_grid(threshold_step, settings["threshold"]),
    )
    dashboard_path.parent.mkdir(parents=True, exist_ok=True)
    write_threshold_dashboard(
        dashboard_path,
        selection=selection,
        presentation="fixed",
        separate_plots=True,
    )


def create_prediction_plots(
    *,
    prediction_log: Path,
    config_path: Path,
    metrics_json: Path,
    dashboard_path: Path,
    errors_path: Path,
    events_path: Path,
    threshold_step: float = 0.001,
    max_error_plots: int = 12,
    error_window_frames: int = 25,
) -> None:
    """Write fixed-threshold analysis artifacts without changing model files."""
    config = json.loads(config_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_json.read_text(encoding="utf-8"))
    prediction_df = read_prediction_frame(prediction_log)

    settings = prediction_analysis_settings(config, metrics)
    target_col = settings["target_col"]
    threshold = settings["threshold"]
    holdoff_frames = settings["holdoff_frames"]
    error_association_frames = settings["error_association_frames"]
    sensitivity_efficiency = settings["sensitivity_efficiency"]

    probabilities, targets = pack_prediction_sequences(
        prediction_df,
        target_col=target_col,
        holdoff_frames=holdoff_frames,
        error_association_frames=error_association_frames,
    )
    selection = fixed_threshold_analysis(
        probabilities,
        np.flatnonzero(targets == 1),
        threshold=threshold,
        sensitivity_efficiency=sensitivity_efficiency,
        holdoff_frames=holdoff_frames,
        error_association_frames=error_association_frames,
        exact_tolerance_frames=settings["exact_tolerance_frames"],
        outer_error_association_frames=settings["outer_error_association_frames"],
        thresholds=threshold_grid(threshold_step, threshold),
        recovery_threshold=settings.get("recovery_threshold"),
    )
    event_df = classify_validation_events(
        prediction_df,
        threshold=threshold,
        target_col=target_col,
        holdoff_frames=holdoff_frames,
        error_association_frames=error_association_frames,
        exact_tolerance_frames=settings["exact_tolerance_frames"],
        outer_error_association_frames=settings["outer_error_association_frames"],
        recovery_threshold=settings.get("recovery_threshold"),
    )

    for output_path in (dashboard_path, errors_path, events_path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
    event_df.to_csv(events_path, index=False)
    write_threshold_dashboard(
        dashboard_path,
        selection=selection,
        presentation="fixed",
        separate_plots=False,
    )
    write_error_examples(
        errors_path,
        validation_df=prediction_df,
        event_df=event_df,
        threshold=threshold,
        target_col=target_col,
        max_plots=max_error_plots,
        window_frames=error_window_frames,
        recovery_threshold=settings.get("recovery_threshold"),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-log", type=Path, action="append", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--metrics-json", type=Path, action="append", required=True)
    parser.add_argument("--dashboard", type=Path, required=True)
    parser.add_argument("--errors", type=Path)
    parser.add_argument("--events", type=Path)
    parser.add_argument(
        "--pooled",
        action="store_true",
        help="Write a pooled dashboard and three standalone plots; repeat --prediction-log and --metrics-json in matching order.",
    )
    parser.add_argument("--threshold-step", type=float, default=0.001)
    parser.add_argument("--max-error-plots", type=int, default=12)
    parser.add_argument("--error-window-frames", type=int, default=25)
    args = parser.parse_args()

    if args.max_error_plots < 0:
        parser.error("--max-error-plots must be zero or positive")
    if args.error_window_frames < 1:
        parser.error("--error-window-frames must be at least 1")
    if not args.pooled:
        if len(args.prediction_log) != 1 or len(args.metrics_json) != 1:
            parser.error("Multiple prediction logs require --pooled")
        if args.errors is None or args.events is None:
            parser.error("--errors and --events are required unless --pooled is used")
    try:
        if args.pooled:
            create_pooled_prediction_dashboard(
                prediction_logs=args.prediction_log,
                metrics_jsons=args.metrics_json,
                config_path=args.config,
                dashboard_path=args.dashboard,
                threshold_step=args.threshold_step,
            )
            print(
                f"Wrote pooled dashboard and standalone plots to {repo_relative_path(args.dashboard.parent)}"
            )
            return 0
        create_prediction_plots(
            prediction_log=args.prediction_log[0],
            config_path=args.config,
            metrics_json=args.metrics_json[0],
            dashboard_path=args.dashboard,
            errors_path=args.errors,
            events_path=args.events,
            threshold_step=args.threshold_step,
            max_error_plots=args.max_error_plots,
            error_window_frames=args.error_window_frames,
        )
    except (FileNotFoundError, KeyError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))

    print(f"Wrote dashboard to {repo_relative_path(args.dashboard)}")
    print(f"Wrote error examples to {repo_relative_path(args.errors)}")
    print(f"Wrote event classifications to {repo_relative_path(args.events)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
