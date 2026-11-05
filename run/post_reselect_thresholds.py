# Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Developed at the German Cancer Research Center (DKFZ).

"""Analyze and reselect runtime thresholds from saved validation probabilities.

The script scans ``models/`` for complete training and optional fine-tuning
artifacts, then applies the version-2 exact-frame event contract and the chosen
``min_error`` or ``min_fp`` policy. Analysis reports are always written;
runtime configuration changes require an explicit apply option.

Examples:

    .venv/bin/python run/post_reselect_thresholds.py
    .venv/bin/python run/post_reselect_thresholds.py \
        --sensitivity-efficiency 0.95 --apply-auto
    .venv/bin/python run/post_reselect_thresholds.py \
        --model target_r --apply-threshold 0.598
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dkfz.core.network_helpers import (  # noqa: E402
    ThresholdSelection,
    select_threshold,
    streaming_trigger_peaks,
)
from dkfz.live.io import repo_relative_path  # noqa: E402
from dkfz.live.event_matching import (  # noqa: E402
    EXACT_TOLERANCE_FRAMES,
    METRICS_SCHEMA_VERSION,
    compute_event_position_metrics,
)
from dkfz.live.threshold_analysis import (  # noqa: E402
    classify_validation_events,
    operating_regions,
    selection_rows,
    write_error_examples,
    write_threshold_dashboard,
)


@dataclass(frozen=True)
class VariantArtifacts:
    modelname: str
    variant: str
    config_path: Path
    validation_path: Path
    metrics_dir: Path

    @property
    def report_prefix(self) -> str:
        return "reselect" if self.variant == "best" else "reselect_tuned"

    @property
    def search_path(self) -> Path:
        return self.metrics_dir / f"{self.report_prefix}_threshold_search.csv"

    @property
    def selection_path(self) -> Path:
        return self.metrics_dir / f"{self.report_prefix}_threshold_selection.json"

    @property
    def analysis_path(self) -> Path:
        return self.metrics_dir / f"{self.report_prefix}_threshold_analysis.csv"

    @property
    def analysis_json_path(self) -> Path:
        return self.metrics_dir / f"{self.report_prefix}_threshold_analysis.json"

    @property
    def dashboard_path(self) -> Path:
        return self.metrics_dir / f"{self.report_prefix}_threshold_dashboard.pdf"

    @property
    def events_path(self) -> Path:
        return self.metrics_dir / f"{self.report_prefix}_threshold_events.csv"

    @property
    def errors_path(self) -> Path:
        return self.metrics_dir / f"{self.report_prefix}_threshold_errors.pdf"

    @property
    def history_path(self) -> Path:
        return self.metrics_dir / f"{self.report_prefix}_threshold_history.jsonl"


def discover_variants(
    models_dir: Path,
    *,
    modelnames: set[str] | None,
    variants: set[str],
) -> tuple[list[VariantArtifacts], list[str]]:
    """Return complete model variants and human-readable skip messages."""
    discovered: list[VariantArtifacts] = []
    skipped: list[str] = []

    for model_dir in sorted(path for path in models_dir.iterdir() if path.is_dir()):
        modelname = model_dir.name
        if modelnames is not None and modelname not in modelnames:
            continue

        candidates = []
        if "best" in variants:
            candidates.append(
                VariantArtifacts(
                    modelname=modelname,
                    variant="best",
                    config_path=model_dir / "train_config.json",
                    validation_path=model_dir / "train_all_events.parquet",
                    metrics_dir=model_dir / "metrics",
                )
            )
        if "tuned" in variants:
            candidates.append(
                VariantArtifacts(
                    modelname=modelname,
                    variant="tuned",
                    config_path=model_dir / "tuning/tuned_config.json",
                    validation_path=model_dir / "tuning/tuned_all_events.parquet",
                    metrics_dir=model_dir / "metrics",
                )
            )

        for candidate in candidates:
            missing = [
                path
                for path in (candidate.config_path, candidate.validation_path)
                if not path.is_file()
            ]
            if not missing:
                discovered.append(candidate)
            elif any(
                path.exists()
                for path in (candidate.config_path, candidate.validation_path)
            ):
                missing_text = ", ".join(repo_relative_path(path) for path in missing)
                skipped.append(
                    f"{modelname}:{candidate.variant}: missing {missing_text}"
                )

    if modelnames is not None:
        found_names = {artifact.modelname for artifact in discovered}
        known_names = {path.name for path in models_dir.iterdir() if path.is_dir()}
        for modelname in sorted(modelnames - known_names):
            skipped.append(f"{modelname}: model directory not found")
        for modelname in sorted((modelnames & known_names) - found_names):
            if not any(message.startswith(f"{modelname}:") for message in skipped):
                skipped.append(f"{modelname}: no complete requested variant")

    return discovered, skipped


def threshold_grid(step: float) -> np.ndarray:
    """Return an inclusive threshold grid from zero to one."""
    step = float(step)
    if not 0.0 < step <= 1.0:
        raise ValueError("threshold_step must be greater than 0 and at most 1.")
    values = np.append(np.arange(0.0, 1.0, step, dtype=np.float64), 1.0)
    values = np.clip(values, 0.0, 1.0)
    return np.unique(values).astype(np.float32)


def pack_validation_sequences(
    df: pd.DataFrame,
    *,
    target_col: str,
    holdoff_frames: int,
    error_association_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Pack independent sequences with gaps matching training evaluation."""
    required = {"prob", target_col}
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"Validation parquet is missing column(s): {', '.join(missing)}")

    groups = df.groupby("sid", sort=False) if "sid" in df.columns else [(0, df)]
    gap_len = int(max(holdoff_frames + 1, 2 * error_association_frames + 1, 20))
    probability_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []

    for _sid, group in groups:
        probabilities = group["prob"].to_numpy(dtype=np.float32)
        targets = group[target_col].to_numpy(dtype=np.float32)
        probability_parts.extend((probabilities, np.zeros(gap_len, dtype=np.float32)))
        target_parts.extend((targets, np.zeros(gap_len, dtype=np.float32)))

    if not probability_parts:
        raise ValueError("Validation parquet contains no rows.")
    return np.concatenate(probability_parts), np.concatenate(target_parts)


def read_validation_analysis_rows(path: Path, target_col: str) -> pd.DataFrame:
    """Read only columns needed for threshold replay and error visualization."""
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError(
            "Threshold analysis requires pyarrow; install the project analysis extra."
        ) from exc
    available = set(parquet.read_schema(path).names)
    required = {"sid", "prob", target_col}
    missing = sorted(required - available)
    if missing:
        raise KeyError(f"Validation parquet is missing column(s): {', '.join(missing)}")
    optional = ("frame_idx", "raw_idx", "time_ms", "source_file")
    columns = [
        "sid",
        "prob",
        target_col,
        *(name for name in optional if name in available),
    ]
    return pd.read_parquet(path, columns=columns)


def metrics_at_threshold(
    probabilities: np.ndarray,
    targets: np.ndarray,
    *,
    threshold: float,
    holdoff_frames: int,
    error_association_frames: int,
) -> dict[str, float | int]:
    """Evaluate one existing threshold through the common event contract."""
    target_peaks = np.flatnonzero(targets == 1)
    predicted_peaks = streaming_trigger_peaks(
        probabilities,
        threshold=threshold,
        holdoff_frames=holdoff_frames,
    )
    return compute_event_position_metrics(
        predicted_peaks,
        target_peaks,
        error_association_frames=error_association_frames,
    )


def write_json_atomic(path: Path, data: dict) -> None:
    """Write JSON through a sibling temporary file before replacement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(path)


def append_jsonl_atomic(path: Path, record: dict) -> None:
    """Append one JSONL record through an atomic sibling replacement."""
    previous = path.read_text(encoding="utf-8") if path.is_file() else ""
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        previous + json.dumps(record, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def apply_selection(
    artifacts: VariantArtifacts,
    config: dict,
    selection: ThresholdSelection,
    applied_threshold: float,
    applied_metrics: dict[str, float | int],
    selection_mode: str,
    previous_threshold: float,
    previous_metrics: dict[str, float | int],
    holdoff_frames: int,
    error_association_frames: int,
) -> None:
    """Persist the selected runtime threshold and auditable search artifacts."""
    artifacts.metrics_dir.mkdir(parents=True, exist_ok=True)
    search = selection_rows(selection)
    search["applied"] = np.isclose(search["threshold"], applied_threshold)
    search.to_csv(artifacts.search_path, index=False)

    for legacy_key in (
        "temp_tolerance",
        "threshold_min_recall",
        "threshold_min_exact_sensitivity",
        "threshold_constraint_met",
        "threshold_validation_true_positive",
        "threshold_validation_false_positive",
        "threshold_validation_false_negative",
        "threshold_validation_precision",
        "threshold_validation_recall",
        "threshold_validation_f1",
        "threshold_validation_detection_error",
    ):
        config.pop(legacy_key, None)

    selection_document = {
        "modelname": artifacts.modelname,
        "model_variant": artifacts.variant,
        "source_validation_file": repo_relative_path(artifacts.validation_path),
        "previous_threshold": previous_threshold,
        "previous_exact_sensitivity": float(previous_metrics["exact_sensitivity"]),
        "previous_exact_positive_predictive_value": float(
            previous_metrics["exact_positive_predictive_value"]
        ),
        "previous_unique_detection_error": float(
            previous_metrics["unique_detection_error"]
        ),
        "previous_position_error": int(previous_metrics["position_error"]),
        "previous_unmatched_false_positive": int(
            previous_metrics["unmatched_false_positive"]
        ),
        "previous_unmatched_false_negative": int(
            previous_metrics["unmatched_false_negative"]
        ),
        "threshold": applied_threshold,
        "threshold_selection_mode": selection_mode,
        "recommended_threshold": selection.threshold,
        "threshold_objective": selection.objective,
        "threshold_sensitivity_efficiency": selection.sensitivity_efficiency,
        "threshold_best_attainable_exact_sensitivity": (
            selection.best_attainable_sensitivity
        ),
        "threshold_effective_exact_sensitivity_floor": (
            selection.effective_sensitivity_floor
        ),
        "holdoff_frames": holdoff_frames,
        "metrics_schema_version": METRICS_SCHEMA_VERSION,
        "exact_tolerance_frames": EXACT_TOLERANCE_FRAMES,
        "error_association_frames": error_association_frames,
        **{f"validation_{key}": value for key, value in applied_metrics.items()},
        "threshold_search_file": repo_relative_path(artifacts.search_path),
        "threshold_analysis_file": repo_relative_path(artifacts.analysis_json_path),
        "threshold_dashboard_file": repo_relative_path(artifacts.dashboard_path),
        "threshold_error_examples_file": repo_relative_path(artifacts.errors_path),
    }
    write_json_atomic(artifacts.selection_path, selection_document)

    history_record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "selection_mode": selection_mode,
        "previous_threshold": previous_threshold,
        "applied_threshold": applied_threshold,
        "recommended_threshold": selection.threshold,
        "objective": selection.objective,
        "sensitivity_efficiency": selection.sensitivity_efficiency,
        "best_attainable_exact_sensitivity": selection.best_attainable_sensitivity,
        "effective_exact_sensitivity_floor": selection.effective_sensitivity_floor,
        "previous_metrics": previous_metrics,
        "applied_metrics": applied_metrics,
    }
    append_jsonl_atomic(artifacts.history_path, history_record)

    config.update(
        {
            "threshold": applied_threshold,
            **selection.policy_metadata(),
            "threshold_selection_mode": selection_mode,
            "threshold_recommended": selection.threshold,
            "threshold_objective": selection.objective,
            "threshold_sensitivity_efficiency": selection.sensitivity_efficiency,
            "threshold_best_attainable_exact_sensitivity": (
                selection.best_attainable_sensitivity
            ),
            "threshold_effective_exact_sensitivity_floor": (
                selection.effective_sensitivity_floor
            ),
            **{
                f"threshold_validation_{key}": value
                for key, value in applied_metrics.items()
            },
            "metrics_schema_version": METRICS_SCHEMA_VERSION,
            "exact_tolerance_frames": EXACT_TOLERANCE_FRAMES,
            "error_association_frames": error_association_frames,
            f"{artifacts.report_prefix}_threshold_search_file": repo_relative_path(
                artifacts.search_path
            ),
            f"{artifacts.report_prefix}_threshold_selection_file": repo_relative_path(
                artifacts.selection_path
            ),
            f"{artifacts.report_prefix}_threshold_history_file": repo_relative_path(
                artifacts.history_path
            ),
            f"{artifacts.report_prefix}_threshold_analysis_file": repo_relative_path(
                artifacts.analysis_json_path
            ),
            f"{artifacts.report_prefix}_threshold_dashboard_file": repo_relative_path(
                artifacts.dashboard_path
            ),
            f"{artifacts.report_prefix}_threshold_error_examples_file": repo_relative_path(
                artifacts.errors_path
            ),
        }
    )
    write_json_atomic(artifacts.config_path, config)


def reselect_variant(
    artifacts: VariantArtifacts,
    *,
    sensitivity_efficiency: float,
    objective: str,
    thresholds: np.ndarray,
    apply_auto: bool,
    apply_threshold: float | None,
    max_error_plots: int,
    error_window_frames: int,
) -> dict:
    """Analyze and optionally persist one model variant's threshold."""
    config = json.loads(artifacts.config_path.read_text(encoding="utf-8"))
    target_col = str(config.get("target_col", "target"))
    holdoff_frames = int(config.get("holdoff_frames", config.get("min_distance", 20)))
    error_association_frames = int(config.get("error_association_frames", 1))
    previous_threshold = float(config["threshold"])

    validation_df = read_validation_analysis_rows(artifacts.validation_path, target_col)
    probabilities, targets = pack_validation_sequences(
        validation_df,
        target_col=target_col,
        holdoff_frames=holdoff_frames,
        error_association_frames=error_association_frames,
    )
    target_peaks = np.flatnonzero(targets == 1)
    previous_metrics = metrics_at_threshold(
        probabilities,
        targets,
        threshold=previous_threshold,
        holdoff_frames=holdoff_frames,
        error_association_frames=error_association_frames,
    )
    if apply_threshold is not None:
        apply_threshold = float(apply_threshold)
        if not 0.0 < apply_threshold <= 1.0:
            raise ValueError(
                "A manually applied threshold must be greater than 0 and at most 1."
            )
        thresholds = np.unique(
            np.append(np.asarray(thresholds, dtype=np.float32), apply_threshold)
        )
    selection = select_threshold(
        probabilities,
        target_peaks,
        objective=objective,
        sensitivity_efficiency=sensitivity_efficiency,
        holdoff_frames=holdoff_frames,
        error_association_frames=error_association_frames,
        thresholds=thresholds,
    )

    selection_mode = None
    applied_threshold = None
    applied_metrics = None
    if apply_auto:
        selection_mode = "automatic"
        applied_threshold = selection.threshold
        applied_metrics = selection.metrics
    elif apply_threshold is not None:
        selection_mode = "manual"
        applied_threshold = apply_threshold
        applied_metrics = metrics_at_threshold(
            probabilities,
            targets,
            threshold=apply_threshold,
            holdoff_frames=holdoff_frames,
            error_association_frames=error_association_frames,
        )

    report_threshold = (
        selection.threshold if applied_threshold is None else applied_threshold
    )
    report_metrics = selection.metrics if applied_metrics is None else applied_metrics
    artifacts.metrics_dir.mkdir(parents=True, exist_ok=True)

    search = selection_rows(selection)
    search["applied"] = (
        False
        if applied_threshold is None
        else np.isclose(search["threshold"], applied_threshold)
    )
    search.to_csv(artifacts.search_path, index=False)
    regions = operating_regions(selection)
    regions["applied"] = False
    if applied_threshold is not None:
        applied_rows = search[search["applied"]]
        if not applied_rows.empty:
            applied_signature = applied_rows.iloc[0]["predicted_event_signature"]
            regions["applied"] = (
                regions["predicted_event_signature"] == applied_signature
            ) & regions["selectable"]
    regions.to_csv(artifacts.analysis_path, index=False)

    event_df = classify_validation_events(
        validation_df,
        threshold=report_threshold,
        target_col=target_col,
        holdoff_frames=holdoff_frames,
        error_association_frames=error_association_frames,
    )
    event_df.to_csv(artifacts.events_path, index=False)
    write_threshold_dashboard(
        artifacts.dashboard_path,
        selection=selection,
        presentation="comparison",
        previous_threshold=previous_threshold,
        previous_metrics=previous_metrics,
        applied_threshold=applied_threshold,
    )
    error_plots_written = write_error_examples(
        artifacts.errors_path,
        validation_df=validation_df,
        event_df=event_df,
        threshold=report_threshold,
        target_col=target_col,
        max_plots=max_error_plots,
        window_frames=error_window_frames,
    )

    category_counts = {
        category: int((event_df["category"] == category).sum())
        for category in ("TP", "PE", "oPE", "FP", "FN")
    }
    emission_counts = (
        event_df.loc[
            event_df["emission_offset_frames"].notna(), "emission_offset_frames"
        ]
        .astype(int)
        .value_counts()
        .sort_index()
    )
    analysis_document = {
        "metrics_schema_version": METRICS_SCHEMA_VERSION,
        "modelname": artifacts.modelname,
        "model_variant": artifacts.variant,
        "source_validation_file": repo_relative_path(artifacts.validation_path),
        "threshold_objective": selection.objective,
        "threshold_sensitivity_efficiency": selection.sensitivity_efficiency,
        "threshold_best_attainable_exact_sensitivity": (
            selection.best_attainable_sensitivity
        ),
        "threshold_effective_exact_sensitivity_floor": (
            selection.effective_sensitivity_floor
        ),
        "previous_threshold": previous_threshold,
        "previous_metrics": previous_metrics,
        "recommended_threshold": selection.threshold,
        "recommended_metrics": selection.metrics,
        "report_threshold": report_threshold,
        "report_metrics": report_metrics,
        "apply_mode": selection_mode,
        "applied_threshold": applied_threshold,
        "manual_threshold_meets_effective_floor": (
            None
            if apply_threshold is None
            else bool(
                float(report_metrics["exact_sensitivity"])
                >= selection.effective_sensitivity_floor
            )
        ),
        "event_counts": category_counts,
        "emission_offset_counts": {
            str(int(offset)): int(count) for offset, count in emission_counts.items()
        },
        "num_threshold_candidates": int(len(search)),
        "num_operating_regions": int(len(regions)),
        "num_error_plots": int(error_plots_written),
        "threshold_search_file": repo_relative_path(artifacts.search_path),
        "operating_regions_file": repo_relative_path(artifacts.analysis_path),
        "event_classifications_file": repo_relative_path(artifacts.events_path),
        "dashboard_file": repo_relative_path(artifacts.dashboard_path),
        "error_examples_file": repo_relative_path(artifacts.errors_path),
    }
    write_json_atomic(artifacts.analysis_json_path, analysis_document)

    if selection_mode is not None:
        assert applied_threshold is not None and applied_metrics is not None
        apply_selection(
            artifacts,
            config,
            selection,
            applied_threshold,
            applied_metrics,
            selection_mode,
            previous_threshold,
            previous_metrics,
            holdoff_frames,
            error_association_frames,
        )

    return {
        "model": f"{artifacts.modelname}:{artifacts.variant}",
        "old_threshold": previous_threshold,
        "new_threshold": report_threshold,
        "recommended_threshold": selection.threshold,
        "old_position_error": int(previous_metrics["position_error"]),
        "new_position_error": int(report_metrics["position_error"]),
        "old_false_positive": int(previous_metrics["unmatched_false_positive"]),
        "new_false_positive": int(report_metrics["unmatched_false_positive"]),
        "old_false_negative": int(previous_metrics["unmatched_false_negative"]),
        "new_false_negative": int(report_metrics["unmatched_false_negative"]),
        "old_sensitivity": float(previous_metrics["exact_sensitivity"]),
        "new_sensitivity": float(report_metrics["exact_sensitivity"]),
        "old_precision": float(previous_metrics["exact_positive_predictive_value"]),
        "new_precision": float(report_metrics["exact_positive_predictive_value"]),
        "old_detection_error": float(previous_metrics["unique_detection_error"]),
        "new_detection_error": float(report_metrics["unique_detection_error"]),
        "best_attainable_sensitivity": selection.best_attainable_sensitivity,
        "effective_sensitivity_floor": selection.effective_sensitivity_floor,
        "operating_regions": int(len(regions)),
        "dashboard_path": repo_relative_path(artifacts.dashboard_path),
        "errors_path": repo_relative_path(artifacts.errors_path),
        "analysis_path": repo_relative_path(artifacts.analysis_path),
        "applied": selection_mode is not None,
        "selection_mode": selection_mode,
    }


def print_result(result: dict) -> None:
    action = "APPLIED" if result["applied"] else "ANALYSIS"
    print(
        f"{action:7} {result['model']:<48} "
        f"threshold {result['old_threshold']:.3f} -> {result['new_threshold']:.3f}  "
        f"PE {result['old_position_error']} -> {result['new_position_error']}  "
        f"FP {result['old_false_positive']} -> {result['new_false_positive']}  "
        f"FN {result['old_false_negative']} -> {result['new_false_negative']}"
    )
    print(
        f"{'':56}"
        f"Se {100.0 * result['old_sensitivity']:.2f}% -> "
        f"{100.0 * result['new_sensitivity']:.2f}%  "
        f"+P {100.0 * result['old_precision']:.2f}% -> "
        f"{100.0 * result['new_precision']:.2f}%  "
        f"DER {100.0 * result['old_detection_error']:.2f}% -> "
        f"{100.0 * result['new_detection_error']:.2f}%"
    )
    print(
        f"{'':56}"
        f"attainable Se {100.0 * result['best_attainable_sensitivity']:.2f}%  "
        f"effective floor {100.0 * result['effective_sensitivity_floor']:.2f}%"
    )
    if not np.isclose(result["new_threshold"], result["recommended_threshold"]):
        print(
            f"{'':56}automatic recommendation "
            f"{result['recommended_threshold']:.6f}; manual threshold reported/applied"
        )
    print(
        f"{'':56}{result['operating_regions']} operating regions | "
        f"dashboard {result['dashboard_path']}"
    )
    print(f"{'':56}error examples {result['errors_path']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=ROOT / "models",
        help="Model directory to scan. Defaults to the repository models directory.",
    )
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        help="Restrict processing to this model name. May be supplied repeatedly.",
    )
    parser.add_argument(
        "--variant",
        choices=("best", "tuned", "all"),
        default="all",
        help="Model variant to process. Defaults to all complete variants.",
    )
    parser.add_argument(
        "--objective",
        choices=("min_error", "min_fp"),
        default="min_error",
        help="Threshold policy. Defaults to min_error.",
    )
    parser.add_argument(
        "--sensitivity-efficiency",
        type=float,
        default=0.95,
        help=(
            "Fraction of best attainable exact sensitivity retained as the "
            "selection floor. Defaults to 0.95."
        ),
    )
    parser.add_argument(
        "--threshold-step",
        type=float,
        default=0.001,
        help="Threshold grid step. Defaults to 0.001 (1,001 candidates).",
    )
    apply_group = parser.add_mutually_exclusive_group()
    apply_group.add_argument(
        "--apply-auto",
        action="store_true",
        help="Apply the automatic recommendation after writing analysis reports.",
    )
    apply_group.add_argument(
        "--apply-threshold",
        type=float,
        help="Apply this explicit threshold after writing analysis reports.",
    )
    parser.add_argument(
        "--max-error-plots",
        type=int,
        default=12,
        help=(
            "Maximum representative FP/PE/FN examples written to PDF. "
            "Defaults to 12."
        ),
    )
    parser.add_argument(
        "--error-window-frames",
        type=int,
        default=25,
        help="Frames shown on each side of an error example. Defaults to 25.",
    )
    args = parser.parse_args()

    if not 0.0 < args.sensitivity_efficiency <= 1.0:
        parser.error("--sensitivity-efficiency must be greater than 0 and at most 1")
    if args.apply_threshold is not None and not 0.0 < args.apply_threshold <= 1.0:
        parser.error("--apply-threshold must be greater than 0 and at most 1")
    if args.max_error_plots < 0:
        parser.error("--max-error-plots must be zero or positive")
    if args.error_window_frames < 1:
        parser.error("--error-window-frames must be at least 1")
    if not args.models_dir.is_dir():
        parser.error(f"models directory not found: {args.models_dir}")

    variants = {"best", "tuned"} if args.variant == "all" else {args.variant}
    artifacts, skipped = discover_variants(
        args.models_dir,
        modelnames=None if args.models is None else set(args.models),
        variants=variants,
    )
    for message in skipped:
        print(f"SKIP    {message}")
    if not artifacts:
        print("No complete model variants found.", file=sys.stderr)
        return 1

    thresholds = threshold_grid(args.threshold_step)
    failures = 0
    for artifact in artifacts:
        try:
            result = reselect_variant(
                artifact,
                sensitivity_efficiency=float(args.sensitivity_efficiency),
                objective=str(args.objective),
                thresholds=thresholds,
                apply_auto=bool(args.apply_auto),
                apply_threshold=args.apply_threshold,
                max_error_plots=int(args.max_error_plots),
                error_window_frames=int(args.error_window_frames),
            )
        except Exception as exc:
            failures += 1
            print(
                f"ERROR   {artifact.modelname}:{artifact.variant}: {exc}",
                file=sys.stderr,
            )
            continue
        print_result(result)

    if not args.apply_auto and args.apply_threshold is None:
        print(
            "Analysis reports written; model configs are unchanged. Use "
            "--apply-auto or --apply-threshold to apply a selection."
        )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
