#!/usr/bin/env python3
# Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Developed at the German Cancer Research Center (DKFZ).

"""Aggregate nested-LOPO training results without modifying model artifacts."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import io
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dkfz.live.event_matching import METRICS_SCHEMA_VERSION  # noqa: E402
from dkfz.live.io import repo_relative_path  # noqa: E402
from dkfz.live.metrics_overview import duration_minutes, format_totals  # noqa: E402

CV_METHODS = ("nested_lopo", "nested_lopo_hpo", "nested_recovery_lopo")
CONFIGURATION_KEYS = (
    "input_dim",
    "hidden_dim",
    "num_layers",
    "window_size",
    "lr",
    "batch_size",
    "include_puls",
    "ecg3_col",
    "ecg_slope_mode",
    "signal_quality_mode",
    "recovery_head",
    "recovery_max_fp_increase",
)


@dataclass(frozen=True)
class TrainingRun:
    """Paths identifying one completed nested-LOPO result."""

    modelname: str
    cv_method: str
    run_dir: Path

    @property
    def summary_path(self) -> Path:
        return self.run_dir / "summary.json"

    @property
    def config_path(self) -> Path:
        return self.run_dir / "train_config.json"

    @property
    def selection_path(self) -> Path:
        return self.run_dir / "best_selection.json"

    @property
    def outer_results_path(self) -> Path:
        return self.run_dir / "outer_results.csv"


def normalize_modelname(value: str) -> str:
    """Normalize a model or model-family directory name to its base name."""
    name = Path(value).name
    return name[:-3] if name.endswith("_cv") else name


def discover_training_runs(
    models_dir: Path,
    *,
    cv_methods: set[str],
    modelnames: set[str] | None = None,
) -> tuple[list[TrainingRun], list[str]]:
    """Discover completed requested LOPO runs and return skip diagnostics."""
    if not models_dir.is_dir():
        raise FileNotFoundError(f"Training models directory not found: {models_dir}")

    normalized_names = (
        None
        if modelnames is None
        else {normalize_modelname(name) for name in modelnames}
    )
    discovered: list[TrainingRun] = []
    skipped: list[str] = []
    known_names: set[str] = set()

    for family_dir in sorted(path for path in models_dir.iterdir() if path.is_dir()):
        if not family_dir.name.endswith("_cv"):
            continue
        modelname = normalize_modelname(family_dir.name)
        known_names.add(modelname)
        if normalized_names is not None and modelname not in normalized_names:
            continue

        model_runs = 0
        for cv_method in CV_METHODS:
            if cv_method not in cv_methods:
                continue
            run = TrainingRun(
                modelname=modelname,
                cv_method=cv_method,
                run_dir=family_dir / cv_method,
            )
            if run.summary_path.is_file():
                discovered.append(run)
                model_runs += 1
            elif run.run_dir.exists():
                skipped.append(
                    f"{modelname}:{cv_method}: missing "
                    f"{repo_relative_path(run.summary_path)}"
                )
        if normalized_names is not None and model_runs == 0:
            skipped.append(
                f"{modelname}: no complete requested CV result "
                f"({', '.join(sorted(cv_methods))})"
            )

    if normalized_names is not None:
        for modelname in sorted(normalized_names - known_names):
            skipped.append(f"{modelname}: model family directory not found")

    method_order = {method: index for index, method in enumerate(CV_METHODS)}
    discovered.sort(key=lambda run: (run.modelname, method_order[run.cv_method]))
    return discovered, skipped


def _read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _required_object(parent: dict[str, Any], key: str, path: Path) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{path} is missing object {key!r}")
    return value


def _required_number(parent: dict[str, Any], key: str, path: Path) -> float:
    value = parent.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} is missing numeric field {key!r}")
    return float(value)


def _macro_metric(
    macro: dict[str, Any],
    key: str,
    path: Path,
) -> dict[str, float]:
    values = _required_object(macro, key, path)
    return {
        "mean": _required_number(values, "mean", path),
        "std": _required_number(values, "std", path),
    }


def _rate(numerator: int, denominator: int) -> float:
    return 0.0 if denominator <= 0 else float(numerator / denominator)


def _evaluated_timepoints(run: TrainingRun, num_outer_folds: int) -> int | None:
    """Count only the held-out folds represented in the pooled summary."""
    if not run.outer_results_path.is_file():
        return None
    with run.outer_results_path.open(encoding="utf-8", newline="") as stream:
        patients = [row["test_patient_id"] for row in csv.DictReader(stream)]
    if len(patients) != num_outer_folds or len(set(patients)) != num_outer_folds:
        return None
    total = 0
    for patient in patients:
        path = run.run_dir / "outer" / patient / "metrics" / "test_metrics.json"
        if not path.is_file():
            return None
        metrics = _read_json_object(path)
        count = _required_number(metrics, "num_all_event_rows", path)
        if count < 0 or not count.is_integer():
            raise ValueError(f"Invalid num_all_event_rows in {path}: {count}")
        total += int(count)
    return total


def build_training_record(run: TrainingRun) -> dict[str, Any]:
    """Read and normalize the metrics for one nested-LOPO result."""
    summary = _read_json_object(run.summary_path)
    event = _required_object(summary, "pooled_event_position", run.summary_path)
    schema_version = int(
        _required_number(event, "metrics_schema_version", run.summary_path)
    )
    if schema_version not in {2, 3, METRICS_SCHEMA_VERSION}:
        raise ValueError(
            f"{run.summary_path} uses metrics schema {schema_version}; "
            f"expected 2, 3 or {METRICS_SCHEMA_VERSION}"
        )
    macro = _required_object(summary, "macro_patient_metrics", run.summary_path)
    best_model = _required_object(summary, "best_model", run.summary_path)

    integer_event_keys = (
        "num_predicted_events",
        "num_target_events",
        "exact_true_positive",
        "position_error",
        "unmatched_false_positive",
        "unmatched_false_negative",
    )
    event_counts = {
        key: int(_required_number(event, key, run.summary_path))
        for key in integer_event_keys
    }
    outer_errors = int(event.get("outer_position_error", 0))
    pooled_metrics = {
        "exact_sensitivity": _required_number(
            event, "exact_sensitivity", run.summary_path
        ),
        "exact_positive_predictive_value": _required_number(
            event, "exact_positive_predictive_value", run.summary_path
        ),
        "unique_detection_error": _required_number(
            event, "unique_detection_error", run.summary_path
        ),
    }
    macro_metrics = {
        key: _macro_metric(macro, key, run.summary_path)
        for key in (
            "exact_sensitivity",
            "exact_positive_predictive_value",
            "unique_detection_error",
        )
    }

    config = _read_json_object(run.config_path) if run.config_path.is_file() else {}
    selection = (
        _read_json_object(run.selection_path) if run.selection_path.is_file() else {}
    )
    configured_method = summary.get("cv_method")
    if configured_method is not None and configured_method != run.cv_method:
        raise ValueError(
            f"{run.summary_path} declares cv_method={configured_method!r}, "
            f"expected {run.cv_method!r}"
        )

    selected_epoch = int(
        _required_number(best_model, "selected_epoch", run.summary_path)
    )
    threshold = _required_number(best_model, "threshold", run.summary_path)
    targets = event_counts["num_target_events"]
    predictions = event_counts["num_predicted_events"]
    associated = (
        event_counts["exact_true_positive"]
        + event_counts["position_error"]
        + outer_errors
    )
    beat_metrics = {
        "beat_sensitivity": _rate(associated, targets),
        "beat_positive_predictive_value": _rate(associated, predictions),
        "beat_detection_error": _rate(
            event_counts["unmatched_false_positive"]
            + event_counts["unmatched_false_negative"],
            predictions,
        ),
    }
    num_outer_folds = int(
        _required_number(summary, "num_outer_folds", run.summary_path)
    )
    timepoints = _evaluated_timepoints(run, num_outer_folds)
    selection_best_sensitivity = selection.get(
        "best_attainable_sensitivity",
        config.get("threshold_best_attainable_exact_sensitivity"),
    )

    return {
        "modelname": run.modelname,
        "cv_method": run.cv_method,
        "evaluation_scope": summary.get(
            "evaluation_scope", "full_detector_patient_holdout"
        ),
        "base_patient_holdout_verified": summary.get("base_patient_holdout_verified"),
        "metrics_schema_version": schema_version,
        "num_patients": int(
            _required_number(summary, "num_patients", run.summary_path)
        ),
        "num_outer_folds": num_outer_folds,
        "timepoints": timepoints,
        "duration_minutes": (
            None if timepoints is None else duration_minutes(timepoints, config)
        ),
        "deployment_selected_epoch": selected_epoch,
        "deployment_threshold": threshold,
        "fp_threshold": config.get("fp_threshold", selection.get("fp_threshold")),
        "der_threshold": config.get("der_threshold", selection.get("der_threshold")),
        "recovery_head": bool(config.get("recovery_head", False)),
        "recovery_threshold": config.get("recovery_threshold"),
        "recovery_max_fp_increase": config.get("recovery_max_fp_increase"),
        "recovery_validation_metrics": config.get("recovery_validation_metrics"),
        "selection_best_attainable_exact_sensitivity": (
            None
            if selection_best_sensitivity is None
            else float(selection_best_sensitivity)
        ),
        "pooled_event_position": {**event, **beat_metrics},
        "macro_patient_metrics": macro_metrics,
        "error_rates": {
            "position_error_per_target": _rate(event_counts["position_error"], targets),
            "false_positive_per_prediction": _rate(
                event_counts["unmatched_false_positive"], predictions
            ),
            "false_negative_per_target": _rate(
                event_counts["unmatched_false_negative"], targets
            ),
        },
        "configuration": {
            key: config[key] for key in CONFIGURATION_KEYS if key in config
        },
        "source_files": {
            "summary": repo_relative_path(run.summary_path),
            "train_config": (
                repo_relative_path(run.config_path)
                if run.config_path.is_file()
                else None
            ),
            "best_selection": (
                repo_relative_path(run.selection_path)
                if run.selection_path.is_file()
                else None
            ),
            "outer_results": (
                repo_relative_path(run.outer_results_path)
                if run.outer_results_path.is_file()
                else None
            ),
        },
        "_overview": {
            **event_counts,
            **pooled_metrics,
            **beat_metrics,
        },
    }


def _overview_row(record: dict[str, Any]) -> dict[str, Any]:
    event = record["_overview"]
    macro = record["macro_patient_metrics"]
    rates = record["error_rates"]
    return {
        "model": record["modelname"],
        "cv_method": record["cv_method"],
        "num_patients": record["num_patients"],
        "num_outer_folds": record["num_outer_folds"],
        "timepoints": record["timepoints"],
        "duration_minutes": record["duration_minutes"],
        "deployment_selected_epoch": record["deployment_selected_epoch"],
        "deployment_threshold": record["deployment_threshold"],
        "fp_threshold": record.get("fp_threshold"),
        "der_threshold": record.get("der_threshold"),
        "recovery_head": record["recovery_head"],
        "recovery_threshold": record["recovery_threshold"],
        "recovery_max_fp_increase": record["recovery_max_fp_increase"],
        "outer_position_error": (
            event.get("outer_position_error")
            if record["metrics_schema_version"] >= 3
            else None
        ),
        "weighted_detection_error": event.get("weighted_detection_error"),
        "pooled_exact_sensitivity": event["exact_sensitivity"],
        "pooled_exact_positive_predictive_value": event[
            "exact_positive_predictive_value"
        ],
        "pooled_unique_detection_error": event["unique_detection_error"],
        "pooled_beat_sensitivity": event["beat_sensitivity"],
        "pooled_beat_positive_predictive_value": event[
            "beat_positive_predictive_value"
        ],
        "pooled_beat_detection_error": event["beat_detection_error"],
        "macro_exact_sensitivity_mean": macro["exact_sensitivity"]["mean"],
        "macro_exact_sensitivity_std": macro["exact_sensitivity"]["std"],
        "macro_exact_positive_predictive_value_mean": macro[
            "exact_positive_predictive_value"
        ]["mean"],
        "macro_exact_positive_predictive_value_std": macro[
            "exact_positive_predictive_value"
        ]["std"],
        "macro_unique_detection_error_mean": macro["unique_detection_error"]["mean"],
        "macro_unique_detection_error_std": macro["unique_detection_error"]["std"],
        "num_target_events": event["num_target_events"],
        "num_predicted_events": event["num_predicted_events"],
        "exact_true_positive": event["exact_true_positive"],
        "position_error": event["position_error"],
        "unmatched_false_positive": event["unmatched_false_positive"],
        "unmatched_false_negative": event["unmatched_false_negative"],
        "position_error_per_target": rates["position_error_per_target"],
        "false_positive_per_prediction": rates["false_positive_per_prediction"],
        "false_negative_per_target": rates["false_negative_per_target"],
        "selection_best_attainable_exact_sensitivity": record[
            "selection_best_attainable_exact_sensitivity"
        ],
    }


def _percentage(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def _macro_percentage(metric: dict[str, float]) -> str:
    return f"{100.0 * metric['mean']:.2f}±{100.0 * metric['std']:.2f}%"


def format_text_overview(records: list[dict[str, Any]]) -> str:
    """Return a compact human-readable overview of normalized records."""
    header_format = (
        "{:<34} {:<16} {:>4} {:>5} {:>6} {:>9} {:>9} "
        "{:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>15} {:>15} {:>15} "
        "{:>8} {:>8} {:>8} {:>8} {:>10} {:>10} {:>10}"
    )
    lines = [
        header_format.format(
            "model",
            "CV method",
            "pts",
            "folds",
            "dep ep",
            "deploy th",
            "recov th",
            "Exact Se",
            "Exact +P",
            "Exact DER",
            "Beat Se",
            "Beat +P",
            "Beat DER",
            "macro Se±SD",
            "macro +P±SD",
            "macro DER±SD",
            "TP",
            "PE",
            "FP",
            "FN",
            "PE/target",
            "FP/pred",
            "FN/target",
        ),
        header_format.format(
            "-----",
            "---------",
            "---",
            "-----",
            "-----",
            "---------",
            "---------",
            "-------",
            "-------",
            "--------",
            "-------",
            "-------",
            "--------",
            "-----------",
            "-----------",
            "------------",
            "--",
            "--",
            "--",
            "--",
            "---------",
            "-------",
            "---------",
        ),
    ]
    for record in records:
        event = record["_overview"]
        macro = record["macro_patient_metrics"]
        rates = record["error_rates"]
        recovery_threshold = record["recovery_threshold"]
        recovery_label = (
            f"{recovery_threshold:.4f}"
            if recovery_threshold is not None
            else "disabled" if record["recovery_head"] else "off"
        )
        lines.append(
            header_format.format(
                record["modelname"],
                record["cv_method"],
                record["num_patients"],
                record["num_outer_folds"],
                record["deployment_selected_epoch"],
                f"{record['deployment_threshold']:.4f}",
                recovery_label,
                _percentage(event["exact_sensitivity"]),
                _percentage(event["exact_positive_predictive_value"]),
                _percentage(event["unique_detection_error"]),
                _percentage(event["beat_sensitivity"]),
                _percentage(event["beat_positive_predictive_value"]),
                _percentage(event["beat_detection_error"]),
                _macro_percentage(macro["exact_sensitivity"]),
                _macro_percentage(macro["exact_positive_predictive_value"]),
                _macro_percentage(macro["unique_detection_error"]),
                event["exact_true_positive"],
                event["position_error"],
                event["unmatched_false_positive"],
                event["unmatched_false_negative"],
                _percentage(rates["position_error_per_target"]),
                _percentage(rates["false_positive_per_prediction"]),
                _percentage(rates["false_negative_per_target"]),
            )
        )
    totals = [
        {
            "system": f"{record['modelname']} ({record['cv_method']})",
            "num_target_events": record["_overview"]["num_target_events"],
            "num_predicted_events": record["_overview"]["num_predicted_events"],
            "duration_minutes": record["duration_minutes"],
        }
        for record in records
    ]
    lines.extend(("", format_totals(totals, scope="all held-out outer folds")))
    if any(record["cv_method"] == "nested_recovery_lopo" for record in records):
        lines.append(
            "Recovery LOPO holds out patients from recovery-head training/selection; the frozen base may have seen them."
        )
    return "\n".join(lines) + "\n"


def _write_text_atomic(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(contents, encoding="utf-8")
    temporary.replace(path)


def write_training_metrics(
    records: list[dict[str, Any]],
    *,
    output_dir: Path,
) -> tuple[Path, Path, list[Path]]:
    """Write text/CSV overviews and one normalized JSON document per model."""
    output_dir.mkdir(parents=True, exist_ok=True)
    text_path = output_dir / "train_metrics_overview.txt"
    csv_path = output_dir / "train_metrics_overview.csv"
    _write_text_atomic(text_path, format_text_overview(records))

    rows = [_overview_row(record) for record in records]
    csv_buffer = io.StringIO()
    writer = csv.DictWriter(csv_buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    _write_text_atomic(csv_path, csv_buffer.getvalue())

    model_paths: list[Path] = []
    modelnames = sorted({str(record["modelname"]) for record in records})
    for modelname in modelnames:
        model_records = [
            {key: value for key, value in record.items() if key != "_overview"}
            for record in records
            if record["modelname"] == modelname
        ]
        model_path = output_dir / modelname / "training_metrics.json"
        _write_text_atomic(
            model_path,
            json.dumps(
                {
                    "metrics_schema_version": METRICS_SCHEMA_VERSION,
                    "modelname": modelname,
                    "runs": model_records,
                },
                indent=2,
            )
            + "\n",
        )
        model_paths.append(model_path)

    return text_path, csv_path, model_paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "models_dir",
        type=Path,
        help="Root containing <modelname>_cv training-family directories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Artifact root. Defaults to the project's metrics/ directory.",
    )
    parser.add_argument(
        "--cv-method",
        choices=(*CV_METHODS, "all"),
        default="all",
        help="CV result type to aggregate. Defaults to all, including recovery-head LOPO.",
    )
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        help="Restrict processing to this base model name. Repeatable.",
    )
    parser.add_argument(
        "--only-missing",
        action="store_true",
        help=(
            "Reuse existing training results, as this command always does. "
            "Overview tables and normalized JSON are still refreshed to include "
            "the current metrics and totals; no training is run."
        ),
    )
    args = parser.parse_args()

    methods = set(CV_METHODS) if args.cv_method == "all" else {args.cv_method}
    try:
        runs, skipped = discover_training_runs(
            args.models_dir,
            cv_methods=methods,
            modelnames=None if args.models is None else set(args.models),
        )
    except FileNotFoundError as exc:
        parser.error(str(exc))
    for message in skipped:
        print(f"SKIP    {message}", file=sys.stderr)
    if not runs:
        print("No complete requested nested-LOPO results found.", file=sys.stderr)
        return 1

    records: list[dict[str, Any]] = []
    failures = 0
    for run in runs:
        try:
            records.append(build_training_record(run))
        except (json.JSONDecodeError, OSError, TypeError, ValueError) as exc:
            failures += 1
            print(
                f"ERROR   {run.modelname}:{run.cv_method}: {exc}",
                file=sys.stderr,
            )
    if not records:
        return 1

    output_dir = args.output_dir or ROOT / "metrics"
    text_path, csv_path, model_paths = write_training_metrics(
        records,
        output_dir=output_dir,
    )
    print(f"Wrote {repo_relative_path(text_path)}")
    print(f"Wrote {repo_relative_path(csv_path)}")
    print(f"Wrote {len(model_paths)} per-model training metrics file(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
