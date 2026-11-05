#!/usr/bin/env python3
# Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Developed at the German Cancer Research Center (DKFZ).

"""Create per-model and pooled prediction-metrics overviews."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dkfz.live.metrics_overview import duration_minutes, format_totals  # noqa: E402
from dkfz.core.gradient_noise import gradient_config_id  # noqa: E402

COUNT_FIELDS = (
    "timepoints",
    "num_target_events",
    "exact_true_positive",
    "position_error",
    "outer_position_error",
    "unmatched_false_positive",
    "unmatched_false_negative",
)
CSV_FIELDS = (
    "scope",
    "input",
    "main_frequency_hz",
    "system",
    "system_type",
    "evaluated_input_files",
    "requested_input_files",
    "timepoints",
    "num_target_events",
    "num_predicted_events",
    "duration_minutes",
    "num_associated_events",
    "exact_true_positive",
    "position_error",
    "outer_position_error",
    "unmatched_false_positive",
    "unmatched_false_negative",
    "unique_error_count",
    "exact_sensitivity",
    "exact_positive_predictive_value",
    "unique_detection_error",
    "beat_sensitivity",
    "beat_positive_predictive_value",
    "beat_detection_error",
    "position_error_per_target",
    "outer_position_error_per_target",
    "false_positive_per_prediction",
    "false_negative_per_target",
)


def _rate(numerator: int, denominator: int) -> float:
    return 0.0 if denominator <= 0 else float(numerator / denominator)


def normalize_record(value: dict[str, Any]) -> dict[str, Any]:
    """Validate one input/system result and normalize its count fields."""
    input_path = str(value.get("input", "")).strip()
    system = str(value.get("system", "")).strip()
    system_type = str(value.get("system_type", "")).strip()
    if not input_path:
        raise ValueError("Prediction metrics record is missing 'input'.")
    if not system:
        raise ValueError("Prediction metrics record is missing 'system'.")
    if system_type not in {"model", "baseline"}:
        raise ValueError(
            f"Prediction metrics record for {system!r} has invalid "
            f"system_type={system_type!r}."
        )

    record: dict[str, Any] = {
        "input": input_path,
        "system": system,
        "system_type": system_type,
        "main_frequency_hz": None,
    }
    frequency = value.get("main_frequency_hz")
    if frequency is not None:
        gradient_config_id(frequency)
        record["main_frequency_hz"] = float(frequency)
    for field in COUNT_FIELDS:
        # Reports from before the outer association tier have no oPE count.
        raw = (
            value.get(field, 0) if field == "outer_position_error" else value.get(field)
        )
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError(
                f"Prediction metrics record for {system!r} is missing numeric "
                f"field {field!r}."
            )
        number = int(raw)
        if number < 0 or number != raw:
            raise ValueError(
                f"Prediction metrics record for {system!r} has invalid "
                f"{field}={raw!r}."
            )
        record[field] = number

    accounted_targets = (
        record["exact_true_positive"]
        + record["position_error"]
        + record["outer_position_error"]
        + record["unmatched_false_negative"]
    )
    if accounted_targets != record["num_target_events"]:
        raise ValueError(
            f"Prediction metrics record for {system!r} accounts for "
            f"{accounted_targets} targets, expected {record['num_target_events']}."
        )
    minutes = value.get("duration_minutes", duration_minutes(record["timepoints"], {}))
    if isinstance(minutes, bool) or not isinstance(minutes, (int, float)):
        raise ValueError(f"Invalid duration_minutes for {system!r}: {minutes!r}.")
    if not math.isfinite(minutes) or minutes < 0:
        raise ValueError(f"Invalid duration_minutes for {system!r}: {minutes!r}.")
    record["duration_minutes"] = float(minutes)
    return record


def load_records(path: Path) -> list[dict[str, Any]]:
    """Load newline-delimited input/system count records."""
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, str, float | None]] = set()
    system_types: dict[str, str] = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(raw, dict):
                raise ValueError(f"Expected a JSON object on {path}:{line_number}.")
            record = normalize_record(raw)
            key = (record["input"], record["system"], record["main_frequency_hz"])
            if key in seen:
                raise ValueError(
                    f"Duplicate prediction metrics for input={key[0]!r}, "
                    f"system={key[1]!r}, main_frequency_hz={key[2]!r}."
                )
            seen.add(key)
            previous_type = system_types.setdefault(
                record["system"], record["system_type"]
            )
            if previous_type != record["system_type"]:
                raise ValueError(
                    f"System {record['system']!r} is both {previous_type!r} and "
                    f"{record['system_type']!r}."
                )
            records.append(record)
    if not records:
        raise ValueError(f"No prediction metrics records found in {path}.")
    if len({record["main_frequency_hz"] is None for record in records}) > 1:
        raise ValueError("Cannot mix clean and gradient-configuration records")
    return records


def _metrics_row(
    counts: dict[str, Any],
    *,
    scope: str,
    input_label: str,
    evaluated_input_files: int,
    requested_input_files: int,
    include_outer_position_errors: bool = False,
) -> dict[str, Any]:
    true_positive = int(counts["exact_true_positive"])
    position_error = int(counts["position_error"])
    outer_position_error = int(counts["outer_position_error"])
    false_positive = int(counts["unmatched_false_positive"])
    false_negative = int(counts["unmatched_false_negative"])
    if not include_outer_position_errors:
        # Without the outer association tier, each pair is an unmatched FP/FN.
        false_positive += outer_position_error
        false_negative += outer_position_error
        outer_position_error = 0
    targets = int(counts["num_target_events"])
    associated = true_positive + position_error + outer_position_error
    predicted = associated + false_positive
    unique_errors = (
        position_error + outer_position_error + false_positive + false_negative
    )
    return {
        "scope": scope,
        "input": input_label,
        "main_frequency_hz": counts.get("main_frequency_hz"),
        "system": counts["system"],
        "system_type": counts["system_type"],
        "evaluated_input_files": evaluated_input_files,
        "requested_input_files": requested_input_files,
        "timepoints": int(counts["timepoints"]),
        "duration_minutes": counts["duration_minutes"],
        "num_target_events": targets,
        "num_predicted_events": predicted,
        "num_associated_events": associated,
        "exact_true_positive": true_positive,
        "position_error": position_error,
        "outer_position_error": outer_position_error,
        "unmatched_false_positive": false_positive,
        "unmatched_false_negative": false_negative,
        "unique_error_count": unique_errors,
        "exact_sensitivity": _rate(true_positive, targets),
        "exact_positive_predictive_value": _rate(true_positive, predicted),
        "unique_detection_error": _rate(unique_errors, predicted),
        "beat_sensitivity": _rate(associated, targets),
        "beat_positive_predictive_value": _rate(associated, predicted),
        "beat_detection_error": _rate(false_positive + false_negative, predicted),
        "position_error_per_target": _rate(position_error, targets),
        "outer_position_error_per_target": _rate(outer_position_error, targets),
        "false_positive_per_prediction": _rate(false_positive, predicted),
        "false_negative_per_target": _rate(false_negative, targets),
    }


def input_rows(
    records: Iterable[dict[str, Any]],
    *,
    requested_input_files: int,
    include_outer_position_errors: bool = False,
) -> list[dict[str, Any]]:
    """Convert normalized input records to derived metric rows."""
    return [
        _metrics_row(
            record,
            scope="input",
            input_label=str(record["input"]),
            evaluated_input_files=1,
            requested_input_files=requested_input_files,
            include_outer_position_errors=include_outer_position_errors,
        )
        for record in records
    ]


def pooled_rows(
    records: Iterable[dict[str, Any]],
    *,
    requested_input_files: int,
    include_outer_position_errors: bool = False,
) -> list[dict[str, Any]]:
    """Pool event counts by gradient configuration and system, then derive rates."""
    grouped: dict[tuple[float | None, str], list[dict[str, Any]]] = {}
    for record in records:
        key = (record.get("main_frequency_hz"), str(record["system"]))
        grouped.setdefault(key, []).append(record)

    rows: list[dict[str, Any]] = []
    for (frequency, system), system_records in grouped.items():
        pooled: dict[str, Any] = {
            "system": system,
            "system_type": system_records[0]["system_type"],
            "main_frequency_hz": frequency,
        }
        for field in COUNT_FIELDS:
            pooled[field] = sum(int(record[field]) for record in system_records)
        pooled["duration_minutes"] = sum(
            record["duration_minutes"] for record in system_records
        )
        rows.append(
            _metrics_row(
                pooled,
                scope="gradient_config" if frequency is not None else "all_inputs",
                input_label="__all_inputs__",
                evaluated_input_files=len(system_records),
                requested_input_files=requested_input_files,
                include_outer_position_errors=include_outer_position_errors,
            )
        )
    return sorted(
        rows,
        key=lambda row: (
            -(row["main_frequency_hz"] or 0),
            row["system_type"] != "baseline",
            row["system"],
        ),
    )


def _percentage(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def _format_table(
    rows: Iterable[dict[str, Any]], *, include_outer_position_errors: bool = False
) -> str:
    row_list = list(rows)
    header = (
        f"{'system':<40} {'type':<9} {'inputs':>7} {'Exact Se':>10} {'Exact +P':>10} "
        f"{'Exact DER':>10} {'Beat Se':>10} {'Beat +P':>10} {'Beat DER':>10} "
        f"{'TP':>8} {'PE':>8} "
        + (f"{'oPE':>8} " if include_outer_position_errors else "")
        + f"{'FP':>8} {'FN':>8}"
    )
    lines = [header, "-" * len(header)]
    for row in row_list:
        coverage = f"{row['evaluated_input_files']}/{row['requested_input_files']}"
        lines.append(
            f"{row['system']:<40} {row['system_type']:<9} {coverage:>7} "
            f"{_percentage(row['exact_sensitivity']):>10} "
            f"{_percentage(row['exact_positive_predictive_value']):>10} "
            f"{_percentage(row['unique_detection_error']):>10} "
            f"{_percentage(row['beat_sensitivity']):>10} "
            f"{_percentage(row['beat_positive_predictive_value']):>10} "
            f"{_percentage(row['beat_detection_error']):>10} "
            f"{row['exact_true_positive']:>8} {row['position_error']:>8} "
            + (
                f"{row['outer_position_error']:>8} "
                if include_outer_position_errors
                else ""
            )
            + f"{row['unmatched_false_positive']:>8} "
            f"{row['unmatched_false_negative']:>8}"
        )
    return "\n".join(lines)


def format_root_overview(
    records: list[dict[str, Any]],
    *,
    requested_input_files: int,
    include_outer_position_errors: bool = False,
) -> str:
    """Format per-input sections followed by one all-system pooled section."""
    if any(record.get("main_frequency_hz") is not None for record in records):
        return format_gradient_overview(
            records,
            requested_input_files=requested_input_files,
            include_outer_position_errors=include_outer_position_errors,
        )
    lines = ["Prediction metrics by input"]
    inputs = list(dict.fromkeys(str(record["input"]) for record in records))
    for input_path in inputs:
        selected = [record for record in records if record["input"] == input_path]
        rows = input_rows(
            selected,
            requested_input_files=requested_input_files,
            include_outer_position_errors=include_outer_position_errors,
        )
        rows.sort(key=lambda row: (row["system_type"] != "baseline", row["system"]))
        lines.extend(
            (
                "",
                input_path,
                "",
                _format_table(
                    rows, include_outer_position_errors=include_outer_position_errors
                ),
            )
        )
    pooled = pooled_rows(
        records,
        requested_input_files=requested_input_files,
        include_outer_position_errors=include_outer_position_errors,
    )
    lines.extend(
        (
            "",
            "Pooled over all input files",
            "",
            _format_table(
                pooled, include_outer_position_errors=include_outer_position_errors
            ),
            "",
            format_totals(pooled, scope="all input files"),
        )
    )
    return "\n".join(lines) + "\n"


def format_model_overview(
    system: str,
    records: list[dict[str, Any]],
    *,
    requested_input_files: int,
    include_outer_position_errors: bool = False,
) -> str:
    """Format one model's input rows and event-weighted pooled result."""
    if any(record.get("main_frequency_hz") is not None for record in records):
        return format_gradient_overview(
            records,
            requested_input_files=requested_input_files,
            system=system,
            include_outer_position_errors=include_outer_position_errors,
        )
    rows = input_rows(
        records,
        requested_input_files=requested_input_files,
        include_outer_position_errors=include_outer_position_errors,
    )
    pooled = pooled_rows(
        records,
        requested_input_files=requested_input_files,
        include_outer_position_errors=include_outer_position_errors,
    )
    lines = [f"Prediction metrics across inputs: {system}"]
    for row in rows:
        lines.extend(
            (
                "",
                str(row["input"]),
                "",
                _format_table(
                    [row], include_outer_position_errors=include_outer_position_errors
                ),
            )
        )
    lines.extend(
        (
            "",
            "Pooled over all input files",
            "",
            _format_table(
                pooled, include_outer_position_errors=include_outer_position_errors
            ),
        )
    )
    lines.extend(("", format_totals(pooled, scope="all input files")))
    return "\n".join(lines) + "\n"


def format_gradient_overview(
    records: list[dict[str, Any]],
    *,
    requested_input_files: int,
    system: str | None = None,
    include_outer_position_errors: bool = False,
) -> str:
    """One section per frequency, pooling recordings without pooling configurations."""
    title = "Prediction metrics by gradient configuration"
    if system is not None:
        title += f": {system}"
    lines = [title]
    rows = pooled_rows(
        records,
        requested_input_files=requested_input_files,
        include_outer_position_errors=include_outer_position_errors,
    )
    for frequency in dict.fromkeys(row["main_frequency_hz"] for row in rows):
        selected = [row for row in rows if row["main_frequency_hz"] == frequency]
        label = f"Main frequency {frequency:g} Hz"
        lines.extend(
            (
                "",
                label,
                "Pooled over all input files",
                "",
                _format_table(
                    selected,
                    include_outer_position_errors=include_outer_position_errors,
                ),
                "",
                format_totals(selected, scope=label),
            )
        )
    return "\n".join(lines) + "\n"


def _write_text_atomic(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(contents, encoding="utf-8")
    temporary.replace(path)


def _write_csv_atomic(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    include_outer_position_errors: bool = False,
) -> None:
    buffer = io.StringIO()
    fields = [
        field
        for field in CSV_FIELDS
        if include_outer_position_errors or not field.startswith("outer_position_error")
    ]
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows({field: row[field] for field in fields} for row in rows)
    _write_text_atomic(path, buffer.getvalue())


def write_prediction_overviews(
    records: list[dict[str, Any]],
    *,
    metrics_root: Path,
    overview_text_path: Path,
    requested_input_files: int,
    include_outer_position_errors: bool = False,
) -> tuple[Path, Path, list[Path]]:
    """Write root pooled comparison and per-model all-input overviews."""
    if requested_input_files <= 0:
        raise ValueError("requested_input_files must be positive.")
    metrics_root.mkdir(parents=True, exist_ok=True)
    overview_csv_path = overview_text_path.with_suffix(".csv")
    pooled = pooled_rows(
        records,
        requested_input_files=requested_input_files,
        include_outer_position_errors=include_outer_position_errors,
    )
    _write_text_atomic(
        overview_text_path,
        format_root_overview(
            records,
            requested_input_files=requested_input_files,
            include_outer_position_errors=include_outer_position_errors,
        ),
    )
    _write_csv_atomic(
        overview_csv_path,
        pooled,
        include_outer_position_errors=include_outer_position_errors,
    )

    model_paths: list[Path] = []
    model_names = sorted(
        {
            str(record["system"])
            for record in records
            if record["system_type"] == "model"
        }
    )
    for model_name in model_names:
        model_records = [record for record in records if record["system"] == model_name]
        synthetic = any(
            record.get("main_frequency_hz") is not None for record in model_records
        )
        model_rows = (
            []
            if synthetic
            else input_rows(
                model_records,
                requested_input_files=requested_input_files,
                include_outer_position_errors=include_outer_position_errors,
            )
        )
        model_rows.extend(
            pooled_rows(
                model_records,
                requested_input_files=requested_input_files,
                include_outer_position_errors=include_outer_position_errors,
            )
        )
        model_root = metrics_root / model_name
        model_csv_path = model_root / "prediction_metrics_all_inputs.csv"
        model_text_path = model_root / "prediction_metrics_all_inputs.txt"
        _write_csv_atomic(
            model_csv_path,
            model_rows,
            include_outer_position_errors=include_outer_position_errors,
        )
        _write_text_atomic(
            model_text_path,
            format_model_overview(
                model_name,
                model_records,
                requested_input_files=requested_input_files,
                include_outer_position_errors=include_outer_position_errors,
            ),
        )
        model_paths.extend((model_csv_path, model_text_path))
    return overview_text_path, overview_csv_path, model_paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True, type=Path)
    parser.add_argument("--metrics-root", required=True, type=Path)
    parser.add_argument("--overview-text", required=True, type=Path)
    parser.add_argument("--requested-input-files", required=True, type=int)
    parser.add_argument(
        "--include-outer-position-errors",
        action="store_true",
        help="Include outer position errors in counts and columns. By default, "
        "each outer match counts as one FP and one FN, and oPE columns are omitted.",
    )
    args = parser.parse_args()

    try:
        records = load_records(args.records)
        text_path, csv_path, model_paths = write_prediction_overviews(
            records,
            metrics_root=args.metrics_root,
            overview_text_path=args.overview_text,
            requested_input_files=args.requested_input_files,
            include_outer_position_errors=args.include_outer_position_errors,
        )
    except (OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))

    print(f"Wrote {text_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {len(model_paths) // 2} per-model all-input overview(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
