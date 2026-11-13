# Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Developed at the German Cancer Research Center (DKFZ).

"""Compare probability and trigger JSONL from Python and C++ runtimes."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

PROBABILITY_SCHEMA = "dkfz-live-probability-v1"
TRIGGER_SCHEMA = "dkfz-live-trigger-v1"


@dataclass(frozen=True)
class NativeRuntimeParityReport:
    probability_frames: int
    trigger_events: int
    max_probability_abs_diff: float


def _read_records(
    path: str | Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    probabilities = []
    events = []
    with Path(path).open("r", encoding="utf-8") as source:
        for line_number, raw_line in enumerate(source, 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object.")
            schema = record.get("schema")
            if schema == PROBABILITY_SCHEMA:
                probabilities.append(record)
            elif schema == TRIGGER_SCHEMA or "peak_frame_idx_50hz" in record:
                events.append(record)
            else:
                raise ValueError(
                    f"{path}:{line_number}: unsupported runtime output record."
                )
    return probabilities, events


def _identity(record: dict[str, Any], fields: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(str(record.get(field)) for field in fields)


def _require_close_optional(
    python_value: Any,
    native_value: Any,
    *,
    label: str,
    atol: float,
    rtol: float,
) -> None:
    if python_value is None or native_value is None:
        if python_value is not None or native_value is not None:
            raise ValueError(
                f"{label} differs: Python={python_value}, native={native_value}."
            )
        return
    if not math.isclose(
        float(python_value),
        float(native_value),
        abs_tol=float(atol),
        rel_tol=float(rtol),
    ):
        raise ValueError(
            f"{label} differs: Python={python_value}, native={native_value}."
        )


def compare_native_runtime_outputs(
    python_output: str | Path,
    native_output: str | Path,
    *,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> NativeRuntimeParityReport:
    """Require aligned frame probabilities and trigger events in two outputs."""
    python_probs, python_events = _read_records(python_output)
    native_probs, native_events = _read_records(native_output)
    if len(python_probs) != len(native_probs):
        raise ValueError(
            "Probability frame count differs: "
            f"Python={len(python_probs)}, native={len(native_probs)}."
        )

    frame_fields = (
        "sid",
        "frame_idx_50hz",
        "start_sample_idx_400hz",
        "end_sample_idx_400hz",
        "trigger_emitted_now",
    )
    max_abs_diff = 0.0
    for index, (python_record, native_record) in enumerate(
        zip(python_probs, native_probs)
    ):
        if _identity(python_record, frame_fields) != _identity(
            native_record, frame_fields
        ):
            raise ValueError(
                f"Probability frame identity differs at output index {index}: "
                f"Python={_identity(python_record, frame_fields)}, "
                f"native={_identity(native_record, frame_fields)}."
            )
        for field in (
            "prediction_delay_frames",
            "observed_frame_idx",
            "observed_raw_idx",
            "observed_raw_frame_end_idx",
        ):
            if python_record.get(field, 0) != native_record.get(field, 0):
                raise ValueError(
                    f"Observation identity differs at output index {index}: {field}"
                )
        _require_close_optional(
            python_record.get("observed_time_ms"),
            native_record.get("observed_time_ms"),
            label=f"Observation time at output index {index}",
            atol=atol,
            rtol=rtol,
        )
        _require_close_optional(
            python_record.get("time_ms"),
            native_record.get("time_ms"),
            label=f"Probability time at output index {index}",
            atol=atol,
            rtol=rtol,
        )
        _require_close_optional(
            python_record.get("recovery_prob"),
            native_record.get("recovery_prob"),
            label=f"Recovery probability at output index {index}",
            atol=atol,
            rtol=rtol,
        )
        if bool(python_record.get("recovery_emitted_now", False)) != bool(
            native_record.get("recovery_emitted_now", False)
        ):
            raise ValueError(f"Recovery emission differs at output index {index}")
        python_probability = float(python_record["probability"])
        if python_record.get("recovery_prob") is not None:
            max_abs_diff = max(
                max_abs_diff,
                abs(
                    float(python_record["recovery_prob"])
                    - float(native_record["recovery_prob"])
                ),
            )
        native_probability = float(native_record["probability"])
        difference = abs(python_probability - native_probability)
        max_abs_diff = max(max_abs_diff, difference)
        if not math.isclose(
            python_probability,
            native_probability,
            abs_tol=float(atol),
            rel_tol=float(rtol),
        ):
            raise ValueError(
                f"Probability differs at frame {python_record['frame_idx_50hz']}: "
                f"Python={python_probability}, native={native_probability}, "
                f"abs_diff={difference}."
            )

    if len(python_events) != len(native_events):
        raise ValueError(
            "Trigger event count differs: "
            f"Python={len(python_events)}, native={len(native_events)}."
        )
    event_fields = (
        "sid",
        "peak_frame_idx_50hz",
        "peak_sample_idx_400hz",
        "emitted_on_frame_idx_50hz",
        "emitted_on_sample_idx_400hz",
    )
    for index, (python_event, native_event) in enumerate(
        zip(python_events, native_events)
    ):
        if _identity(python_event, event_fields) != _identity(
            native_event, event_fields
        ):
            raise ValueError(
                f"Trigger identity differs at event {index}: "
                f"Python={_identity(python_event, event_fields)}, "
                f"native={_identity(native_event, event_fields)}."
            )
        if not math.isclose(
            float(python_event["probability"]),
            float(native_event["probability"]),
            abs_tol=float(atol),
            rel_tol=float(rtol),
        ):
            raise ValueError(f"Trigger probability differs at event {index}.")
        for time_field in (
            "peak_time_ms",
            "emitted_on_time_ms",
        ):
            _require_close_optional(
                python_event.get(time_field),
                native_event.get(time_field),
                label=f"{time_field} at event {index}",
                atol=atol,
                rtol=rtol,
            )

    return NativeRuntimeParityReport(
        probability_frames=len(python_probs),
        trigger_events=len(python_events),
        max_probability_abs_diff=max_abs_diff,
    )
