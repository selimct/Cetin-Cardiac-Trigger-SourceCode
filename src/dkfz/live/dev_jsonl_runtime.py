# Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Developed at the German Cancer Research Center (DKFZ).

"""Development JSONL adapter for live predictor smoke tests and demos.

This module is intentionally a harness around the real live predictor boundary:
`predictor.process_sample(...)`. It is useful on a development machine because
stdin/stdout JSON lines are easy to generate, inspect, and test without GPIO
hardware. The production pin-29 acquisition path is intentionally separate
from this model-specific sample harness.
"""

from __future__ import annotations

from dataclasses import asdict
import json
import sys
from typing import TextIO


def require_numeric(sample: dict, key: str, line_number: int) -> float:
    """Extract one numeric field with a clear error message."""
    if key not in sample:
        raise KeyError(f"Missing required field {key!r} on input line {line_number}.")
    return float(sample[key])


def sample_time_ms(
    sample: dict,
    *,
    time_col: str | None,
    sample_idx: int,
    raw_hz: float,
) -> float:
    """Return sample time in milliseconds, deriving it when absent."""
    if time_col is not None and time_col in sample:
        return float(sample[time_col])
    return 1000.0 * float(sample_idx) / float(raw_hz)


def run_jsonl_predictor(
    *,
    predictor,
    raw_hz: float,
    include_puls: bool,
    ecg1_col: str,
    ecg2_col: str,
    puls_col: str,
    optional_puls: bool = False,
    time_col: str | None,
    sid_col: str,
    raw_idx_col: str,
    finished_label: str,
    input_stream: TextIO | None = None,
    ecg3_col: str | None = None,
    emit_probabilities: bool = False,
) -> int:
    """Feed JSONL raw samples into a live predictor and emit JSON events."""
    input_stream = input_stream or sys.stdin
    current_sid: int | str = 0
    processed_samples = 0
    sid_sample_idx = 0

    for line_number, raw_line in enumerate(input_stream, 1):
        line = raw_line.strip()
        if not line:
            continue

        try:
            sample = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON on input line {line_number}: {exc}"
            ) from exc

        if not isinstance(sample, dict):
            raise TypeError(
                f"Each input line must be a JSON object, got {type(sample).__name__} "
                f"on line {line_number}."
            )

        sid = sample.get(sid_col, current_sid)
        if processed_samples > 0 and sid != current_sid:
            final_event = (
                None
                if getattr(predictor, "prediction_delay_frames", 0)
                else predictor.flush(sid=current_sid)
            )
            if final_event is not None:
                print(json.dumps(asdict(final_event)), flush=True)
            predictor.reset()
            sid_sample_idx = 0
        current_sid = sid

        if raw_idx_col in sample:
            sample_idx = int(sample[raw_idx_col])
        elif "idx" in sample:
            sample_idx = int(sample["idx"])
        else:
            sample_idx = sid_sample_idx

        time_ms = sample_time_ms(
            sample,
            time_col=time_col,
            sample_idx=sample_idx,
            raw_hz=raw_hz,
        )
        puls = None
        if include_puls:
            puls = require_numeric(sample, puls_col, line_number)
        elif optional_puls and puls_col in sample:
            puls = require_numeric(sample, puls_col, line_number)

        event = predictor.process_sample(
            ecg1=require_numeric(sample, ecg1_col, line_number),
            ecg2=require_numeric(sample, ecg2_col, line_number),
            ecg3=(
                None
                if ecg3_col is None
                else require_numeric(sample, ecg3_col, line_number)
            ),
            puls=puls,
            sample_idx_400hz=sample_idx,
            time_ms=time_ms,
            sid=sid,
        )
        processed_samples += 1
        sid_sample_idx += 1

        if (
            emit_probabilities
            and predictor.last_probability_updated
            and predictor.last_probability is not None
            and predictor.last_frame is not None
        ):
            print(
                json.dumps(
                    {
                        "schema": "dkfz-live-probability-v1",
                        "sid": sid,
                        "frame_idx_50hz": predictor.last_frame.frame_idx_50hz,
                        "start_sample_idx_400hz": (
                            predictor.last_frame.start_sample_idx_400hz
                        ),
                        "end_sample_idx_400hz": (
                            predictor.last_frame.end_sample_idx_400hz
                        ),
                        "time_ms": predictor.last_frame.end_time_ms,
                        "probability": predictor.last_probability,
                        "trigger_emitted_now": event is not None,
                        **(
                            dict(
                                prediction_delay_frames=predictor.prediction_delay_frames,
                                observed_frame_idx=predictor.last_input_frame.frame_idx_50hz,
                                observed_raw_idx=predictor.last_input_frame.start_sample_idx_400hz,
                                observed_raw_frame_end_idx=predictor.last_input_frame.end_sample_idx_400hz,
                                observed_time_ms=predictor.last_input_frame.end_time_ms,
                            )
                            if getattr(predictor, "prediction_delay_frames", 0)
                            else {}
                        ),
                        **(
                            {
                                "recovery_prob": predictor.last_recovery_probability,
                                "recovery_emitted_now": predictor.trigger_logic.last_event_was_recovery,
                            }
                            if predictor.recovery_head
                            else {}
                        ),
                    }
                ),
                flush=True,
            )

        if event is not None:
            print(json.dumps(asdict(event)), flush=True)

    final_event = predictor.flush(sid=current_sid)
    if final_event is not None:
        print(json.dumps(asdict(final_event)), flush=True)

    print(f"{finished_label} after {processed_samples:_} samples.", file=sys.stderr)
    return processed_samples
