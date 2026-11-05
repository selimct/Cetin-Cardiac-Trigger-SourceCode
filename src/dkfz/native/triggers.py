# Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Developed at the German Cancer Research Center (DKFZ).

"""Python-compatible adapter for the native streaming trigger detector."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dkfz.native import _live_core


@dataclass(frozen=True)
class SiemensTriggerCleanStats:
    """Counts from normalizing Siemens trigger marker pairs."""

    complete_events: int
    removed_incomplete: int
    changed_samples: int


def clean_siemens_trigger_pairs(
    values,
    *,
    min_pair_gap_samples: int = 1,
    max_pair_gap_samples: int = 50,
) -> tuple[np.ndarray, SiemensTriggerCleanStats]:
    """Normalize complete Siemens trigger-marker pairs and remove strays."""
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError("siemens_tr values must be one-dimensional.")
    if min_pair_gap_samples < 1:
        raise ValueError("min_pair_gap_samples must be at least 1.")
    if max_pair_gap_samples < min_pair_gap_samples:
        raise ValueError("max_pair_gap_samples must be >= min_pair_gap_samples.")

    unexpected = np.setdiff1d(np.unique(arr), np.array([0, 1, 2], dtype=arr.dtype))
    if unexpected.size:
        preview = ", ".join(str(value) for value in unexpected[:10])
        raise ValueError(f"siemens_tr contains unsupported value(s): {preview}")

    out = np.zeros(len(arr), dtype=np.uint8)
    marker_indices = np.flatnonzero(arr > 0)
    complete_events = 0
    removed_incomplete = 0

    index = 0
    while index < len(marker_indices):
        first = int(marker_indices[index])
        if index + 1 < len(marker_indices):
            second = int(marker_indices[index + 1])
            gap = second - first
            if min_pair_gap_samples <= gap <= max_pair_gap_samples:
                out[first] = 1
                out[second] = 2
                complete_events += 1
                index += 2
                continue

        removed_incomplete += 1
        index += 1

    changed_samples = int(np.count_nonzero(arr.astype(np.uint8, copy=False) != out))
    stats = SiemensTriggerCleanStats(
        complete_events=complete_events,
        removed_incomplete=removed_incomplete,
        changed_samples=changed_samples,
    )
    return out, stats


@dataclass(frozen=True)
class TriggerEvent:
    """One predicted trigger event emitted by the live predictor."""

    sid: int | str
    peak_frame_idx_50hz: int
    peak_sample_idx_400hz: int
    peak_time_ms: float | None
    emitted_on_frame_idx_50hz: int
    emitted_on_sample_idx_400hz: int
    emitted_on_time_ms: float | None
    probability: float


class StreamingHoldoffLocalMaxTrigger:
    """Expose the canonical C++ trigger state machine through the Python API."""

    def __init__(self, threshold: float, holdoff_frames: int):
        self.threshold = float(threshold)
        self.holdoff_frames = int(holdoff_frames)
        self._native = _live_core.NativeStreamingTriggerDetector(
            self.threshold,
            self.holdoff_frames,
        )
        self._sid_values: dict[str, int | str] = {}

    @staticmethod
    def _sid_token(sid: int | str) -> str:
        prefix = "i" if isinstance(sid, int) and not isinstance(sid, bool) else "s"
        return f"{prefix}:{sid}"

    def _remember_sid(self, sid: int | str) -> str:
        token = self._sid_token(sid)
        self._sid_values[token] = sid
        return token

    def _convert_event(self, event) -> TriggerEvent | None:
        if event is None:
            return None
        return TriggerEvent(
            sid=self._sid_values.get(event.sid, event.sid),
            peak_frame_idx_50hz=int(event.peak_frame_idx_50hz),
            peak_sample_idx_400hz=int(event.peak_sample_idx_400hz),
            peak_time_ms=event.peak_time_ms,
            emitted_on_frame_idx_50hz=int(event.emitted_on_frame_idx_50hz),
            emitted_on_sample_idx_400hz=int(event.emitted_on_sample_idx_400hz),
            emitted_on_time_ms=event.emitted_on_time_ms,
            probability=float(event.probability),
        )

    def reset(self) -> None:
        self._native.reset()
        self._sid_values.clear()

    def update(
        self,
        probability: float,
        current_frame_idx_50hz: int,
        current_sample_idx_400hz: int,
        current_time_ms: float | None,
        sid: int | str = 0,
    ) -> TriggerEvent | None:
        event = self._native.update(
            probability=float(probability),
            current_frame_idx_50hz=int(current_frame_idx_50hz),
            current_sample_idx_400hz=int(current_sample_idx_400hz),
            current_time_ms=current_time_ms,
            sid=self._remember_sid(sid),
        )
        return self._convert_event(event)

    def flush(
        self,
        current_frame_idx_50hz: int,
        current_sample_idx_400hz: int,
        current_time_ms: float | None,
        sid: int | str = 0,
    ) -> TriggerEvent | None:
        event = self._native.flush(
            current_frame_idx_50hz=int(current_frame_idx_50hz),
            current_sample_idx_400hz=int(current_sample_idx_400hz),
            current_time_ms=current_time_ms,
            sid=self._remember_sid(sid),
        )
        return self._convert_event(event)


def streaming_trigger_peaks(
    probabilities,
    threshold: float,
    holdoff_frames: int = 0,
    frame_offset: int = 0,
    sid: int | str = 0,
) -> list[int]:
    """Return peak indices through one bulk call into the C++ trigger core."""
    del sid
    return [
        int(value)
        for value in _live_core.streaming_trigger_peaks(
            np.asarray(probabilities, dtype=np.float32),
            float(threshold),
            int(holdoff_frames),
            int(frame_offset),
        )
    ]


__all__ = [
    "SiemensTriggerCleanStats",
    "StreamingHoldoffLocalMaxTrigger",
    "TriggerEvent",
    "clean_siemens_trigger_pairs",
    "streaming_trigger_peaks",
]
