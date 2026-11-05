# Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Developed at the German Cancer Research Center (DKFZ).

"""Exact-frame event association shared by training and prediction reports.

An exact match is the only true positive.  After exact matches are removed,
nearby prediction/target pairs may be associated solely to attribute one
positioning mistake instead of reporting an unrelated false positive and false
negative. Association grants true-positive credit only for beat metrics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

METRICS_SCHEMA_VERSION = 3
EXACT_TOLERANCE_FRAMES = 0
DEFAULT_ERROR_ASSOCIATION_FRAMES = 1
DEFAULT_OUTER_ERROR_ASSOCIATION_FRAMES = 2
ERROR_WEIGHTS = {"FP": 20, "FN": 10, "PE": 1, "oPE": 5}
EVENT_COUNT_KEYS = (
    "exact_true_positive",
    "position_error",
    "position_error_early",
    "position_error_late",
    "outer_position_error",
    "outer_position_error_early",
    "outer_position_error_late",
    "unmatched_false_positive",
    "unmatched_false_negative",
)


def outer_association_frames(raw_hz=400.0, frame_size_raw=8) -> int:
    """Largest integral frame offset within the inclusive 40 ms window."""
    return int(np.floor(0.040 * float(raw_hz) / int(frame_size_raw) + 1e-9))


@dataclass(frozen=True)
class EventAssociation:
    """One-to-one exact and positional matches plus unmatched event indices."""

    exact_matches: tuple[tuple[int, int], ...]
    position_matches: tuple[tuple[int, int], ...]
    unmatched_prediction_indices: tuple[int, ...]
    unmatched_target_indices: tuple[int, ...]
    outer_position_matches: tuple[tuple[int, int], ...] = ()


def _positions(values, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if array.size == 0:
        return np.asarray([], dtype=np.int64)
    if np.any(~np.isfinite(array.astype(float, copy=False))):
        raise ValueError(f"{name} must contain finite frame positions.")
    integer = array.astype(np.int64, copy=False)
    if not np.array_equal(array, integer):
        raise ValueError(f"{name} must contain integer frame positions.")
    if len(np.unique(integer)) != len(integer):
        raise ValueError(f"{name} must not contain duplicate frame positions.")
    return integer


def associate_event_positions(
    predicted_positions,
    target_positions,
    *,
    error_association_frames: int = DEFAULT_ERROR_ASSOCIATION_FRAMES,
    outer_error_association_frames: int | None = DEFAULT_OUTER_ERROR_ASSOCIATION_FRAMES,
) -> EventAssociation:
    """Associate events while reserving true positives for exact frames only.

    Exact pairs are claimed first. Remaining sorted events are paired
    one-to-one within ``error_association_frames``. These latter pairs are
    position errors, not true positives.
    """
    radius = int(error_association_frames)
    if radius < 0:
        raise ValueError("error_association_frames must be zero or positive.")

    predicted = _positions(predicted_positions, "predicted_positions")
    targets = _positions(target_positions, "target_positions")

    pred_order = np.argsort(predicted, kind="stable")
    target_order = np.argsort(targets, kind="stable")
    pred_by_position = {int(predicted[i]): int(i) for i in pred_order}
    target_by_position = {int(targets[i]): int(i) for i in target_order}

    exact_positions = sorted(set(pred_by_position) & set(target_by_position))
    exact_matches = tuple(
        (pred_by_position[position], target_by_position[position])
        for position in exact_positions
    )
    exact_pred = {pred_idx for pred_idx, _ in exact_matches}
    exact_target = {target_idx for _, target_idx in exact_matches}

    remaining_pred = [int(i) for i in pred_order if int(i) not in exact_pred]
    remaining_target = [int(i) for i in target_order if int(i) not in exact_target]

    position_matches: list[tuple[int, int]] = []
    unmatched_pred: list[int] = []
    unmatched_target: list[int] = []
    pred_cursor = 0
    target_cursor = 0

    while pred_cursor < len(remaining_pred) and target_cursor < len(remaining_target):
        pred_idx = remaining_pred[pred_cursor]
        target_idx = remaining_target[target_cursor]
        delta = int(predicted[pred_idx] - targets[target_idx])
        if abs(delta) <= radius:
            position_matches.append((pred_idx, target_idx))
            pred_cursor += 1
            target_cursor += 1
        elif delta < -radius:
            unmatched_pred.append(pred_idx)
            pred_cursor += 1
        else:
            unmatched_target.append(target_idx)
            target_cursor += 1

    unmatched_pred.extend(remaining_pred[pred_cursor:])
    unmatched_target.extend(remaining_target[target_cursor:])
    inner = EventAssociation(
        exact_matches=exact_matches,
        position_matches=tuple(position_matches),
        unmatched_prediction_indices=tuple(unmatched_pred),
        unmatched_target_indices=tuple(unmatched_target),
    )

    if (
        outer_error_association_frames is None
        or int(outer_error_association_frames) <= radius
    ):
        return inner
    outer = associate_event_positions(
        predicted[list(inner.unmatched_prediction_indices)],
        targets[list(inner.unmatched_target_indices)],
        error_association_frames=int(outer_error_association_frames),
        outer_error_association_frames=None,
    )
    pi, ti = inner.unmatched_prediction_indices, inner.unmatched_target_indices
    return EventAssociation(
        exact_matches=inner.exact_matches,
        position_matches=inner.position_matches,
        outer_position_matches=tuple(
            (pi[p], ti[t]) for p, t in outer.exact_matches + outer.position_matches
        ),
        unmatched_prediction_indices=tuple(
            pi[p] for p in outer.unmatched_prediction_indices
        ),
        unmatched_target_indices=tuple(ti[t] for t in outer.unmatched_target_indices),
    )


def _safe_ratio(numerator: int, denominator: int) -> float:
    return 0.0 if denominator <= 0 else float(numerator / denominator)


def compute_event_position_metrics(
    predicted_positions,
    target_positions,
    *,
    error_association_frames: int = DEFAULT_ERROR_ASSOCIATION_FRAMES,
    outer_error_association_frames: int | None = DEFAULT_OUTER_ERROR_ASSOCIATION_FRAMES,
) -> dict[str, float | int]:
    """Return exact-frame counts and unique-error metrics.

    ``position_error`` consumes one prediction and one target but contributes
    only once to ``unique_detection_error``. Beat metrics count it as a true
    positive, so ``beat_detection_error`` includes only unmatched FP and FN.
    """
    predicted = _positions(predicted_positions, "predicted_positions")
    targets = _positions(target_positions, "target_positions")
    association = associate_event_positions(
        predicted,
        targets,
        error_association_frames=error_association_frames,
        outer_error_association_frames=outer_error_association_frames,
    )
    offsets = [int(predicted[p] - targets[t]) for p, t in association.position_matches]
    outer_offsets = [
        int(predicted[p] - targets[t]) for p, t in association.outer_position_matches
    ]
    counts = {
        "exact_true_positive": len(association.exact_matches),
        "position_error": len(offsets),
        "position_error_early": sum(d < 0 for d in offsets),
        "position_error_late": sum(d > 0 for d in offsets),
        "outer_position_error": len(outer_offsets),
        "outer_position_error_early": sum(d < 0 for d in outer_offsets),
        "outer_position_error_late": sum(d > 0 for d in outer_offsets),
        "unmatched_false_positive": len(association.unmatched_prediction_indices),
        "unmatched_false_negative": len(association.unmatched_target_indices),
    }
    return event_metrics_from_counts(
        counts, error_association_frames, outer_error_association_frames
    )


def event_metrics_from_counts(
    counts, error_association_frames=1, outer_error_association_frames=2
):
    """Recompute pooled rates without counting either positional tier as FP/FN."""
    c = {key: int(counts.get(key, 0)) for key in EVENT_COUNT_KEYS}
    tp, pe, ope, fp, fn = (
        c[key]
        for key in (
            "exact_true_positive",
            "position_error",
            "outer_position_error",
            "unmatched_false_positive",
            "unmatched_false_negative",
        )
    )
    predicted, targets = tp + pe + ope + fp, tp + pe + ope + fn
    associated = tp + pe + ope
    unique_errors = pe + ope + fp + fn
    cost = 20 * fp + 10 * fn + pe + 5 * ope
    return {
        "metrics_schema_version": METRICS_SCHEMA_VERSION,
        "exact_tolerance_frames": EXACT_TOLERANCE_FRAMES,
        "error_association_frames": int(error_association_frames),
        "outer_error_association_frames": int(outer_error_association_frames or 0),
        "num_predicted_events": predicted,
        "num_target_events": targets,
        "num_associated_events": associated,
        **c,
        "unique_error_count": unique_errors,
        "weighted_error_count": cost,
        "weighted_detection_error": _safe_ratio(cost, targets),
        "exact_sensitivity": _safe_ratio(tp, targets),
        "exact_positive_predictive_value": _safe_ratio(tp, predicted),
        "unique_detection_error": _safe_ratio(unique_errors, predicted),
        "beat_sensitivity": _safe_ratio(associated, targets),
        "beat_positive_predictive_value": _safe_ratio(associated, predicted),
        "beat_detection_error": _safe_ratio(fp + fn, predicted),
    }
