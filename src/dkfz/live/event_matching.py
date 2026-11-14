# Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Developed at the German Cancer Research Center (DKFZ).

"""One-to-one TP/PE/oPE association shared by training and prediction reports."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

METRICS_SCHEMA_VERSION = 4
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
    """Option-2 outer tier at 100 Hz; retain the legacy rule at other rates."""
    if np.isclose(float(raw_hz) / int(frame_size_raw), 100.0):
        return 5
    return int(np.floor(0.040 * float(raw_hz) / int(frame_size_raw) + 1e-9))


def tp_tolerance_frames(frame_hz=50.0) -> int:
    return 1 if np.isclose(float(frame_hz), 100.0) else 0


def event_tolerances(settings, *, error_association_frames=None):
    """Resolve the current policy, migrating old 100 Hz reporting defaults.

    Schema-3 100 Hz configs recorded 0/2/4. Those counts must be recomputed,
    rather than retained as explicit overrides of the new 1/3/5 policy.
    Explicit CLI PE radii and schema-4 saved PE overrides remain supported.
    """
    raw_hz = float(settings.get("raw_hz", 400))
    size = int(settings.get("frame_size_raw", 8))
    exact = tp_tolerance_frames(raw_hz / size)
    if error_association_frames is None:
        error_association_frames = (
            3
            if exact and int(settings.get("metrics_schema_version", 0) or 0) < 4
            else settings.get("error_association_frames", 3 if exact else 1)
        )
    return dict(
        exact_tolerance_frames=exact,
        error_association_frames=int(error_association_frames),
        outer_error_association_frames=outer_association_frames(raw_hz, size),
    )


def _claim_within(predicted, targets, pred_indices, target_indices, radius):
    """Chronological one-to-one matching on the remaining sorted positions."""
    pairs, unmatched_pred, unmatched_target = [], [], []
    p = t = 0
    while p < len(pred_indices) and t < len(target_indices):
        pi, ti = pred_indices[p], target_indices[t]
        delta = int(predicted[pi] - targets[ti])
        if abs(delta) <= radius:
            pairs.append((pi, ti))
            p += 1
            t += 1
        elif delta < -radius:
            unmatched_pred.append(pi)
            p += 1
        else:
            unmatched_target.append(ti)
            t += 1
    return (
        pairs,
        unmatched_pred + pred_indices[p:],
        unmatched_target + target_indices[t:],
    )


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
    exact_tolerance_frames: int = EXACT_TOLERANCE_FRAMES,
) -> EventAssociation:
    """Claim identical frames, then tolerant TPs, PE, and oPE in that order.

    Identical frames have first priority, followed by remaining pairs within
    the TP radius. Each event is used at most once, including in later tiers.
    """
    radius = int(error_association_frames)
    if radius < 0:
        raise ValueError("error_association_frames must be zero or positive.")
    exact_radius = int(exact_tolerance_frames)
    if exact_radius < 0 or radius < exact_radius:
        raise ValueError(
            "Require 0 <= exact_tolerance_frames <= error_association_frames"
        )

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

    if exact_radius:
        nearby, remaining_pred, remaining_target = _claim_within(
            predicted, targets, remaining_pred, remaining_target, exact_radius
        )
        exact_matches += tuple(nearby)
    position_matches, remaining_pred, remaining_target = _claim_within(
        predicted, targets, remaining_pred, remaining_target, radius
    )
    outer_matches = []
    if (
        outer_error_association_frames is not None
        and int(outer_error_association_frames) > radius
    ):
        outer_matches, remaining_pred, remaining_target = _claim_within(
            predicted,
            targets,
            remaining_pred,
            remaining_target,
            int(outer_error_association_frames),
        )
    return EventAssociation(
        exact_matches=exact_matches,
        position_matches=tuple(position_matches),
        outer_position_matches=tuple(outer_matches),
        unmatched_prediction_indices=tuple(remaining_pred),
        unmatched_target_indices=tuple(remaining_target),
    )


def _safe_ratio(numerator: int, denominator: int) -> float:
    return 0.0 if denominator <= 0 else float(numerator / denominator)


def compute_event_position_metrics(
    predicted_positions,
    target_positions,
    *,
    error_association_frames: int = DEFAULT_ERROR_ASSOCIATION_FRAMES,
    outer_error_association_frames: int | None = DEFAULT_OUTER_ERROR_ASSOCIATION_FRAMES,
    exact_tolerance_frames: int = EXACT_TOLERANCE_FRAMES,
) -> dict[str, float | int | None]:
    """Return configured TP/PE/oPE counts and unique-error metrics.

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
        exact_tolerance_frames=exact_tolerance_frames,
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
        counts,
        error_association_frames,
        outer_error_association_frames,
        exact_tolerance_frames,
    )


def event_metrics_from_counts(
    counts,
    error_association_frames=1,
    outer_error_association_frames=2,
    exact_tolerance_frames=0,
):
    """Recompute pooled rates without counting either positional tier as FP/FN.

    Weighted DER is undefined without targets. Keep the error count available
    and return None for that rate so target-free false positives cannot look
    like a perfect score or produce non-finite JSON values.
    """
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
        "exact_tolerance_frames": int(exact_tolerance_frames),
        "error_association_frames": int(error_association_frames),
        "outer_error_association_frames": int(outer_error_association_frames or 0),
        "num_predicted_events": predicted,
        "num_target_events": targets,
        "num_associated_events": associated,
        **c,
        "unique_error_count": unique_errors,
        "weighted_error_count": cost,
        "weighted_detection_error": float(cost / targets) if targets else None,
        "exact_sensitivity": _safe_ratio(tp, targets),
        "exact_positive_predictive_value": _safe_ratio(tp, predicted),
        "unique_detection_error": _safe_ratio(unique_errors, predicted),
        "beat_sensitivity": _safe_ratio(associated, targets),
        "beat_positive_predictive_value": _safe_ratio(associated, predicted),
        "beat_detection_error": _safe_ratio(fp + fn, predicted),
    }
