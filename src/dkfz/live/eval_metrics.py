# Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Developed at the German Cancer Research Center (DKFZ).

"""Prediction summaries using exact-frame event-position semantics."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from dkfz.live.event_matching import (
    DEFAULT_ERROR_ASSOCIATION_FRAMES,
    EVENT_COUNT_KEYS,
    event_metrics_from_counts,
    EXACT_TOLERANCE_FRAMES,
    METRICS_SCHEMA_VERSION,
    associate_event_positions,
    compute_event_position_metrics,
)


def _safe_ratio(numerator: int, denominator: int) -> float:
    return 0.0 if denominator <= 0 else float(numerator / denominator)


def _coerce_binary(values, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.size == 0:
        return np.asarray([], dtype=np.int8)
    if np.any(pd.isna(array)):
        raise ValueError(f"{name} contains missing values; cannot compute metrics.")
    if array.dtype.kind in {"b", "i", "u"}:
        out = array.astype(np.int8, copy=False)
    else:
        out = array.astype(np.float64, copy=False)
        if not np.all(np.isin(out, (0.0, 1.0))):
            unique = np.unique(out)
            raise ValueError(
                f"{name} must be binary 0/1; got values {unique.tolist()}."
            )
        out = out.astype(np.int8)
    if not np.all(np.isin(out, (0, 1))):
        unique = np.unique(out)
        raise ValueError(f"{name} must be binary 0/1; got values {unique.tolist()}.")
    return out


def select_target_column(
    df: pd.DataFrame,
    preferred: str | None = None,
) -> str | None:
    """Pick the target column to evaluate against from a prediction log."""
    for col in (preferred, "target"):
        if col and col in df.columns:
            return col
    return None


def _frame_positions(group: pd.DataFrame) -> np.ndarray:
    if "frame_idx" in group.columns:
        return group["frame_idx"].to_numpy(dtype=int)
    return np.arange(len(group), dtype=int)


def _groups(df: pd.DataFrame):
    return (
        df.groupby("sid", sort=False, dropna=False)
        if "sid" in df.columns
        else [(0, df)]
    )


def metrics_from_event_counts(
    counts: Counter,
    error_association_frames: int,
    outer_error_association_frames: int = 2,
) -> dict:
    """Recompute event totals and rates from summed TP/PE/oPE/FP/FN."""
    return event_metrics_from_counts(
        counts, error_association_frames, outer_error_association_frames
    )


def build_event_position_summary(
    df: pd.DataFrame,
    *,
    pred_col: str = "pred",
    target_col: str = "target",
    error_association_frames: int = DEFAULT_ERROR_ASSOCIATION_FRAMES,
    outer_error_association_frames: int | None = None,
) -> dict:
    """Return exact-frame event metrics grouped by independent sequence."""
    if pred_col not in df.columns or target_col not in df.columns:
        raise KeyError(f"Missing {pred_col!r} or {target_col!r} in prediction log.")
    outer_error_association_frames = _outer_radius(df, outer_error_association_frames)
    counts: Counter = Counter()
    for _sid, group in _groups(df):
        positions = _frame_positions(group)
        pred_mask = _coerce_binary(group[pred_col].to_numpy(), pred_col) == 1
        target_mask = _coerce_binary(group[target_col].to_numpy(), target_col) == 1
        metrics = compute_event_position_metrics(
            positions[pred_mask],
            positions[target_mask],
            error_association_frames=error_association_frames,
            outer_error_association_frames=outer_error_association_frames,
        )
        for key in EVENT_COUNT_KEYS:
            counts[key] += int(metrics[key])
    return metrics_from_event_counts(
        counts, int(error_association_frames), outer_error_association_frames
    )


def _estimate_frame_period_ms(df: pd.DataFrame) -> float | None:
    if "frame_idx" not in df.columns or "time_ms" not in df.columns:
        return None
    estimates: list[float] = []
    for _sid, group in _groups(df):
        frames = pd.to_numeric(group["frame_idx"], errors="coerce").to_numpy(float)
        times = pd.to_numeric(group["time_ms"], errors="coerce").to_numpy(float)
        frame_delta = np.diff(frames)
        time_delta = np.diff(times)
        valid = np.isfinite(frame_delta) & np.isfinite(time_delta) & (frame_delta > 0)
        estimates.extend((time_delta[valid] / frame_delta[valid]).tolist())
    return float(np.median(estimates)) if estimates else None


def _outer_radius(df, supplied):
    if supplied is not None:
        return int(supplied)
    period = _estimate_frame_period_ms(df)
    return 2 if period is None or period <= 0 else int(np.floor(40.0 / period + 1e-9))


def build_emission_timing_summary(
    prediction_df: pd.DataFrame,
    events_df: pd.DataFrame,
    *,
    target_col: str,
    error_association_frames: int = DEFAULT_ERROR_ASSOCIATION_FRAMES,
    outer_error_association_frames: int | None = None,
) -> dict:
    """Summarize emission offsets for peak events associated with target beats."""
    required = {"peak_frame_idx_50hz", "emitted_on_frame_idx_50hz"}
    missing = sorted(required - set(events_df.columns))
    if missing:
        raise KeyError(f"Trigger events are missing column(s): {', '.join(missing)}")
    outer_error_association_frames = _outer_radius(
        prediction_df, outer_error_association_frames
    )
    frame_period_ms = _estimate_frame_period_ms(prediction_df)
    offsets: Counter = Counter()
    prediction_groups = {sid: group for sid, group in _groups(prediction_df)}
    event_groups = {sid: group for sid, group in _groups(events_df)}
    for sid, prediction_group in prediction_groups.items():
        target_positions = _frame_positions(prediction_group)[
            _coerce_binary(prediction_group[target_col].to_numpy(), target_col) == 1
        ]
        event_group = event_groups.get(sid)
        if event_group is None or event_group.empty:
            continue
        peak_positions = event_group["peak_frame_idx_50hz"].to_numpy(dtype=int)
        association = associate_event_positions(
            peak_positions,
            target_positions,
            error_association_frames=error_association_frames,
            outer_error_association_frames=outer_error_association_frames,
        )
        for pred_idx, target_idx in (
            association.exact_matches
            + association.position_matches
            + association.outer_position_matches
        ):
            emitted = int(event_group.iloc[pred_idx]["emitted_on_frame_idx_50hz"])
            offsets[emitted - int(target_positions[target_idx])] += 1
    rows = [
        {
            "offset_frames": int(offset),
            "offset_ms": (
                None if frame_period_ms is None else float(offset * frame_period_ms)
            ),
            "count": int(count),
        }
        for offset, count in sorted(offsets.items())
    ]
    total = sum(row["count"] for row in rows)
    for row in rows:
        row["share"] = _safe_ratio(int(row["count"]), total)
    return {
        "associated_predictions": int(total),
        "frame_period_ms": frame_period_ms,
        "offset_counts": rows,
    }


def build_prediction_metrics_report(
    prediction_df: pd.DataFrame,
    *,
    target_col: str,
    error_association_frames: int = DEFAULT_ERROR_ASSOCIATION_FRAMES,
    outer_error_association_frames: int | None = None,
    pred_col: str = "pred",
    threshold: float | None = None,
    events_df: pd.DataFrame | None = None,
) -> dict:
    """Return versioned frame diagnostics and exact event-position metrics."""
    report = {
        "metrics_schema_version": METRICS_SCHEMA_VERSION,
        "num_rows": int(len(prediction_df)),
        "target_col": target_col,
        "pred_col": pred_col,
        "threshold": None if threshold is None else float(threshold),
        "event_position": build_event_position_summary(
            prediction_df,
            pred_col=pred_col,
            target_col=target_col,
            error_association_frames=error_association_frames,
            outer_error_association_frames=outer_error_association_frames,
        ),
    }
    if events_df is not None:
        report["emission_timing"] = build_emission_timing_summary(
            prediction_df,
            events_df,
            target_col=target_col,
            error_association_frames=error_association_frames,
            outer_error_association_frames=outer_error_association_frames,
        )
    return report


def build_siemens_trigger_metrics_report(
    prediction_df: pd.DataFrame,
    *,
    target_col: str,
    error_association_frames: int = DEFAULT_ERROR_ASSOCIATION_FRAMES,
    outer_error_association_frames: int | None = None,
    source_col: str = "siemens_tr",
) -> dict:
    """Return exact-frame metrics for Siemens first-marker trigger events."""
    if source_col not in prediction_df.columns:
        raise KeyError(f"Missing {source_col!r} in prediction log.")
    values = pd.to_numeric(prediction_df[source_col], errors="raise")
    non_missing = values.dropna().to_numpy()
    unexpected = np.setdiff1d(np.unique(non_missing), np.array([0, 1, 2]))
    if unexpected.size:
        preview = ", ".join(str(value) for value in unexpected[:10])
        raise ValueError(f"{source_col} contains unsupported value(s): {preview}")
    eval_df = prediction_df.copy()
    pred_col = f"__{source_col}_first_marker"
    eval_df[pred_col] = values.fillna(0).eq(1).astype(np.int8)
    report = build_prediction_metrics_report(
        eval_df,
        target_col=target_col,
        pred_col=pred_col,
        error_association_frames=error_association_frames,
        outer_error_association_frames=outer_error_association_frames,
    )
    report["pred_col"] = source_col
    report["pred_positive_value"] = 1
    return report


def _pct(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def _offset_label(offset_frames: int, offset_ms: float | None) -> str:
    if offset_frames == 0:
        return "Target frame"
    direction = "late" if offset_frames > 0 else "early"
    frames = abs(offset_frames)
    label = f"{frames} frame{'s' if frames != 1 else ''} {direction}"
    if offset_ms is not None:
        label += f" ({offset_ms:+.0f} ms)"
    return label


def format_prediction_metrics_report(metrics: dict) -> str:
    """Return the table-oriented exact-event report used in logs and sidecars."""
    event = metrics["event_position"]
    threshold = metrics.get("threshold")
    lines = [
        "",
        "Validation trigger evaluation",
        "=============================",
        "",
        "Configuration",
        "-------------",
        f"{'Setting':<28} Value",
        f"{'-' * 27} {'-' * 28}",
        f"{'Threshold':<28} {'-' if threshold is None else f'{float(threshold):.6f}'}",
        f"{'Exact-match tolerance':<28} {event['exact_tolerance_frames']} frames",
        f"{'Error-association window':<28} ±{event['error_association_frames']} frame",
        f"{'Classification position':<28} Probability peak frame",
        f"{'Emission position':<28} Informational only",
        "",
        "Prediction-frame counts",
        "-----------------------",
        f"{'Category':<32} {'Count':>8}",
        f"{'-' * 32} {'-' * 8}",
        f"{'Exact true positives':<32} {event['exact_true_positive']:8,d}",
        f"{'Position errors':<32} {event['position_error']:8,d}",
        f"{'  Early position errors':<32} {event['position_error_early']:8,d}",
        f"{'  Late position errors':<32} {event['position_error_late']:8,d}",
        f"{'Outer position errors (oPE)':<32} {event['outer_position_error']:8,d}",
        f"{'Weighted error count':<32} {event['weighted_error_count']:8,d}",
        f"{'Unmatched false positives':<32} {event['unmatched_false_positive']:8,d}",
        f"{'Unmatched false negatives':<32} {event['unmatched_false_negative']:8,d}",
        "",
        "Exact-frame metrics",
        "-------------------",
        f"{'Metric':<9} {'Formula':<39} {'Result':>8}",
        f"{'-' * 9} {'-' * 39} {'-' * 8}",
        f"{'Se':<9} {'TP / (TP + PE + oPE + FN)':<39} {_pct(event['exact_sensitivity']):>8}",
        f"{'+P':<9} {'TP / (TP + PE + oPE + FP)':<39} {_pct(event['exact_positive_predictive_value']):>8}",
        f"{'wDER':<9} {'(20 FP + 10 FN + PE + 5 oPE) / targets':<39} {_pct(event['weighted_detection_error']):>8}",
        f"{'DER':<9} {'(PE + oPE + FP + FN) / predicted events':<39} {_pct(event['unique_detection_error']):>8}",
        "",
        "Beat-association metrics",
        "------------------------",
        "PE and oPE count as true positives for all beat metrics.",
        f"{'Metric':<9} {'Formula':<39} {'Result':>8}",
        f"{'-' * 9} {'-' * 39} {'-' * 8}",
        f"{'Beat Se':<9} {'(TP + PE + oPE) / target events':<39} {_pct(event['beat_sensitivity']):>8}",
        f"{'Beat +P':<9} {'(TP + PE + oPE) / predicted events':<39} {_pct(event['beat_positive_predictive_value']):>8}",
        f"{'Beat DER':<9} {'(FP + FN) / predicted events':<39} {_pct(event['beat_detection_error']):>8}",
    ]
    emission = metrics.get("emission_timing")
    if emission is not None:
        lines.extend(
            [
                "",
                "Emission timing relative to target",
                "----------------------------------",
                f"{'Emission position':<32} {'Count':>8} {'Share':>8}",
                f"{'-' * 32} {'-' * 8} {'-' * 8}",
            ]
        )
        for row in emission["offset_counts"]:
            label = _offset_label(int(row["offset_frames"]), row["offset_ms"])
            lines.append(f"{label:<32} {int(row['count']):8,d} {_pct(row['share']):>8}")
        lines.extend(
            [
                "",
                "Metric definitions",
                "------------------",
                "TP  Exact true positive",
                "PE  Position error",
                "oPE Outer position error (within ±40 ms)",
                "Weighted cost = 20 FP + 10 FN + PE + 5 oPE",
                "FP  Unmatched false positive",
                "FN  Unmatched false negative",
                "",
                "Classification uses only the probability peak frame.",
                "Emission timing does not affect TP, PE, FP, FN, Se, +P, or DER.",
            ]
        )
    return "\n".join(lines)


def write_prediction_metrics_report(
    report_text: str,
    *,
    text_path: str | Path,
    metrics: dict | None = None,
) -> Path:
    """Write the readable report and, when supplied, its structured JSON peer."""
    text_path = Path(text_path)
    text_path.parent.mkdir(parents=True, exist_ok=True)
    text_path.write_text(report_text + "\n", encoding="utf-8")
    if metrics is not None:
        json_path = text_path.with_suffix(".json")
        json_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return text_path
