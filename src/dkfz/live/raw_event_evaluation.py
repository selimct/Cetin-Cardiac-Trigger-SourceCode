# Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Developed at the German Cancer Research Center (DKFZ).
"""Compare model events at different frame rates on unchanged raw annotations."""

from __future__ import annotations

import numpy as np
import pandas as pd


def distribution(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {key: None for key in ("mean", "p50", "p95", "p99", "max")}
    return dict(
        zip(
            ("mean", "p50", "p95", "p99", "max"),
            map(
                float,
                [values.mean(), *np.percentile(values, [50, 95, 99]), values.max()],
            ),
        )
    )


def raw_timeline(raw, *, raw_hz=400, time_col="time_ms", time_unit="ms"):
    raw = raw.copy()
    if "sid" not in raw:
        raw["sid"] = 0
    if "raw_idx" not in raw:
        raw["raw_idx"] = (
            raw["idx"] if "idx" in raw else raw.groupby("sid", sort=False).cumcount()
        )
    if raw.duplicated(["sid", "raw_idx"]).any():
        raise ValueError("Raw indices must be unique within a recording ID")
    if time_unit not in ("ms", "s"):
        raise ValueError("time_unit must be ms or s")
    raw["time_ms"] = (
        raw[time_col] * (1000 if time_unit == "s" else 1)
        if time_col is not None and time_col in raw
        else raw.raw_idx * 1000 / raw_hz
    )
    return raw


def observable_intervals(log, frame_size_raw):
    intervals = {}
    for sid, group in log.groupby("sid", sort=False):
        starts = group.raw_idx.to_numpy(dtype=int)
        stop = group.get(
            "raw_frame_end_idx", group.raw_idx + frame_size_raw - 1
        ).to_numpy(dtype=int)
        cuts = np.r_[
            0, np.flatnonzero((np.diff(starts) != frame_size_raw)) + 1, len(group)
        ]
        intervals[sid] = [
            (int(starts[a]), int(stop[b - 1])) for a, b in zip(cuts[:-1], cuts[1:])
        ]
    return intervals


def common_intervals(*intervals):
    common = intervals[0]
    for other in intervals[1:]:
        common = {
            sid: [
                (max(a, c), min(b, d))
                for a, b in ranges
                for c, d in other.get(sid, [])
                if max(a, c) <= min(b, d)
            ]
            for sid, ranges in common.items()
        }
    return common


def _inside(values, intervals, margin=0):
    included = np.zeros(len(values), dtype=bool)
    for lo, hi in intervals:
        included |= (values >= lo + margin) & (values <= hi - margin)
    return included


def evaluate_raw_events(
    raw,
    events,
    log,
    *,
    frame_size_raw,
    target_col="target",
    tolerance_ms=20,
    raw_hz=400,
    intervals=None,
    time_col="time_ms",
    time_unit="ms",
):
    """Chronological one-to-one association, with the same raw-time radius for every model.

    Observable intervals describe *target* frames, excluding warmup and the
    final frame that has no subsequent context. Actual emissions are resolved
    separately against raw sample times. Edge guards avoid partial beat matches.
    """
    if not np.isfinite(tolerance_ms) or tolerance_ms < 0:
        raise ValueError("tolerance_ms must be finite and nonnegative")
    raw = raw_timeline(raw, raw_hz=raw_hz, time_col=time_col, time_unit=time_unit)
    if target_col not in raw:
        raise KeyError(f"Missing raw annotation column {target_col!r}")
    intervals = (
        observable_intervals(log, frame_size_raw) if intervals is None else intervals
    )
    matches, misses, fps = [], [], []
    errors, delays = [], []
    margin = int(np.ceil(tolerance_ms * raw_hz / 1000))
    for sid, group in raw.groupby("sid", sort=False):
        ranges = intervals.get(sid, [])
        truth = group[
            (group[target_col] > 0) & _inside(group.raw_idx.to_numpy(), ranges, margin)
        ].sort_values("time_ms")
        predicted = events[events.sid == sid].copy() if len(events) else events.copy()
        if len(predicted):
            predicted = predicted[
                _inside(predicted.peak_sample_idx_400hz.to_numpy(), ranges, margin)
            ]
            times = group.set_index("raw_idx").time_ms
            predicted["_peak_ms"] = predicted.peak_sample_idx_400hz.map(times)
            predicted["_emission_ms"] = predicted.emitted_on_sample_idx_400hz.map(times)
            if predicted[["_peak_ms", "_emission_ms"]].isna().any().any():
                raise ValueError("Events do not refer to the supplied raw recording")
            predicted = predicted.sort_values("_peak_ms")
        trows = list(truth.to_dict("records"))
        prows = list(predicted.to_dict("records"))
        ti = pi = 0
        while ti < len(trows) or pi < len(prows):
            t = trows[ti] if ti < len(trows) else None
            p = prows[pi] if pi < len(prows) else None
            delta = (
                p["_peak_ms"] - t["time_ms"]
                if p is not None and t is not None
                else None
            )
            if delta is not None and abs(delta) <= tolerance_ms:
                latency = p["_emission_ms"] - t["time_ms"]
                matches.append(
                    {
                        "sid": str(sid),
                        "annotation_raw_idx": int(t["raw_idx"]),
                        "peak_raw_idx": int(p["peak_sample_idx_400hz"]),
                        "emission_raw_idx": int(p["emitted_on_sample_idx_400hz"]),
                        "timing_error_ms": float(delta),
                        "stream_trigger_latency_ms": float(latency),
                    }
                )
                errors.append(delta)
                delays.append(latency)
                ti += 1
                pi += 1
            elif p is not None and (t is None or delta < -tolerance_ms):
                fps.append(
                    {"sid": str(sid), "peak_raw_idx": int(p["peak_sample_idx_400hz"])}
                )
                pi += 1
            else:
                misses.append(
                    {"sid": str(sid), "annotation_raw_idx": int(t["raw_idx"])}
                )
                ti += 1
    return {
        "schema": "dkfz-raw-event-metrics-v1",
        "tolerance_ms": float(tolerance_ms),
        "observable_raw_intervals": {str(k): v for k, v in intervals.items()},
        "matched": len(matches),
        "misses": len(misses),
        "false_positives": len(fps),
        "timing_error_ms": distribution(errors),
        "absolute_timing_error_ms": distribution(np.abs(errors)),
        "stream_trigger_latency_ms": distribution(delays),
        "actual_gpio_latency_ms": None,
        "matches": matches,
        "missed_annotations": misses,
        "false_positive_events": fps,
    }


def false_positive_changes(baseline, candidate, *, tolerance_samples):
    """Pair old/new false events on raw positions before identifying additions."""
    remaining = list(baseline)
    added = []
    for event in candidate:
        eligible = [
            (abs(old["peak_raw_idx"] - event["peak_raw_idx"]), i)
            for i, old in enumerate(remaining)
            if old["sid"] == event["sid"]
            and abs(old["peak_raw_idx"] - event["peak_raw_idx"]) <= tolerance_samples
        ]
        if eligible:
            remaining.pop(min(eligible)[1])
        else:
            added.append(event)
    return {
        "new_false_positive_events": added,
        "removed_false_positive_events": remaining,
    }


def explain_misses(
    raw,
    metrics,
    events,
    log,
    *,
    threshold,
    holdoff_frames,
    frame_size_raw,
    raw_hz=400,
    time_col="time_ms",
    time_unit="ms",
):
    """Expose boundary positions and observable holdoff cases, without inferring causes.

    A low score at a boundary does not establish that the boundary caused the
    miss. The evidence is reported separately from a preceding premature event.
    """
    raw = raw_timeline(raw, raw_hz=raw_hz, time_col=time_col, time_unit=time_unit)
    details = []
    for missed in metrics["missed_annotations"]:
        sid = missed["sid"]
        idx = missed["annotation_raw_idx"]
        rows = log[
            (log.sid.astype(str) == sid)
            & (log.raw_idx <= idx)
            & (log.get("raw_frame_end_idx", log.raw_idx + frame_size_raw - 1) >= idx)
        ]
        detail = dict(missed)
        if not len(rows):
            details.append(dict(detail, classification="no_target_frame"))
            continue
        row = rows.iloc[0]
        end = int(row.get("raw_frame_end_idx", row.raw_idx + frame_size_raw - 1))
        observed_end = int(row.get("observed_raw_frame_end_idx", end))
        detail.update(
            frame_start_raw_idx=int(row.raw_idx),
            frame_end_raw_idx=end,
            r_position_in_frame=idx - int(row.raw_idx),
            r_at_frame_end=idx == end,
            target_probability=float(row.prob),
        )
        previous = (
            events[
                (events.sid.astype(str) == sid)
                & (events.emitted_on_sample_idx_400hz <= observed_end)
            ]
            if len(events)
            else events
        )
        previous = (
            previous.sort_values("emitted_on_sample_idx_400hz")
            if len(previous)
            else previous
        )
        if (
            len(previous)
            and observed_end - int(previous.iloc[-1].emitted_on_sample_idx_400hz)
            < holdoff_frames * frame_size_raw
        ):
            last = previous.iloc[-1]
            times = raw[raw.sid.astype(str) == sid].set_index("raw_idx").time_ms
            early = (
                float(times.loc[int(last.peak_sample_idx_400hz)] - times.loc[idx])
                < -metrics["tolerance_ms"]
            )
            premature = early and int(last.emitted_on_sample_idx_400hz) < idx
            detail.update(
                classification=(
                    "holdoff_after_premature_trigger"
                    if premature
                    else "holdoff_after_previous_trigger"
                ),
                preceding_peak_raw_idx=int(last.peak_sample_idx_400hz),
                preceding_emission_raw_idx=int(last.emitted_on_sample_idx_400hz),
            )
        else:
            detail["classification"] = (
                "below_threshold_in_target_frame"
                if row.prob < threshold
                else "threshold_excursion_timing"
            )
        details.append(detail)
    return details
