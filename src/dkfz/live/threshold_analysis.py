# Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Developed at the German Cancer Research Center (DKFZ).

"""Shared post-selection event analysis and publication-quality plots."""

from __future__ import annotations

from collections import Counter
import hashlib
import os
from pathlib import Path
import tempfile
from typing import Literal

import numpy as np
import pandas as pd

from dkfz.core.network_helpers import ThresholdSelection, evaluate_threshold_grid
from dkfz.live.event_matching import (
    DEFAULT_ERROR_ASSOCIATION_FRAMES,
    EVENT_COUNT_KEYS,
    associate_event_positions,
)
from dkfz.live.eval_metrics import (
    metrics_from_event_counts,
    frame_event_tolerances,
    independent_prediction_groups,
)
from dkfz.live.triggers import StreamingHoldoffLocalMaxTrigger

DEFAULT_ERROR_PLOTS = 12
DEFAULT_ERROR_WINDOW_FRAMES = 25


def fixed_threshold_analysis(
    probabilities,
    target_peaks,
    *,
    threshold: float,
    sensitivity_efficiency: float = 0.95,
    holdoff_frames: int = 10,
    error_association_frames: int = 1,
    thresholds=None,
    additional_sequences=(),
    recovery_threshold: float | None = None,
    exact_tolerance_frames: int = 0,
    outer_error_association_frames: int = 2,
) -> ThresholdSelection:
    """Analyze a threshold grid while keeping the configured threshold fixed.

    This deliberately performs no threshold selection. The returned object uses
    the configured threshold and its metrics so the shared dashboard renderer
    can visualize a live evaluation without proposing or applying a replacement.
    Additional (probabilities, target_peaks) pairs are evaluated independently;
    their event counts are summed before recomputing the pooled rates.
    """
    threshold = float(threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    sensitivity_efficiency = float(sensitivity_efficiency)
    if not 0.0 < sensitivity_efficiency <= 1.0:
        raise ValueError("sensitivity_efficiency must be greater than 0 and at most 1")

    if thresholds is None:
        threshold_values = np.linspace(0.0, 1.0, 1001, dtype=np.float32)
    else:
        threshold_values = np.asarray(thresholds, dtype=np.float32)
    threshold_values = np.unique(
        np.append(threshold_values, np.asarray([threshold], dtype=np.float32))
    )
    rows = [
        dict(row)
        for row in evaluate_threshold_grid(
            probabilities,
            target_peaks,
            holdoff_frames=holdoff_frames,
            error_association_frames=error_association_frames,
            thresholds=threshold_values,
            recovery_threshold=recovery_threshold,
            exact_tolerance_frames=exact_tolerance_frames,
            outer_error_association_frames=outer_error_association_frames,
        )
    ]
    count_keys = EVENT_COUNT_KEYS
    for sequence_probabilities, sequence_targets in additional_sequences:
        sequence_rows = evaluate_threshold_grid(
            sequence_probabilities,
            sequence_targets,
            holdoff_frames=holdoff_frames,
            error_association_frames=error_association_frames,
            thresholds=threshold_values,
            recovery_threshold=recovery_threshold,
            exact_tolerance_frames=exact_tolerance_frames,
            outer_error_association_frames=outer_error_association_frames,
        )
        for row, sequence_row in zip(rows, sequence_rows):
            counts = Counter(
                {key: int(row[key]) + int(sequence_row[key]) for key in count_keys}
            )
            row.update(
                metrics_from_event_counts(
                    counts,
                    error_association_frames,
                    outer_error_association_frames,
                    exact_tolerance_frames,
                )
            )
            signature = f"{row['predicted_event_signature']}:{sequence_row['predicted_event_signature']}"
            row["predicted_event_signature"] = hashlib.sha256(
                signature.encode()
            ).hexdigest()[:16]
    fixed_index = int(
        np.argmin(
            np.abs(
                np.asarray([row["threshold"] for row in rows], dtype=float) - threshold
            )
        )
    )
    selectable = [row for row in rows if bool(row["selectable_threshold"])]
    best_sensitivity = max(
        (float(row["exact_sensitivity"]) for row in selectable),
        default=0.0,
    )
    effective_floor = sensitivity_efficiency * best_sensitivity
    for index, row in enumerate(rows):
        row["sensitivity_floor_met"] = bool(
            row["selectable_threshold"]
            and float(row["exact_sensitivity"]) >= effective_floor
        )
        row["selected"] = index == fixed_index

    fixed_row = rows[fixed_index]
    metrics = {
        key: value
        for key, value in fixed_row.items()
        if key
        not in {
            "threshold",
            "predicted_event_signature",
            "selectable_threshold",
            "sensitivity_floor_met",
            "selected",
        }
    }
    return ThresholdSelection(
        threshold=float(fixed_row["threshold"]),
        objective="fixed",
        sensitivity_efficiency=sensitivity_efficiency,
        best_attainable_sensitivity=best_sensitivity,
        effective_sensitivity_floor=effective_floor,
        metrics=metrics,
        search_rows=tuple(rows),
        recovery_threshold=recovery_threshold,
    )


def selection_rows(selection: ThresholdSelection) -> pd.DataFrame:
    """Return the threshold search, deriving missing beat rates from counts."""
    beat_fields = (
        "beat_sensitivity",
        "beat_positive_predictive_value",
        "beat_detection_error",
    )
    rows = []
    for search_row in selection.search_rows:
        row = dict(search_row)
        missing = [field for field in beat_fields if field not in row]
        if missing:
            derived = metrics_from_event_counts(
                Counter(row),
                error_association_frames=int(
                    row.get(
                        "error_association_frames", DEFAULT_ERROR_ASSOCIATION_FRAMES
                    )
                ),
            )
            row.update({field: derived[field] for field in missing})
        rows.append(row)
    return pd.DataFrame(rows)


def operating_regions(selection: ThresholdSelection) -> pd.DataFrame:
    """Compress adjacent candidates that emit exactly the same event frames."""
    search = selection_rows(selection).sort_values("threshold").reset_index(drop=True)
    split_columns = [
        "predicted_event_signature",
        "selectable_threshold",
        "sensitivity_floor_met",
    ]
    region_ids = (
        search[split_columns].ne(search[split_columns].shift()).any(axis=1).cumsum()
    )
    metric_columns = [
        "num_predicted_events",
        "num_target_events",
        "exact_true_positive",
        "position_error",
        "position_error_early",
        "position_error_late",
        "outer_position_error",
        "outer_position_error_early",
        "outer_position_error_late",
        "weighted_error_count",
        "weighted_detection_error",
        "unmatched_false_positive",
        "unmatched_false_negative",
        "unique_error_count",
        "exact_sensitivity",
        "exact_positive_predictive_value",
        "unique_detection_error",
        "beat_sensitivity",
        "beat_positive_predictive_value",
        "beat_detection_error",
    ]
    rows: list[dict] = []
    for region_id, group in search.groupby(region_ids, sort=False):
        first = group.iloc[0]
        rows.append(
            {
                "region_id": int(region_id),
                "threshold_start": float(group["threshold"].min()),
                "threshold_end": float(group["threshold"].max()),
                "threshold_samples": int(len(group)),
                "selectable": bool(first["selectable_threshold"]),
                "eligible": bool(first["sensitivity_floor_met"]),
                "recommended": bool(group["selected"].any()),
                "predicted_event_signature": first["predicted_event_signature"],
                **{column: first[column] for column in metric_columns},
            }
        )
    return pd.DataFrame(rows)


def _scalar(value):
    return value.item() if isinstance(value, np.generic) else value


def classify_validation_events(
    validation_df: pd.DataFrame,
    *,
    threshold: float,
    target_col: str,
    holdoff_frames: int,
    error_association_frames: int,
    outer_error_association_frames: int | None = None,
    recovery_threshold: float | None = None,
    exact_tolerance_frames: int | None = None,
) -> pd.DataFrame:
    """Return one TP/PE/FP/FN row per event at a selected threshold."""
    required = {"sid", "prob", target_col}
    missing = sorted(required - set(validation_df.columns))
    if missing:
        raise KeyError(f"Validation parquet is missing column(s): {', '.join(missing)}")

    policy = frame_event_tolerances(
        validation_df,
        error_association_frames=error_association_frames,
        outer_error_association_frames=outer_error_association_frames,
        exact_tolerance_frames=exact_tolerance_frames,
    )
    event_rows: list[dict] = []
    groups = list(independent_prediction_groups(validation_df))
    split_streams = (
        len(groups) > validation_df.sid.nunique(dropna=False)
        and "raw_idx" in validation_df
    )
    for sid, original_group in groups:
        group = original_group.reset_index(drop=True)
        frame_positions = (
            group["frame_idx"].to_numpy(dtype=int)
            if "frame_idx" in group.columns
            else np.arange(len(group), dtype=int)
        )
        raw_positions = (
            group["raw_idx"].to_numpy(dtype=int)
            if "raw_idx" in group.columns
            else frame_positions
        )
        times = (
            group["time_ms"].to_numpy(dtype=float)
            if "time_ms" in group.columns
            else np.full(len(group), np.nan)
        )
        delayed = "prediction_delay_frames" in group and bool(
            group.prediction_delay_frames.max()
        )
        observed_frames = (
            group["observed_frame_idx"].to_numpy(dtype=int)
            if delayed
            else frame_positions
        )
        observed_raw = (
            group["observed_raw_frame_end_idx"].to_numpy(dtype=int)
            if delayed
            else raw_positions
        )
        observed_times = (
            group["observed_time_ms"].to_numpy(dtype=float) if delayed else times
        )
        if delayed:
            raw_positions = group["raw_frame_end_idx"].to_numpy(dtype=int)
        probabilities = group["prob"].to_numpy(dtype=float)
        target_positions = frame_positions[group[target_col].to_numpy(dtype=float) == 1]

        trigger = StreamingHoldoffLocalMaxTrigger(
            threshold=float(threshold),
            holdoff_frames=int(holdoff_frames),
            recovery_threshold=recovery_threshold,
        )
        events = []
        for index, probability in enumerate(probabilities):
            event = trigger.update(
                probability=float(probability),
                current_frame_idx_50hz=int(observed_frames[index]),
                current_sample_idx_400hz=int(observed_raw[index]),
                current_time_ms=(
                    None
                    if np.isnan(observed_times[index])
                    else float(observed_times[index])
                ),
                sid=_scalar(sid),
                **(
                    dict(
                        target_frame_idx=int(frame_positions[index]),
                        target_sample_idx=int(raw_positions[index]),
                        target_time_ms=(
                            None if np.isnan(times[index]) else float(times[index])
                        ),
                    )
                    if delayed
                    else {}
                ),
                recovery_probability=(
                    float(group["recovery_prob"].iloc[index])
                    if "recovery_prob" in group
                    else None
                ),
            )
            if event is not None:
                events.append(event)
        if len(group):
            event = trigger.flush(
                current_frame_idx_50hz=int(observed_frames[-1]),
                current_sample_idx_400hz=int(observed_raw[-1]),
                current_time_ms=(
                    None if np.isnan(observed_times[-1]) else float(observed_times[-1])
                ),
                sid=_scalar(sid),
            )
            if event is not None:
                events.append(event)

        segment_metadata = (
            dict(
                segment_raw_start=int(group.raw_idx.min()),
                segment_raw_end=int(group.raw_idx.max()),
            )
            if split_streams
            else {}
        )
        predicted_positions = np.asarray(
            [event.peak_frame_idx_50hz for event in events], dtype=int
        )
        association = associate_event_positions(
            predicted_positions,
            target_positions,
            error_association_frames=error_association_frames,
            outer_error_association_frames=policy["outer_error_association_frames"],
            exact_tolerance_frames=policy["exact_tolerance_frames"],
        )
        source_file = (
            str(group.iloc[0]["source_file"])
            if "source_file" in group.columns and len(group)
            else None
        )

        def add_prediction(
            category: str,
            pred_index: int,
            target_index: int | None,
        ) -> None:
            event = events[pred_index]
            predicted_frame = int(predicted_positions[pred_index])
            target_frame = (
                None if target_index is None else int(target_positions[target_index])
            )
            event_rows.append(
                {
                    **segment_metadata,
                    "sid": _scalar(sid),
                    "source_file": source_file,
                    "category": category,
                    "target_frame": target_frame,
                    "predicted_frame": predicted_frame,
                    "position_offset_frames": (
                        None if target_frame is None else predicted_frame - target_frame
                    ),
                    "emitted_on_frame": int(event.emitted_on_frame_idx_50hz),
                    "emission_offset_frames": int(
                        event.emitted_on_frame_idx_50hz - predicted_frame
                    ),
                    "probability": float(event.probability),
                }
            )

        for pred_index, target_index in association.exact_matches:
            add_prediction("TP", pred_index, target_index)
        for pred_index, target_index in association.position_matches:
            add_prediction("PE", pred_index, target_index)
        for pred_index, target_index in association.outer_position_matches:
            add_prediction("oPE", pred_index, target_index)
        for pred_index in association.unmatched_prediction_indices:
            add_prediction("FP", pred_index, None)
        for target_index in association.unmatched_target_indices:
            event_rows.append(
                {
                    **segment_metadata,
                    "sid": _scalar(sid),
                    "source_file": source_file,
                    "category": "FN",
                    "target_frame": int(target_positions[target_index]),
                    "predicted_frame": None,
                    "position_offset_frames": None,
                    "emitted_on_frame": None,
                    "emission_offset_frames": None,
                    "probability": None,
                }
            )

    columns = [
        "sid",
        "source_file",
        "category",
        "target_frame",
        "predicted_frame",
        "position_offset_frames",
        "emitted_on_frame",
        "emission_offset_frames",
        "probability",
    ]
    if split_streams:
        columns.extend(["segment_raw_start", "segment_raw_end"])
    return pd.DataFrame(event_rows, columns=columns).sort_values(
        ["sid", "target_frame", "predicted_frame"],
        na_position="last",
        ignore_index=True,
    )


def _matplotlib_modules():
    cache_dir = Path(tempfile.gettempdir()) / "dkfz-threshold-matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    os.environ.setdefault("MPLBACKEND", "Agg")
    try:
        from matplotlib.backends.backend_pdf import PdfPages
        from dkfz.core.utils import getpyplot
    except ImportError as exc:
        raise RuntimeError(
            "Threshold plots require matplotlib and mplhep; install the project "
            "analysis or train extra."
        ) from exc
    return getpyplot(), PdfPages


def _legend(axis, *, fontsize: int, loc: str, ncols: int = 1):
    legend = axis.legend(
        fontsize=fontsize,
        loc=loc,
        ncols=ncols,
        frameon=True,
        facecolor="white",
        framealpha=0.78,
        edgecolor="0.7",
    )
    legend.get_frame().set_linewidth(0.8)
    return legend


def _style_axis(axis, *, title_size: int = 18) -> None:
    axis.title.set_fontsize(title_size)
    axis.xaxis.label.set_fontsize(15)
    axis.yaxis.label.set_fontsize(15)
    axis.tick_params(
        axis="both",
        which="major",
        labelsize=12,
        width=0.9,
        length=5,
    )
    axis.tick_params(axis="both", which="minor", width=0.7, length=3)
    for spine in axis.spines.values():
        spine.set_linewidth(0.9)


def _plot_threshold_event_errors(axis, search: pd.DataFrame) -> None:
    threshold = search["threshold"].to_numpy(dtype=float)
    axis.plot(
        threshold,
        search["unmatched_false_positive"],
        color="tab:red",
        linewidth=1.5,
        label="FP",
    )
    if "outer_position_error" in search:
        axis.plot(
            threshold,
            search["outer_position_error"],
            color="tab:cyan",
            linewidth=1.5,
            label="oPE",
        )
    axis.plot(
        threshold,
        search["position_error"],
        color="tab:orange",
        linewidth=1.5,
        label="PE",
    )
    axis.plot(
        threshold,
        search["unmatched_false_negative"],
        color="tab:purple",
        linewidth=1.5,
        label="FN",
    )
    axis.set(title="Unique event errors", xlabel="Threshold", ylabel="Count")
    _legend(axis, fontsize=11, loc="upper left")


def _plot_threshold_rates(
    axis, search: pd.DataFrame, selection: ThresholdSelection
) -> None:
    threshold = search["threshold"].to_numpy(dtype=float)
    axis.plot(
        threshold,
        100.0 * search["exact_sensitivity"],
        label="TP Se",
        color="tab:blue",
        linewidth=1.5,
    )
    axis.plot(
        threshold,
        100.0 * search["exact_positive_predictive_value"],
        label="TP +P",
        color="tab:green",
        linewidth=1.5,
    )
    axis.plot(
        threshold,
        100.0 * search["unique_detection_error"],
        label="Event DER (display clipped at 105%)",
        color="tab:red",
        linewidth=1.5,
    )
    for metric, label, color in (
        ("beat_sensitivity", "Beat Se", "tab:blue"),
        ("beat_positive_predictive_value", "Beat +P", "tab:green"),
        ("beat_detection_error", "Beat DER (display clipped at 105%)", "tab:red"),
    ):
        axis.plot(
            threshold,
            100.0 * search[metric],
            label=label,
            color=color,
            linestyle="--",
            linewidth=1.5,
        )
    axis.axhline(
        100.0 * selection.effective_sensitivity_floor,
        color="black",
        linestyle=":",
        linewidth=1.2,
        label=(f"{100.0 * selection.sensitivity_efficiency:.0f}% attainable Se floor"),
    )
    axis.set(
        title="Efficiency and error rates",
        xlabel="Threshold",
        ylabel="Percent",
        ylim=(-2, 105),
    )
    _legend(axis, fontsize=10, loc="center left")


def _plot_threshold_tradeoff(
    axis, search: pd.DataFrame, selection: ThresholdSelection, selected_label: str
) -> None:
    eligible = search["sensitivity_floor_met"].to_numpy(dtype=bool)
    scatter = axis.scatter(
        search.loc[eligible, "unmatched_false_positive"],
        100.0 * search.loc[eligible, "exact_sensitivity"],
        c=search.loc[eligible, "threshold"],
        cmap="viridis",
        s=20,
        label="Eligible",
    )
    axis.scatter(
        search.loc[~eligible, "unmatched_false_positive"],
        100.0 * search.loc[~eligible, "exact_sensitivity"],
        color="lightgray",
        s=10,
        label="Outside floor",
    )
    chosen = selection.metrics
    # Training grids select the current head first. Recovery has its own audit
    # and FP budget; fixed runtime dashboards replay both heads together.
    axis.scatter(
        [chosen["unmatched_false_positive"]],
        [100.0 * chosen["exact_sensitivity"]],
        marker="x",
        color="red",
        linewidths=1.5,
        s=150,
        label=selected_label,
        zorder=5,
    )
    frontier_source = search[search["selectable_threshold"]].copy()
    frontier_source = frontier_source.sort_values(
        ["unmatched_false_positive", "exact_sensitivity"],
        ascending=[True, False],
    )
    frontier_rows = []
    best_seen = -1.0
    for _false_positive, group in frontier_source.groupby(
        "unmatched_false_positive", sort=True
    ):
        row = group.iloc[0]
        sensitivity = float(row["exact_sensitivity"])
        if sensitivity > best_seen:
            frontier_rows.append(row)
            best_seen = sensitivity
    if frontier_rows:
        frontier = pd.DataFrame(frontier_rows)
        axis.plot(
            frontier["unmatched_false_positive"],
            100.0 * frontier["exact_sensitivity"],
            color="tab:blue",
            linestyle="--",
            linewidth=1.2,
            label="Pareto frontier",
        )
    axis.set(
        title="False-positive/sensitivity trade-off",
        xlabel="Unmatched FP",
        ylabel="TP Se (%)",
    )
    _legend(axis, fontsize=10, loc="lower right")
    colorbar = axis.figure.colorbar(scatter, ax=axis)
    colorbar.set_label("Threshold", fontsize=14)
    colorbar.ax.tick_params(labelsize=11, width=0.8, length=4)
    colorbar.outline.set_linewidth(0.8)


def _decorate_threshold_panel(
    axis,
    search: pd.DataFrame,
    selection: ThresholdSelection,
    previous_threshold: float | None,
    applied_threshold: float | None,
) -> None:
    threshold = search["threshold"].to_numpy(dtype=float)
    eligible = search["sensitivity_floor_met"].to_numpy(dtype=bool)
    _style_axis(axis)
    if eligible.any():
        axis.fill_between(
            threshold,
            0,
            1,
            where=eligible,
            color="tab:green",
            alpha=0.08,
            transform=axis.get_xaxis_transform(),
        )
    if previous_threshold is not None:
        axis.axvline(
            previous_threshold,
            color="tab:gray",
            linestyle="--",
            linewidth=1.0,
        )
    axis.axvline(
        selection.threshold,
        color="black",
        linestyle="-",
        linewidth=1.2,
    )
    if applied_threshold is not None and not np.isclose(
        applied_threshold, selection.threshold
    ):
        axis.axvline(
            applied_threshold,
            color="tab:cyan",
            linestyle="-.",
            linewidth=1.2,
        )


def write_threshold_dashboard(
    path: Path,
    *,
    selection: ThresholdSelection,
    presentation: Literal["training", "comparison", "fixed"] = "comparison",
    previous_threshold: float | None = None,
    previous_metrics: dict[str, float | int] | None = None,
    applied_threshold: float | None = None,
    separate_plots: bool = False,
) -> None:
    """Write a vector dashboard and optionally render its three plots separately.

    Standalone plots use fresh figures and the same data/renderers as the
    dashboard. Their names append _event_errors, _rates, and _tradeoff.
    """
    if presentation not in {"training", "comparison", "fixed"}:
        raise ValueError("presentation must be 'training', 'comparison', or 'fixed'.")
    if presentation == "comparison" and (
        previous_threshold is None or previous_metrics is None
    ):
        raise ValueError("comparison presentation requires previous threshold metrics.")

    plt, _pdf_pages = _matplotlib_modules()
    search = selection_rows(selection).sort_values("threshold")
    selected_label = {
        "training": "Selected",
        "comparison": "Recommended",
        "fixed": "Configured",
    }[presentation]

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(15.5, 10.5),
        constrained_layout=True,
    )
    count_axis, rate_axis, tradeoff_axis, summary_axis = axes.flat

    panels = (
        ("event_errors", lambda axis: _plot_threshold_event_errors(axis, search)),
        ("rates", lambda axis: _plot_threshold_rates(axis, search, selection)),
        (
            "tradeoff",
            lambda axis: _plot_threshold_tradeoff(
                axis, search, selection, selected_label
            ),
        ),
    )
    for axis, (_name, render) in zip((count_axis, rate_axis, tradeoff_axis), panels):
        render(axis)
        _decorate_threshold_panel(
            axis, search, selection, previous_threshold, applied_threshold
        )

    chosen = selection.metrics
    summary_axis.axis("off")
    if presentation == "training":
        summary_lines = [
            "Threshold selection",
            "",
            f"Selected threshold: {selection.threshold:.6f}",
            f"Objective: {selection.objective}",
            f"fp_threshold: {selection.fp_threshold}",
            f"der_threshold: {selection.der_threshold}",
            f"oPE: {chosen.get('outer_position_error', 'not recorded')}",
            f"TP/PE/oPE radii: {chosen.get('exact_tolerance_frames', 0)}/{chosen.get('error_association_frames', 1)}/{chosen.get('outer_error_association_frames', 2)} frames",
            f"Weighted cost: {chosen.get('weighted_error_count', 'not recorded')}",
            "",
            f"Best attainable TP Se: {selection.best_attainable_sensitivity:.2%}",
            f"Sensitivity efficiency: {selection.sensitivity_efficiency:.2%}",
            f"Effective TP Se floor: {selection.effective_sensitivity_floor:.2%}",
            "",
            "Selected: "
            f"PE={int(chosen['position_error'])}, "
            f"FP={int(chosen['unmatched_false_positive'])}, "
            f"FN={int(chosen['unmatched_false_negative'])}",
            f"Se={float(chosen['exact_sensitivity']):.2%}, "
            f"+P={float(chosen['exact_positive_predictive_value']):.2%}, "
            f"DER={float(chosen['unique_detection_error']):.2%}",
        ]
        if selection.recovery_max_fp_increase is not None:
            summary_lines += [
                "",
                "Above: primary-head threshold search",
                f"Recovery threshold: {selection.recovery_threshold}",
                f"Recovery FP allowance: {selection.recovery_max_fp_increase:.0%}",
                f"Minimum extra FP allowance: {selection.recovery_min_extra_fp_per_minute or 0:g}/minute",
                f"Evaluated validation minutes: {selection.recovery_validation_minutes}",
                "Recovery requires fewer FN and fewer combined FP+FN.",
            ]
    elif presentation == "fixed":
        summary_lines = [
            "Fixed-threshold live evaluation",
            "",
            f"Configured threshold: {selection.threshold:.6f}",
            "Threshold selection: not performed",
            "",
            f"Best attainable TP Se: {selection.best_attainable_sensitivity:.2%}",
            f"Reference sensitivity efficiency: {selection.sensitivity_efficiency:.2%}",
            f"Reference TP Se floor: {selection.effective_sensitivity_floor:.2%}",
            "",
            "Configured threshold: "
            f"PE={int(chosen['position_error'])}, "
            f"FP={int(chosen['unmatched_false_positive'])}, "
            f"FN={int(chosen['unmatched_false_negative'])}",
            f"Se={float(chosen['exact_sensitivity']):.2%}, "
            f"+P={float(chosen['exact_positive_predictive_value']):.2%}, "
            f"DER={float(chosen['unique_detection_error']):.2%}",
        ]
    else:
        assert previous_threshold is not None and previous_metrics is not None
        summary_lines = [
            "Threshold analysis",
            "",
            f"Previous threshold: {previous_threshold:.6f}",
            f"Recommended threshold: {selection.threshold:.6f}",
            (
                "Applied threshold: not applied"
                if applied_threshold is None
                else f"Applied threshold: {applied_threshold:.6f}"
            ),
            "",
            f"Best attainable TP Se: {selection.best_attainable_sensitivity:.2%}",
            f"Sensitivity efficiency: {selection.sensitivity_efficiency:.2%}",
            f"Effective TP Se floor: {selection.effective_sensitivity_floor:.2%}",
            "",
            "Previous: "
            f"PE={int(previous_metrics['position_error'])}, "
            f"FP={int(previous_metrics['unmatched_false_positive'])}, "
            f"FN={int(previous_metrics['unmatched_false_negative'])}",
            "Recommended: "
            f"PE={int(chosen['position_error'])}, "
            f"FP={int(chosen['unmatched_false_positive'])}, "
            f"FN={int(chosen['unmatched_false_negative'])}",
        ]
    summary_lines.extend(
        [
            "",
            "Classification uses the predicted peak frame.",
            "Emission timing is informational only.",
        ]
    )
    summary_axis.text(
        0.0,
        1.0,
        "\n".join(summary_lines),
        va="top",
        family="monospace",
        fontsize=11,
    )
    figure.savefig(
        path,
        bbox_inches="tight",
        metadata={
            "Title": (
                "ECG trigger fixed-threshold evaluation"
                if presentation == "fixed"
                else "ECG trigger threshold selection"
            ),
            "Author": "DKFZ live ECG analysis pipeline",
            "Subject": "Event timing tiers and threshold efficiency",
        },
    )
    plt.close(figure)

    if separate_plots:
        for name, render in panels:
            panel_figure, panel_axis = plt.subplots(
                figsize=(8.0, 5.5), constrained_layout=True
            )
            try:
                render(panel_axis)
                _decorate_threshold_panel(
                    panel_axis, search, selection, previous_threshold, applied_threshold
                )
                panel_figure.savefig(
                    path.with_name(f"{path.stem}_{name}.pdf"),
                    bbox_inches="tight",
                    metadata={
                        "Title": panel_axis.get_title(),
                        "Author": "DKFZ live ECG analysis pipeline",
                        "Subject": "Event timing tiers and threshold efficiency",
                    },
                )
            finally:
                plt.close(panel_figure)


def _representative_errors(event_df: pd.DataFrame, max_plots: int) -> pd.DataFrame:
    category_rank = {"FP": 0, "FN": 1, "oPE": 2, "PE": 3}
    errors = event_df[event_df["category"].isin(category_rank)].copy()
    errors["category_rank"] = errors["category"].map(category_rank)
    errors["center_frame"] = errors["predicted_frame"].fillna(errors["target_frame"])
    errors = errors.sort_values(
        ["category_rank", "probability", "sid", "center_frame"],
        ascending=[True, False, True, True],
        na_position="last",
    )
    selected_indices: list[int] = []
    category_indices = {
        category: errors.index[errors["category"] == category].tolist()
        for category in ("FP", "FN", "oPE", "PE")
    }
    max_plots = max(0, int(max_plots))
    depth = 0
    while len(selected_indices) < max_plots:
        added = False
        for category in ("FP", "FN", "oPE", "PE"):
            indices = category_indices[category]
            if depth < len(indices):
                selected_indices.append(indices[depth])
                added = True
                if len(selected_indices) == max_plots:
                    break
        if not added:
            break
        depth += 1
    return errors.loc[selected_indices]


def write_error_examples(
    path: Path,
    *,
    validation_df: pd.DataFrame,
    event_df: pd.DataFrame,
    threshold: float,
    target_col: str,
    max_plots: int = DEFAULT_ERROR_PLOTS,
    window_frames: int = DEFAULT_ERROR_WINDOW_FRAMES,
    recovery_threshold: float | None = None,
) -> int:
    """Write a limited, representative error-centered timeline PDF."""
    plt, PdfPages = _matplotlib_modules()
    errors = _representative_errors(event_df, max_plots)

    written = 0
    with PdfPages(path) as pdf:
        if errors.empty:
            figure, axis = plt.subplots(figsize=(10, 4))
            axis.axis("off")
            axis.text(0.5, 0.5, "No FP, PE, or FN events", ha="center", va="center")
            pdf.savefig(figure)
            plt.close(figure)
        for _, error in errors.iterrows():
            sid = error["sid"]
            center = int(error["center_frame"])
            group = validation_df[validation_df["sid"] == sid].copy()
            if "segment_raw_start" in error and pd.notna(error["segment_raw_start"]):
                group = group[
                    group.raw_idx.between(
                        error["segment_raw_start"], error["segment_raw_end"]
                    )
                ].copy()
            if "frame_idx" not in group.columns:
                group["frame_idx"] = np.arange(len(group), dtype=int)
            target_frames = group.loc[
                group[target_col].to_numpy(dtype=float) == 1, "frame_idx"
            ].to_numpy(dtype=int)
            nearest_target = None
            nearest_offset = None
            if len(target_frames):
                nearest_target = int(
                    target_frames[np.argmin(np.abs(target_frames - center))]
                )
                nearest_offset = nearest_target - center

            view_start = center - int(window_frames)
            view_end = center + int(window_frames)
            if (
                str(error["category"]) == "FP"
                and nearest_target is not None
                and abs(int(nearest_offset)) <= 4 * int(window_frames)
            ):
                context = max(3, int(window_frames) // 3)
                view_start = min(view_start, nearest_target - context)
                view_end = max(view_end, nearest_target + context)
            window = group[group["frame_idx"].between(view_start, view_end)]
            nearby = event_df[
                (event_df["sid"] == sid)
                & (
                    event_df["predicted_frame"]
                    .fillna(event_df["target_frame"])
                    .between(view_start, view_end)
                )
            ]

            if "segment_raw_start" in error and pd.notna(error["segment_raw_start"]):
                nearby = nearby[nearby.segment_raw_start.eq(error["segment_raw_start"])]
            figure, axis = plt.subplots(figsize=(12, 4.5))
            axis.plot(window["frame_idx"], window["prob"], color="black", linewidth=1)
            if "recovery_prob" in window:
                axis.plot(
                    window["frame_idx"] - 1,
                    window["recovery_prob"],
                    color="tab:orange",
                    linewidth=1,
                    label="Recovery (previous frame)",
                )
                if recovery_threshold is not None:
                    axis.axhline(
                        recovery_threshold,
                        color="tab:orange",
                        linestyle=":",
                        label="Recovery threshold",
                    )
            axis.axhline(
                threshold,
                color="tab:blue",
                linestyle="--",
                linewidth=1.2,
                label="Threshold",
            )
            targets_in_view = target_frames[
                (target_frames >= view_start) & (target_frames <= view_end)
            ]
            for target_index, target_frame in enumerate(targets_in_view):
                is_nearest = (
                    str(error["category"]) == "FP"
                    and nearest_target is not None
                    and int(target_frame) == nearest_target
                )
                axis.axvline(
                    int(target_frame),
                    color="tab:blue",
                    linestyle="-." if is_nearest else ":",
                    linewidth=1.2,
                    alpha=0.85,
                    label=(
                        "Nearest target R peak"
                        if is_nearest
                        else ("Target R peak" if target_index == 0 else None)
                    ),
                )
            colors = {
                "TP": "tab:green",
                "PE": "tab:orange",
                "oPE": "tab:cyan",
                "FP": "tab:red",
                "FN": "tab:purple",
            }
            seen_labels: set[str] = set()
            for _, event in nearby.iterrows():
                category = str(event["category"])
                target_frame = event["target_frame"]
                predicted_frame = event["predicted_frame"]
                first_category = category not in seen_labels
                seen_labels.add(category)
                if pd.notna(target_frame):
                    axis.scatter(
                        [float(target_frame)],
                        [0.02],
                        color=colors[category],
                        marker="^",
                        s=55,
                        label=f"{category} target" if first_category else None,
                        zorder=4,
                    )
                if pd.notna(predicted_frame):
                    probability = float(event["probability"])
                    axis.scatter(
                        [float(predicted_frame)],
                        [probability],
                        color=colors[category],
                        marker="x" if category != "TP" else "o",
                        s=70,
                        label=f"{category} prediction" if first_category else None,
                        zorder=4,
                    )
                    emitted = event["emitted_on_frame"]
                    if pd.notna(emitted):
                        axis.annotate(
                            "",
                            xy=(float(emitted), probability),
                            xytext=(float(predicted_frame), probability),
                            arrowprops={"arrowstyle": "->", "color": colors[category]},
                        )
            nearest_note = ""
            if str(error["category"]) == "FP":
                if nearest_target is None:
                    nearest_note = " | no target in this sequence"
                elif not view_start <= nearest_target <= view_end:
                    nearest_note = (
                        f" | nearest target frame={nearest_target} "
                        f"(offset {nearest_offset:+d}, outside view)"
                    )
                else:
                    nearest_note = (
                        f" | nearest target frame={nearest_target} "
                        f"(offset {nearest_offset:+d})"
                    )
            axis.set(
                title=(
                    f"{error['category']} example | sid={sid} | center frame={center}\n"
                    f"peak classification, emission shown by arrow{nearest_note}"
                ),
                xlabel="Model frame",
                ylabel="Predicted probability",
                ylim=(-0.03, 1.03),
            )
            axis.title.set_fontsize(16)
            axis.xaxis.label.set_fontsize(14)
            axis.yaxis.label.set_fontsize(14)
            axis.tick_params(
                axis="both",
                which="major",
                labelsize=11,
                width=0.9,
                length=5,
            )
            axis.tick_params(axis="both", which="minor", width=0.7, length=3)
            for spine in axis.spines.values():
                spine.set_linewidth(0.9)
            _legend(axis, fontsize=9, loc="upper right", ncols=3)
            figure.subplots_adjust(left=0.10, right=0.98, bottom=0.16, top=0.76)
            pdf.savefig(figure, bbox_inches="tight")
            plt.close(figure)
            written += 1
    return written
