# Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Validation-only calibration of a previous-frame recovery output."""

from dataclasses import replace
import math

import numpy as np

from dkfz.live.event_matching import compute_event_position_metrics


def recovery_fp_budget(
    baseline_fp, relative_allowance, extra_fp_per_minute, validation_minutes
):
    """Pool the duration allowance over real validation frames, excluding padding."""
    if not math.isfinite(relative_allowance) or relative_allowance < 0:
        raise ValueError("recovery_max_fp_increase must be finite and nonnegative")
    if not math.isfinite(extra_fp_per_minute) or extra_fp_per_minute < 0:
        raise ValueError(
            "recovery_min_extra_fp_per_minute must be finite and nonnegative"
        )
    if validation_minutes is not None and (
        not math.isfinite(validation_minutes) or validation_minutes < 0
    ):
        raise ValueError("validation_minutes must be finite and nonnegative")
    if extra_fp_per_minute > 0 and validation_minutes is None:
        raise ValueError(
            "The duration-based recovery allowance requires validation_minutes"
        )
    relative_extra = math.floor(baseline_fp * (1 + relative_allowance)) - baseline_fp
    duration_extra = math.floor(extra_fp_per_minute * (validation_minutes or 0))
    return baseline_fp + max(relative_extra, duration_extra)


def recovery_peaks(
    probabilities,
    *,
    threshold,
    recovery_threshold,
    holdoff_frames=10,
    emission_threshold=None,
):
    from dkfz.native import _live_core

    values = np.asarray(probabilities, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("recovery probabilities must have shape (frames, 2)")
    return _live_core.streaming_trigger_peaks_with_recovery(
        values[:, 0],
        values[:, 1],
        float(threshold),
        None if recovery_threshold is None else float(recovery_threshold),
        int(holdoff_frames),
        None if emission_threshold is None else float(emission_threshold),
    )


def calibrate_recovery(
    probabilities,
    targets,
    selection,
    *,
    holdoff_frames=10,
    error_association_frames=1,
    outer_error_association_frames=2,
    exact_tolerance_frames=None,
    max_fp_increase=0.10,
    min_extra_fp_per_minute=0.0,
    validation_minutes=None,
    thresholds=None,
):
    """Keep the primary threshold fixed; add recovery only if validation improves.

    FP/FN include outer position errors, matching the default overview. Disabled
    recovery is always a candidate, including when no positive threshold works.
    Callers must reset/mask the recovery output at sequence boundaries.
    """
    from dkfz.core.network_helpers import streaming_trigger_peaks

    values = np.asarray(probabilities, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("recovery probabilities must have shape (frames, 2)")
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("probabilities must be finite and in [0, 1]")
    if not math.isfinite(max_fp_increase) or max_fp_increase < 0:
        raise ValueError("recovery_max_fp_increase must be finite and nonnegative")
    if exact_tolerance_frames is None:
        exact_tolerance_frames = int(selection.metrics.get("exact_tolerance_frames", 0))
    options = dict(
        exact_tolerance_frames=exact_tolerance_frames,
        error_association_frames=error_association_frames,
        outer_error_association_frames=outer_error_association_frames,
    )
    baseline = compute_event_position_metrics(
        streaming_trigger_peaks(values[:, 0], selection.threshold, holdoff_frames),
        targets,
        **options,
    )

    def fp(m):
        return m["unmatched_false_positive"] + m["outer_position_error"]

    def fn(m):
        return m["unmatched_false_negative"] + m["outer_position_error"]

    def combined_errors(measured):
        return fp(measured) + fn(measured)

    budget = recovery_fp_budget(
        fp(baseline), max_fp_increase, min_extra_fp_per_minute, validation_minutes
    )
    # The native service can end a primary excursion at the higher FP threshold.
    # Its different emission time changes holdoff and opportunities for recovery.
    # Require the chosen policy to respect the budget in that mode as well.
    emission_reference = max(
        float(selection.threshold),
        (
            float(selection.fp_threshold)
            if selection.fp_threshold is not None
            else float(selection.threshold)
        ),
    )
    guard_baseline = None
    guard_budget = None

    def early_metrics(recovery_threshold):
        return compute_event_position_metrics(
            recovery_peaks(
                values,
                threshold=selection.threshold,
                recovery_threshold=recovery_threshold,
                holdoff_frames=holdoff_frames,
                emission_threshold=emission_reference,
            ),
            targets,
            **options,
        )

    if emission_reference > selection.threshold:
        guard_baseline = early_metrics(None)
        guard_budget = recovery_fp_budget(
            fp(guard_baseline),
            max_fp_increase,
            min_extra_fp_per_minute,
            validation_minutes,
        )

    def audit_row(threshold, measured, guarded, *, selected=False):
        return dict(
            recovery_threshold=threshold,
            selected=selected,
            fp_budget=budget,
            baseline_fp=fp(baseline),
            baseline_fn=fn(baseline),
            validation_minutes=validation_minutes,
            min_extra_fp_per_minute=min_extra_fp_per_minute,
            combined_errors=combined_errors(measured),
            combined_errors_improved=combined_errors(measured)
            < combined_errors(baseline),
            **measured,
            **(
                {
                    "early_emission_threshold": emission_reference,
                    "early_emission_fp_budget": guard_budget,
                    "early_emission_fp": fp(guarded),
                    "early_emission_fn": fn(guarded),
                }
                if guarded is not None
                else {}
            ),
        )

    best, chosen = baseline, None
    best_guard = guard_baseline
    rows = [audit_row(None, baseline, guard_baseline, selected=True)]
    grid = (
        np.linspace(0.001, 1.0, 1000) if thresholds is None else np.asarray(thresholds)
    )
    if (
        grid.ndim != 1
        or not np.isfinite(grid).all()
        or np.any((grid <= 0) | (grid > 1))
    ):
        raise ValueError("recovery thresholds must be finite and in (0, 1]")
    for threshold in grid:
        peaks = recovery_peaks(
            values,
            threshold=selection.threshold,
            recovery_threshold=threshold,
            holdoff_frames=holdoff_frames,
        )
        measured = compute_event_position_metrics(peaks, targets, **options)
        guarded = early_metrics(threshold) if guard_baseline is not None else None
        rows.append(audit_row(float(threshold), measured, guarded))
        guard_ok = guarded is None or (
            fp(guarded) <= guard_budget
            and fn(guarded) <= fn(guard_baseline)
            and combined_errors(guarded) <= combined_errors(guard_baseline)
        )
        if (
            guard_ok
            and fp(measured) <= budget
            and fn(measured) < fn(baseline)
            and combined_errors(measured) < combined_errors(baseline)
        ):
            key = (
                fn(measured),
                fn(guarded) if guarded is not None else 0,
                fp(measured),
                fp(guarded) if guarded is not None else 0,
                measured["position_error"],
                -float(threshold),
            )
            previous = (
                fn(best),
                fn(best_guard) if best_guard is not None else 0,
                fp(best),
                fp(best_guard) if best_guard is not None else 0,
                best["position_error"],
                -(chosen or 1.0),
            )
            if key < previous:
                best, chosen = measured, float(threshold)
                best_guard = guarded
    for row in rows:
        row["selected"] = row["recovery_threshold"] == chosen
    validation_metrics = dict(best)
    validation_metrics.update(
        fp_budget=budget,
        baseline_fp=fp(baseline),
        baseline_fn=fn(baseline),
        validation_minutes=validation_minutes,
        min_extra_fp_per_minute=min_extra_fp_per_minute,
        combined_errors=combined_errors(best),
    )
    if best_guard is not None:
        validation_metrics["early_emission"] = dict(
            emission_threshold=emission_reference,
            fp_budget=guard_budget,
            baseline_fp=fp(guard_baseline),
            baseline_fn=fn(guard_baseline),
            **best_guard,
        )
    return (
        replace(
            selection,
            recovery_threshold=chosen,
            recovery_max_fp_increase=float(max_fp_increase),
            recovery_min_extra_fp_per_minute=float(min_extra_fp_per_minute),
            recovery_validation_minutes=validation_minutes,
            recovery_validation_metrics=validation_metrics,
        ),
        rows,
    )
