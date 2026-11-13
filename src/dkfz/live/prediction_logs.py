# Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Developed at the German Cancer Research Center (DKFZ).

"""Helpers for shaping saved live prediction logs."""

from __future__ import annotations

import numpy as np
import pandas as pd

from dkfz.live.triggers import TriggerEvent


def add_trigger_peak_flags(
    df400: pd.DataFrame,
    triggers: pd.DataFrame,
    *,
    pred_col: str = "pred",
) -> pd.DataFrame:
    """Return a copy of a 400 Hz recording with binary predicted-peak flags.

    ``triggers`` is the event dataframe written by ``offline_predict.py``.
    Match ``peak_sample_idx_400hz`` to ``raw_idx``, falling back to ``idx``
    or zero-based row positions within each recording, as the predictor does.
    Dataframe index labels are not sample positions. Without a sample-index
    column, pass the full original recording in its original row order.

    Match ``sid`` too when present in the recording. Inputs must refer to the
    same recording(s) and sample numbering. Unmatched events raise an error.
    Existing columns, row order, and index labels are preserved; an existing
    ``pred_col`` is replaced. Flags mark peak samples, not trigger emission.
    """
    peak_col = "peak_sample_idx_400hz"
    if peak_col not in triggers.columns:
        raise KeyError(f"Trigger dataframe is missing {peak_col!r}.")

    if "raw_idx" in df400.columns:
        sample_indices = df400["raw_idx"].to_numpy()
    elif "idx" in df400.columns:
        sample_indices = df400["idx"].to_numpy()
    elif "sid" in df400.columns:
        sample_indices = df400.groupby("sid", sort=False).cumcount().to_numpy()
    else:
        sample_indices = np.arange(len(df400))

    if pd.isna(sample_indices).any() or triggers[peak_col].isna().any():
        raise ValueError("Sample indices must not contain missing values.")
    if "sid" in df400.columns:
        if "sid" not in triggers.columns:
            raise KeyError("Trigger dataframe must include 'sid' for this recording.")
        if df400["sid"].isna().any() or triggers["sid"].isna().any():
            raise ValueError("Recording IDs must not contain missing values.")
        sample_keys = pd.MultiIndex.from_arrays([df400["sid"], sample_indices])
        peak_keys = pd.MultiIndex.from_arrays([triggers["sid"], triggers[peak_col]])
    else:
        if "sid" in triggers.columns and triggers["sid"].nunique(dropna=False) > 1:
            raise ValueError("Multiple trigger recording IDs require 'sid' in df400.")
        sample_keys = pd.Index(sample_indices)
        peak_keys = pd.Index(triggers[peak_col])

    if not sample_keys.is_unique:
        raise ValueError("Sample indices must be unique within each recording.")
    positions = sample_keys.get_indexer(peak_keys)
    if (positions < 0).any():
        raise ValueError(
            f"{int((positions < 0).sum())} trigger peaks do not match df400; "
            "use the corresponding recording with its original sample indices."
        )

    result = df400.copy()
    flags = np.zeros(len(result), dtype=np.uint8)
    flags[positions] = 1
    result[pred_col] = flags
    return result


def _reduce_raw_column(values, start: int, end: int):
    """Reduce one raw-sample column over a completed live frame."""
    block = values[int(start) : int(end) + 1]
    if len(block) == 0:
        return np.nan

    if pd.api.types.is_numeric_dtype(block):
        numeric_block = block.astype(float)
        if np.all(np.isnan(numeric_block)):
            return np.nan
        reduced = float(np.nanmax(numeric_block))
        return int(reduced) if reduced.is_integer() else reduced

    non_null = pd.Series(block).dropna()
    if non_null.empty:
        return None
    return non_null.iloc[-1]


def _raw_values_for_log_row(raw_df: pd.DataFrame, row, col: str):
    """Return raw column values covered by one live log row."""
    raw_start = int(row["raw_idx"])
    raw_end = int(row["_raw_frame_end_idx"])

    if "sid" in raw_df.columns:
        raw_group = raw_df[raw_df["sid"] == row["sid"]]
    else:
        raw_group = raw_df

    raw_idx_col = "raw_idx" if "raw_idx" in raw_group.columns else "idx"
    if raw_idx_col in raw_group.columns:
        mask = raw_group[raw_idx_col].between(raw_start, raw_end)
        return raw_group.loc[mask, col].to_numpy()

    values = raw_group[col].to_numpy()
    return values[raw_start : raw_end + 1]


def _frame_idx_column(df: pd.DataFrame) -> str:
    for col in ("frame_idx", "frame_idx_50hz", "idx"):
        if col in df.columns:
            return col
    raise KeyError("Prediction log is missing a frame index column.")


def add_prediction_result_columns(
    log_df: pd.DataFrame,
    events: list[TriggerEvent],
    raw_df: pd.DataFrame,
    reduced_raw_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Add pred and optional raw annotation columns to the probability log."""
    log_df = log_df.copy()

    if "pred" not in log_df.columns:
        log_df["pred"] = 0

    if not log_df.empty and events:
        if "raw_frame_end_idx" in log_df:
            peak_keys = {(event.sid, event.peak_sample_idx_400hz) for event in events}
            row_keys = list(zip(log_df["sid"], log_df["raw_frame_end_idx"]))
        else:
            peak_keys = {(event.sid, event.peak_frame_idx_50hz) for event in events}
            frame_idx_col = _frame_idx_column(log_df)
            row_keys = list(zip(log_df["sid"], log_df[frame_idx_col]))
        log_df.loc[[key in peak_keys for key in row_keys], "pred"] = 1

    cols_to_reduce: list[str] = ["target", "siemens_tr", "puls_peak"]  # , "mhd_peak"]
    for col in reduced_raw_cols or []:
        if col not in cols_to_reduce:
            cols_to_reduce.append(col)

    for col in cols_to_reduce:
        if col not in raw_df.columns or log_df.empty:
            continue

        reduced = []
        for _, row in log_df.iterrows():
            values = _raw_values_for_log_row(raw_df, row, col)
            reduced.append(_reduce_raw_column(values, 0, len(values) - 1))
        log_df[col] = reduced

    return log_df


def order_prediction_log_columns(
    log_df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Keep result columns up front and feature columns together at the end."""
    preferred = [
        "sid",
        "frame_idx",
        "raw_idx",
        "time_ms",
        "prob",
        "pred",
        "target",
        "siemens_tr",
        "puls_peak",
        # "mhd_peak",
        "trigger_emitted_now",
        "feature_dim",
    ]
    ordered = [col for col in preferred if col in log_df.columns]
    ordered += [col for col in feature_cols if col in log_df.columns]
    ordered += [
        col for col in log_df.columns if col not in ordered and not col.startswith("_")
    ]
    return log_df[ordered]


def build_prediction_log(
    log_df: pd.DataFrame,
    events: list[TriggerEvent],
    raw_df: pd.DataFrame,
    feature_cols: list[str],
    reduced_raw_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Build the saved live prediction log table."""
    log_df = add_prediction_result_columns(
        log_df=log_df,
        events=events,
        raw_df=raw_df,
        reduced_raw_cols=reduced_raw_cols,
    )
    return order_prediction_log_columns(log_df, feature_cols)
