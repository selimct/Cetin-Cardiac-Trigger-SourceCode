# Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Developed at the German Cancer Research Center (DKFZ).

"""Helpers for expanding saved model frames back to a raw-sample grid."""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd

from dkfz.core.data_processing import reorder_columns
from dkfz.core.frame_to_50hz import unframe_df_50hz_to_400hz

DEFAULT_TARGET_COLS = ("pred", "target", "siemens_tr", "puls_peak", "mhd_peak")
OMIT_FROM_UNFRAMED_OUTPUT = ("theta", "ecg1_rotated", "ecg2_rotated", "ecg3_rotated")

_FRAMED_ECG_RE = re.compile(r"^(ecg\d+)_t(\d+)$")


def framed_ecg_columns(ecg_cols: tuple[str, ...], factor: int) -> set[str]:
    """Return the expected framed ECG column names for the given channels."""
    return {f"{col}_t{k}" for col in ecg_cols for k in range(factor)}


def infer_framed_ecg_columns(
    df50: pd.DataFrame,
    *,
    factor: int | None = None,
) -> tuple[tuple[str, ...], int]:
    """Infer complete ``ecg*_t0..tN`` channel groups from a framed dataframe."""
    suffixes_by_channel: dict[str, set[int]] = {}
    for col in df50.columns:
        match = _FRAMED_ECG_RE.match(str(col))
        if match is None:
            continue
        suffixes_by_channel.setdefault(match.group(1), set()).add(int(match.group(2)))

    if not suffixes_by_channel:
        raise KeyError("No framed ECG columns matching 'ecgN_tK' were found.")

    if factor is None:
        max_suffix = max(max(suffixes) for suffixes in suffixes_by_channel.values())
        factor = max_suffix + 1

    if factor <= 0:
        raise ValueError("factor must be positive")

    expected_suffixes = set(range(factor))
    ecg_cols = tuple(
        sorted(
            (
                channel
                for channel, suffixes in suffixes_by_channel.items()
                if expected_suffixes.issubset(suffixes)
            ),
            key=lambda value: int(value.removeprefix("ecg")),
        )
    )
    if not ecg_cols:
        raise KeyError(
            f"No complete framed ECG channel has all suffixes 0..{factor - 1}."
        )
    return ecg_cols, factor


def _present_columns(
    df: pd.DataFrame,
    columns: tuple[str, ...] | list[str] | None,
    default: tuple[str, ...],
) -> tuple[str, ...]:
    if columns is None:
        columns = default
    return tuple(col for col in columns if col in df.columns)


def _expanded_time_ms(
    frame_times: pd.Series,
    *,
    factor: int,
    sample_period_ms: float,
) -> np.ndarray:
    offsets = (np.arange(factor, dtype=float) - (factor - 1)) * sample_period_ms
    return (frame_times.to_numpy(dtype=float)[:, None] + offsets[None, :]).reshape(
        len(frame_times) * factor
    )


def unframe_live_dataframe(
    df50: pd.DataFrame,
    *,
    factor: int | None = None,
    raw_idx_col: str = "raw_idx",
    block_index_col: str = "frame_idx",
    ecg_cols: tuple[str, ...] | None = None,
    target_cols: tuple[str, ...] | list[str] | None = None,
    passthrough_50hz_cols: tuple[str, ...] | list[str] | None = None,
    target_reduce: str = "max",
    raw_step: int = 1,
    raw_hz: float = 400.0,
    sample_period_ms: float | None = None,
    expand_time_ms: bool = True,
    rename_raw_signals: bool = True,
    drop_internal_columns: bool = True,
    order_columns: bool = True,
) -> pd.DataFrame:
    """Expand saved model rows to one row per raw sample.

    This is a script-friendly adapter around
    :func:`dkfz.core.frame_to_50hz.unframe_df_50hz_to_400hz`. It accepts both
    saved prepped training frames and ``prediction_all_events`` parquet contents.
    Framed ECG channels are exactly unstacked, frame-level values such as
    ``prob`` and ``puls`` are repeated across the raw samples in each frame, and
    ``time_ms`` is expanded from frame-end times to per-sample times by default.
    """
    if ecg_cols is None:
        ecg_cols, factor = infer_framed_ecg_columns(df50, factor=factor)
    elif factor is None:
        _, factor = infer_framed_ecg_columns(df50, factor=None)

    if factor is None:
        raise ValueError("factor could not be inferred.")
    if factor <= 0:
        raise ValueError("factor must be positive")

    ecg_cols = tuple(ecg_cols)
    required_ecg = framed_ecg_columns(ecg_cols, factor)
    missing_ecg = sorted(required_ecg.difference(df50.columns))
    if missing_ecg:
        raise KeyError(f"Missing framed ECG columns: {missing_ecg}")

    target_cols = _present_columns(df50, target_cols, DEFAULT_TARGET_COLS)

    if passthrough_50hz_cols is None:
        raw_like_cols = required_ecg | set(target_cols) | {raw_idx_col, block_index_col}
        passthrough_50hz_cols = tuple(
            col for col in df50.columns if col not in raw_like_cols
        )
    else:
        passthrough_50hz_cols = tuple(
            col for col in passthrough_50hz_cols if col in df50.columns
        )

    df400 = unframe_df_50hz_to_400hz(
        df50,
        factor=factor,
        raw_idx_col=raw_idx_col,
        ecg_cols=ecg_cols,
        passthrough_50hz_cols=tuple(passthrough_50hz_cols),
        target_cols=target_cols,
        target_reduce=target_reduce,
        raw_step=raw_step,
        keep_50hz_block_index=True,
        block_index_col=block_index_col,
    )

    if expand_time_ms and "time_ms" in df50.columns:
        if sample_period_ms is None:
            sample_period_ms = 1000.0 / float(raw_hz)
        df400["time_ms"] = _expanded_time_ms(
            df50["time_ms"],
            factor=factor,
            sample_period_ms=float(sample_period_ms),
        )

    if rename_raw_signals:
        rename_cols = {
            **{col: f"{col}_raw" for col in ecg_cols},
            "puls": "puls_raw",
        }
        rename_cols = {
            source: target
            for source, target in rename_cols.items()
            if source in df400.columns and target not in df400.columns
        }
        df400 = df400.rename(columns=rename_cols)

    if drop_internal_columns:
        df400 = df400.drop(
            columns=[c for c in OMIT_FROM_UNFRAMED_OUTPUT if c in df400.columns]
        )

    if order_columns:
        df400 = reorder_columns(df400)

    return df400


def read_unframe_live_parquet(
    input_path: str | Path,
    *,
    output_path: str | Path | None = None,
    parquet_compression: str = "snappy",
    **kwargs,
) -> pd.DataFrame:
    """Read a prepped/prediction parquet, return 400 Hz rows, optionally save it."""
    input_path = Path(input_path)
    df400 = unframe_live_dataframe(pd.read_parquet(input_path), **kwargs)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df400.to_parquet(output_path, compression=parquet_compression)

    return df400


unframe_live_parquet = read_unframe_live_parquet
