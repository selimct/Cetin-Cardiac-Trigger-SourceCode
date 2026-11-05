# Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Developed at the German Cancer Research Center (DKFZ).

"""Training-data preparation helpers for live ECG models."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path

import numpy as np
import pandas as pd

from dkfz.live.feature_framing import (
    PREPROCESSING_CONTRACT_VERSION,
    make_live_feature_dataframe,
    make_live_framer,
)
from dkfz.live.io import (
    optional_raw_signal_column,
    repo_relative_path,
    require_raw_signal_column,
)

SPLIT_COL = "split"
TRAIN_SPLIT = "train"
VAL_SPLIT = "val"
LONG_SEQUENCE_SPLIT_SECONDS = 12 * 60
LONG_SEQUENCE_SPLIT_PARTS = 3
PREPROCESSING_CONTRACT_COL = "_preprocessing_contract_version"


def _display_path(path: Path) -> str:
    """Return a current-directory-relative path for user-facing output."""
    return repo_relative_path(path)


def _assign_sequence_ids(
    df: pd.DataFrame,
    start_sid: int,
    idx_col: str = "raw_idx",
    sid_col: str = "sid",
) -> tuple[pd.DataFrame, int]:
    """Assign integer sequence ids from raw index restarts."""
    df = df.copy()
    if df.empty:
        df[sid_col] = pd.Series(dtype="int64")
        return df, start_sid

    source_idx_col = idx_col
    if source_idx_col not in df.columns and "idx" in df.columns:
        source_idx_col = "idx"
    if source_idx_col not in df.columns:
        df[sid_col] = start_sid
        return df, start_sid + 1

    is_restart = df[source_idx_col].eq(0) & df[source_idx_col].shift().notna()
    sequence_offsets = is_restart.cumsum().astype("int64")
    df[sid_col] = start_sid + sequence_offsets
    return df, int(df[sid_col].iloc[-1]) + 1


def _reset_raw_segment(
    df: pd.DataFrame,
    *,
    sid: int,
    split: str,
    time_col: str | None,
    sid_col: str = "sid",
) -> pd.DataFrame:
    """Make a raw segment look like a fresh online stream."""
    segment = df.copy().reset_index(drop=True)
    segment[sid_col] = int(sid)
    segment[SPLIT_COL] = split

    for idx_col in ("raw_idx", "idx"):
        if idx_col in segment.columns:
            segment[idx_col] = np.arange(len(segment), dtype=np.int64)

    if time_col is not None and time_col in segment.columns and not segment.empty:
        values = pd.to_numeric(segment[time_col], errors="coerce")
        first_valid = values.dropna()
        if not first_valid.empty:
            segment[time_col] = values - float(first_valid.iloc[0])

    return segment


def _split_long_raw_sequences(
    df: pd.DataFrame,
    *,
    raw_hz: float,
    max_duration_seconds: float = LONG_SEQUENCE_SPLIT_SECONDS,
    parts: int = LONG_SEQUENCE_SPLIT_PARTS,
    sid_col: str = "sid",
) -> list[pd.DataFrame]:
    """Split raw sequences longer than max_duration_seconds into equal parts."""
    max_samples = int(float(raw_hz) * float(max_duration_seconds))
    if max_samples <= 0:
        raise ValueError("raw_hz and max_duration_seconds must be positive.")
    if parts < 1:
        raise ValueError("parts must be at least 1.")

    out: list[pd.DataFrame] = []
    groups = df.groupby(sid_col, sort=False) if sid_col in df.columns else [(0, df)]
    for _, group in groups:
        group = group.reset_index(drop=True)
        split_parts = int(parts) if len(group) > max_samples else 1
        for part_idx in range(split_parts):
            start = part_idx * len(group) // split_parts
            stop = (part_idx + 1) * len(group) // split_parts
            out.append(group.iloc[start:stop].copy())
    return out


def _split_raw_sequence_train_val(
    df: pd.DataFrame,
    *,
    validation_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return contiguous train/validation raw segments from one base sequence."""
    if not 0.0 < float(validation_fraction) < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1.")
    if len(df) < 2:
        raise ValueError("Cannot split a sequence with fewer than two rows.")

    cut = int(len(df) * (1.0 - float(validation_fraction)))
    cut = min(max(cut, 1), len(df) - 1)
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def _shift_frame_sid_namespace(
    frame_dfs: list[pd.DataFrame],
    sid_offset: int,
    sid_col: str = "sid",
) -> tuple[list[pd.DataFrame], int]:
    """Shift framed segment ids so independently framed files do not collide."""
    if sid_offset == 0:
        max_sid = max(
            (
                int(frame_df[sid_col].max())
                for frame_df in frame_dfs
                if not frame_df.empty
            ),
            default=-1,
        )
        return frame_dfs, max_sid + 1

    shifted: list[pd.DataFrame] = []
    max_sid = -1
    for frame_df in frame_dfs:
        if frame_df.empty:
            shifted.append(frame_df)
            continue
        copy = frame_df.copy()
        copy[sid_col] = copy[sid_col].astype("int64") + sid_offset
        max_sid = max(max_sid, int(copy[sid_col].max()))
        shifted.append(copy)
    return shifted, max_sid + 1


def _build_training_frame_dfs_for_path(
    path: Path,
    *,
    ecg1_col: str,
    ecg2_col: str,
    puls_col: str | None,
    target_col: str,
    not_use_puls: bool,
    time_col: str | None,
    time_unit: str,
    target_reduce: str,
    raw_hz: float,
    frame_size_raw: int,
    ecg_quantile: float,
    ecg_norm_window_samples: int,
    ecg1_offset: float,
    ecg2_offset: float,
    ecg_rotation_angle: float,
    puls_offset: float,
    puls_quantile_low: float,
    puls_quantile_high: float,
    puls_norm_window_frames: int,
    puls_norm_warmup_value: float,
    puls_clip: bool,
    align_to_puls_blocks: bool,
    verify_puls_constant_within_block: bool,
    validation_fraction: float | None,
    long_sequence_split_seconds: float,
    long_sequence_split_parts: int,
    ecg3_col: str | None,
    ecg3_offset: float,
    ecg_slope_mode: str,
    ecg_slope_slow_window_samples: int,
    signal_quality_mode: str,
    signal_quality_stats: dict | None,
    split: str | None,
) -> tuple[list[pd.DataFrame], str]:
    """Build framed rows for one raw parquet file."""
    out: list[pd.DataFrame] = []
    include_puls = not not_use_puls

    df = pd.read_parquet(path)
    df, _ = _assign_sequence_ids(df, 0)

    actual_ecg1_col = require_raw_signal_column(df, ecg1_col, "ECG1")
    actual_ecg2_col = require_raw_signal_column(df, ecg2_col, "ECG2")
    actual_ecg3_col = None
    if ecg3_col is not None:
        actual_ecg3_col = require_raw_signal_column(df, ecg3_col, "ECG3")
    actual_puls_col = None
    if include_puls:
        actual_puls_col = require_raw_signal_column(
            df,
            puls_col or "puls_raw",
            "pulse",
        )
    elif str(signal_quality_mode) == "ecg_puls":
        actual_puls_col = optional_raw_signal_column(
            df,
            puls_col or "puls_raw",
            "pulse",
        )

    if target_col not in df.columns:
        raise KeyError(f"{path}: missing target column {target_col!r}")

    framer = make_live_framer(
        raw_hz=raw_hz,
        frame_size_raw=frame_size_raw,
        include_puls=include_puls,
        ecg_quantile=ecg_quantile,
        ecg_norm_window_samples=ecg_norm_window_samples,
        ecg1_offset=ecg1_offset,
        ecg2_offset=ecg2_offset,
        ecg3_offset=ecg3_offset,
        include_ecg3=actual_ecg3_col is not None,
        ecg_slope_mode=ecg_slope_mode,
        ecg_slope_slow_window_samples=ecg_slope_slow_window_samples,
        signal_quality_mode=signal_quality_mode,
        signal_quality_stats=signal_quality_stats,
        ecg_rotation_angle=ecg_rotation_angle,
        puls_offset=puls_offset,
        puls_quantile_low=puls_quantile_low,
        puls_quantile_high=puls_quantile_high,
        puls_norm_window_frames=puls_norm_window_frames,
        puls_norm_warmup_value=puls_norm_warmup_value,
        puls_clip=puls_clip,
        align_to_puls_blocks=align_to_puls_blocks,
        verify_puls_constant_within_block=verify_puls_constant_within_block,
    )

    if split not in {None, TRAIN_SPLIT, VAL_SPLIT}:
        raise ValueError(
            f"split must be {TRAIN_SPLIT!r}, {VAL_SPLIT!r}, or None, got {split!r}."
        )
    if split is None and validation_fraction is None:
        raise ValueError("validation_fraction is required when split is None.")

    source_frames = 0
    next_segment_sid = 0
    base_sequences = _split_long_raw_sequences(
        df,
        raw_hz=raw_hz,
        max_duration_seconds=long_sequence_split_seconds,
        parts=long_sequence_split_parts,
    )
    for base_sequence in base_sequences:
        if split is None:
            train_raw, val_raw = _split_raw_sequence_train_val(
                base_sequence,
                validation_fraction=float(validation_fraction),
            )
            split_segments = (
                (TRAIN_SPLIT, train_raw),
                (VAL_SPLIT, val_raw),
            )
        else:
            split_segments = ((split, base_sequence),)

        for split_name, raw_segment in split_segments:
            segment = _reset_raw_segment(
                raw_segment,
                sid=next_segment_sid,
                split=split_name,
                time_col=time_col,
            )
            next_segment_sid += 1
            frame_df = make_live_feature_dataframe(
                df=segment,
                ecg1_col=actual_ecg1_col,
                ecg2_col=actual_ecg2_col,
                ecg3_col=actual_ecg3_col,
                puls_col=actual_puls_col,
                target_col=target_col,
                time_col=time_col,
                time_unit=time_unit,
                target_reduce=target_reduce,
                framer=framer,
            )
            frame_df["source_file"] = repo_relative_path(path)
            frame_df[SPLIT_COL] = split_name
            out.append(frame_df)
            source_frames += len(frame_df)

    message = (
        f"{_display_path(path)}: "
        f"{len(df):_} raw rows -> {source_frames:_} live frames"
    )
    return out, message


def _build_training_frame_dfs_for_path_from_kwargs(
    args: tuple[Path, dict],
) -> tuple[list[pd.DataFrame], str]:
    """ProcessPoolExecutor entrypoint for one raw parquet path."""
    path, kwargs = args
    return _build_training_frame_dfs_for_path(path, **kwargs)


def build_training_frame_dfs(
    input_paths: list[Path],
    ecg1_col: str,
    ecg2_col: str,
    puls_col: str | None,
    target_col: str,
    not_use_puls: bool,
    time_col: str | None,
    time_unit: str,
    target_reduce: str,
    raw_hz: float,
    frame_size_raw: int,
    ecg_quantile: float,
    ecg_norm_window_samples: int,
    ecg1_offset: float,
    ecg2_offset: float,
    ecg_rotation_angle: float,
    puls_offset: float,
    puls_quantile_low: float,
    puls_quantile_high: float,
    puls_norm_window_frames: int,
    puls_norm_warmup_value: float,
    puls_clip: bool,
    align_to_puls_blocks: bool,
    verify_puls_constant_within_block: bool,
    validation_fraction: float | None,
    long_sequence_split_seconds: float = LONG_SEQUENCE_SPLIT_SECONDS,
    long_sequence_split_parts: int = LONG_SEQUENCE_SPLIT_PARTS,
    ecg3_col: str | None = None,
    ecg3_offset: float = 0.0,
    ecg_slope_mode: str = "none",
    ecg_slope_slow_window_samples: int = 40,
    signal_quality_mode: str = "none",
    signal_quality_stats: dict | None = None,
    prep_workers: int = 1,
    split: str | None = None,
) -> list[pd.DataFrame]:
    """Build framed rows with either an assigned or within-file split.

    When ``split`` is ``"train"`` or ``"val"``, every row from every input is
    assigned to that split. When it is ``None``, the legacy contiguous
    train/validation split controlled by ``validation_fraction`` is retained.
    """
    out: list[pd.DataFrame] = []
    worker_kwargs = {
        "ecg1_col": ecg1_col,
        "ecg2_col": ecg2_col,
        "puls_col": puls_col,
        "target_col": target_col,
        "not_use_puls": not_use_puls,
        "time_col": time_col,
        "time_unit": time_unit,
        "target_reduce": target_reduce,
        "raw_hz": raw_hz,
        "frame_size_raw": frame_size_raw,
        "ecg_quantile": ecg_quantile,
        "ecg_norm_window_samples": ecg_norm_window_samples,
        "ecg1_offset": ecg1_offset,
        "ecg2_offset": ecg2_offset,
        "ecg_rotation_angle": ecg_rotation_angle,
        "puls_offset": puls_offset,
        "puls_quantile_low": puls_quantile_low,
        "puls_quantile_high": puls_quantile_high,
        "puls_norm_window_frames": puls_norm_window_frames,
        "puls_norm_warmup_value": puls_norm_warmup_value,
        "puls_clip": puls_clip,
        "align_to_puls_blocks": align_to_puls_blocks,
        "verify_puls_constant_within_block": verify_puls_constant_within_block,
        "validation_fraction": validation_fraction,
        "long_sequence_split_seconds": long_sequence_split_seconds,
        "long_sequence_split_parts": long_sequence_split_parts,
        "ecg3_col": ecg3_col,
        "ecg3_offset": ecg3_offset,
        "ecg_slope_mode": ecg_slope_mode,
        "ecg_slope_slow_window_samples": ecg_slope_slow_window_samples,
        "signal_quality_mode": signal_quality_mode,
        "signal_quality_stats": signal_quality_stats,
        "split": split,
    }

    max_workers = int(prep_workers)
    if max_workers < 1:
        max_workers = os.cpu_count() or 1
    max_workers = max(1, min(max_workers, len(input_paths)))

    if max_workers == 1:
        results = [
            _build_training_frame_dfs_for_path(path, **worker_kwargs)
            for path in input_paths
        ]
    else:
        print(f"Preparing training frames with {max_workers} worker processes")
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(
                executor.map(
                    _build_training_frame_dfs_for_path_from_kwargs,
                    ((path, worker_kwargs) for path in input_paths),
                )
            )

    next_sid = 0
    for frame_dfs, message in results:
        shifted, next_sid = _shift_frame_sid_namespace(frame_dfs, next_sid)
        out.extend(shifted)
        print(message)

    return out


def default_prepped_file(modelname: str) -> str:
    """Return the default preprocessed feature cache path for a model name."""
    return f"models/{modelname}/prepped.parquet"


def save_prepped_training_frame_dfs(
    frame_dfs: list[pd.DataFrame],
    out_file: str,
) -> None:
    """Save preprocessed training frame DataFrames as one parquet file."""
    if not frame_dfs:
        raise ValueError("No preprocessed DataFrames to save.")

    out_path = Path(out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output = pd.concat(frame_dfs, ignore_index=True)
    output[PREPROCESSING_CONTRACT_COL] = PREPROCESSING_CONTRACT_VERSION
    output.to_parquet(out_path, compression="snappy")
    print(f"Wrote preprocessed training frames to {repo_relative_path(out_path)}")


def load_prepped_training_frame_dfs(in_file: str) -> list[pd.DataFrame]:
    """Load a regenerated preprocessed training-frame parquet."""
    in_path = Path(in_file)
    if not in_path.exists():
        raise FileNotFoundError(
            f"Preprocessed parquet not found: {repo_relative_path(in_path)}"
        )

    df = pd.read_parquet(in_path)
    if PREPROCESSING_CONTRACT_COL not in df.columns:
        raise ValueError(
            f"{repo_relative_path(in_path)} has no preprocessing contract marker. "
            "Regenerate it from raw parquet input data."
        )
    contract_values = pd.to_numeric(
        df[PREPROCESSING_CONTRACT_COL],
        errors="coerce",
    ).dropna().unique()
    if (
        len(contract_values) != 1
        or int(contract_values[0]) != PREPROCESSING_CONTRACT_VERSION
    ):
        found = contract_values.tolist()
        raise ValueError(
            f"{repo_relative_path(in_path)} uses preprocessing contract {found}; "
            f"expected {PREPROCESSING_CONTRACT_VERSION}. Regenerate it from raw "
            "parquet input data."
        )
    df = df.drop(columns=[PREPROCESSING_CONTRACT_COL])
    required = {"source_file", SPLIT_COL, "sid", "frame_idx"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(
            f"{repo_relative_path(in_path)} is missing regenerated prepped columns "
            f"{missing}. "
            "Regenerate it from raw parquet input data."
        )

    print(
        f"Loaded {len(df):_} preprocessed training frames from "
        f"{repo_relative_path(in_path)}"
    )
    return [df.reset_index(drop=True)]


def separate_framed_train_val(
    frame_dfs: list[pd.DataFrame] | pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separate already-framed rows by their raw-level train/val assignment."""
    if not isinstance(frame_dfs, list):
        frame_dfs = [frame_dfs]

    df = (
        pd.concat(frame_dfs, ignore_index=True)
        if len(frame_dfs) > 1
        else frame_dfs[0].copy()
    )
    if SPLIT_COL not in df.columns:
        raise ValueError(
            f"Framed training data is missing {SPLIT_COL!r}. "
            "Regenerate preprocessed data from raw parquet inputs."
        )

    train_df = df[df[SPLIT_COL] == TRAIN_SPLIT].copy().reset_index(drop=True)
    val_df = df[df[SPLIT_COL] == VAL_SPLIT].copy().reset_index(drop=True)
    if train_df.empty or val_df.empty:
        counts = df[SPLIT_COL].value_counts(dropna=False).to_dict()
        raise ValueError(
            f"Expected non-empty {TRAIN_SPLIT!r} and {VAL_SPLIT!r} splits, "
            f"got {counts}."
        )

    return train_df, val_df
