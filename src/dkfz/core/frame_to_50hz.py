# Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Developed at the German Cancer Research Center (DKFZ).

import pandas as pd
import numpy as np
from pathlib import Path


def _column_with_idx_fallback(df: pd.DataFrame, preferred: str) -> str:
    if preferred in df.columns:
        return preferred
    if preferred in {"raw_idx", "frame_idx"} and "idx" in df.columns:
        return "idx"
    return preferred


def trim_to_full_pulse_blocks(
    df: pd.DataFrame,
    puls_col: str = "puls_raw",
    factor: int = 8,
    fix_time: bool = True,
) -> pd.DataFrame:
    """Trim df to only contain full 50Hz blocks of puls entries.

    Args:
        df (pd.DataFrame): input df
        puls_col (str, optional): Defaults to "puls_raw".
        factor (int, optional): Factor of repeated puls entries. Defaults to 8.
        fix_time (bool, optional): Start time from 0. Defaults to True.

    Raises:
        Exception: df is not long enough to be trimmed.

    Returns:
        pd.DataFrame: trimmed df.
    """
    p = df[puls_col].to_numpy()
    n = len(p)
    if n < factor:
        return df.iloc[0:0].copy()

    # special case: no pulse data, e.g. column is all zeros
    if np.all(p == p[0]):
        Ltrim = (n // factor) * factor
        temp = df.iloc[:Ltrim].copy().reset_index(drop=True)
        temp["raw_idx"] = temp.index

        if fix_time:
            temp["time_ms"] = np.arange(Ltrim, dtype=np.float64) * 2.5

        return temp

    # helper: find first index i such that p[i:i+factor] are all equal
    def first_good_start(p):
        # print("first_good")
        for i in range(0, n - factor + 1):
            w = p[i : i + factor]
            if np.all(w == w[0]):
                return i
        return None

    # helper: find last index j such that p[j-factor+1:j+1] are all equal
    def last_good_end(p):
        # print("last_good")
        # length of the constant run at the very end
        v = p[-1]
        k = n - 1
        while k > 0 and p[k - 1] == v:
            k -= 1
        tail_len = n - k  # how many identical values at the end

        # if the tail is a partial block (1..factor-1), drop it by moving j left
        j = n - 1 if tail_len % factor == 0 else (k - 1)

        # now do exactly what you had: find last constant window of length `factor`
        for j in range(j, factor - 2, -1):
            w = p[j - factor + 1 : j + 1]
            if np.all(w == w[0]):
                return j
        return None

    i0 = first_good_start(p)
    i1 = last_good_end(p)
    if i0 is None or i1 is None or i1 < i0 + factor - 1:
        raise Exception("df does not contain trimmable data.")

    # additionally ensure trimmed length is a multiple of factor
    L = i1 - i0 + 1
    Ltrim = (L // factor) * factor
    temp = df.iloc[i0 : i0 + Ltrim].copy().reset_index(drop=True)
    temp["raw_idx"] = temp.index

    if fix_time:
        temp["time_ms"] = np.arange(Ltrim, dtype=np.float64) * float(2.5)

    return temp


def frame_df_400hz_to_50hz(
    df: pd.DataFrame,
    factor: int = 8,  # 400/50
    raw_idx_col: str = "raw_idx",
    ecg_cols: tuple[str, ...] = ("ecg1_raw", "ecg2_raw"),
    passthrough_50hz_cols: tuple[str, ...] = ("puls_raw",),
    target_cols: tuple[str, ...] = ("target",),
    target_reduce: str = "max",  # "center"|"last"|"mean"|"max"
    keep_block_index: bool = True,
) -> pd.DataFrame:
    """Frame a 400 Hz dataframe to 50 Hz by grouping every `factor` rows.

    Args:
        df (pd.DataFrame): Input df containing 400Hz data.
        factor (int, optional): Raw samples per 50 Hz frame. Defaults to 8.
        raw_idx_col (str, optional): Raw sample index col. Defaults to "raw_idx".
        ecg_cols (tuple[str, ...], optional): Raw ECG signal cols. Defaults to ("ecg1_raw", "ecg2_raw").
        passthrough_50hz_cols (tuple[str, ...], optional): raw feature cols that are already at 50Hz. Defaults to ("puls_raw", ).
        target_cols (tuple[str, ...], optional): target feature cols. Defaults to ("target", ).
        target_reduce (str, optional): Method of downsampling target feature. Defaults to "max".
        keep_block_index (bool, optional): Keep 400Hz sampling index. Defaults to True.

    Raises:
        ValueError:
        KeyError:
        ValueError:

    Returns:
        pd.DataFrame: New df on 50Hz grid.
    """

    df = trim_to_full_pulse_blocks(df, puls_col="puls_raw", factor=factor)

    if factor <= 0:
        raise ValueError("factor must be positive")

    # detect whether puls is just dummy / constant data
    has_real_puls = True
    if "puls_raw" in df.columns:
        p = df["puls_raw"].to_numpy()
        has_real_puls = not np.all(p == p[0])  # False if all entries are identical

    # if puls is dummy, do not pass it through
    if not has_real_puls:
        print("There is no raw pulse column in this DataFrame.")
        passthrough_50hz_cols = tuple(
            c for c in passthrough_50hz_cols if c != "puls_raw"
        )

    source_raw_idx_col = _column_with_idx_fallback(df, raw_idx_col)
    needed = (
        set(ecg_cols)
        | set(passthrough_50hz_cols)
        | set(target_cols)
        | {source_raw_idx_col}
    )

    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in df: {missing}")

    def _reduce_blocks(arr: np.ndarray, how: str) -> np.ndarray:
        # arr: (T50, factor, D)
        if how == "center":
            return arr[:, factor // 2, :]
        if how == "last":
            return arr[:, -1, :]
        if how == "mean":
            return arr.mean(axis=1)
        if how == "max":
            return arr.max(axis=1)
        raise ValueError(f"Unknown target_reduce: {how}")

    def _process_group(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values(source_raw_idx_col).reset_index(drop=True)

        T = len(g)
        T50 = T // factor
        if T50 < 1:
            return g.iloc[0:0].copy()  # empty with same columns

        Ttrim = T50 * factor
        g = g.iloc[:Ttrim].copy()

        # Block index at 50 Hz
        block_idx = np.arange(T50, dtype=np.int64)

        # Frame ECG -> (T50, factor*C)
        ecg_raw = g.loc[:, ecg_cols].to_numpy(dtype=np.float32)  # (Ttrim, C)
        C = ecg_raw.shape[1]
        ecg_blocks = ecg_raw.reshape(T50, factor, C)  # (T50, 8, C)
        ecg_framed = ecg_blocks.transpose(0, 2, 1).reshape(
            T50, factor * C
        )  # (T50, 8*C)

        # Name framed ECG columns
        framed_names = []
        for c in ecg_cols:
            for k in range(factor):
                framed_names.append(f"{c}_t{k}")  # sample k within the 20 ms block

        out = pd.DataFrame(ecg_framed, columns=framed_names)

        # Passthrough 50 Hz columns (repeated on raw grid) -> take last in block
        if passthrough_50hz_cols:
            low_raw = g.loc[:, passthrough_50hz_cols].to_numpy()  # (Ttrim, L)
            low_blocks = low_raw.reshape(T50, factor, -1)  # (T50, 8, L)
            low_50 = low_blocks[:, -1, :]  # (T50, L)
            out.loc[:, list(passthrough_50hz_cols)] = low_50

        # Targets reduced per block
        if target_cols:
            targ_raw = g.loc[:, target_cols].to_numpy()
            targ_blocks = targ_raw.reshape(T50, factor, -1)  # (T50, 8, K)
            targ_50 = _reduce_blocks(targ_blocks, target_reduce)  # (T50, K)
            for j, col in enumerate(target_cols):
                out[col] = targ_50[:, j]

        # Keep a representative 400 Hz index (e.g., last sample index of each block)
        if keep_block_index:
            raw_idxs = g[source_raw_idx_col].to_numpy()
            idx_blocks = raw_idxs.reshape(T50, factor)
            out["raw_idx"] = idx_blocks[:, 0]

        out["frame_idx"] = block_idx
        return out

    return _process_group(df).reset_index(drop=True)


def unframe_df_50hz_to_400hz(
    df50: pd.DataFrame,
    factor: int = 8,  # 400/50
    raw_idx_col: str = "raw_idx",  # start-of-block index from frame_df_400hz_to_50hz
    ecg_cols: tuple[str, ...] = ("ecg1_raw", "ecg2_raw"),
    passthrough_50hz_cols: tuple[str, ...] = ("puls_raw",),
    target_cols: tuple[str, ...] = ("target", "siemens_tr"),
    target_reduce: str = "max",  # must match what was used in framing (controls a consistent expansion)
    raw_step: int = 1,  # increment between consecutive 400 Hz samples
    keep_50hz_block_index: bool = False,  # keep df50["frame_idx"] repeated on 400 Hz grid
    block_index_col: str = "frame_idx",  # name of the 50 Hz block index column in df50
) -> pd.DataFrame:
    """
    Best-effort inverse of frame_df_400hz_to_50hz.

    Exactly inverts the ECG framing (ecg*_t{k} -> ecg* samples) and repeats the
    50 Hz passthrough features across each 400 Hz block.

    For targets: information is lost by downsampling, so this function reconstructs
    a 400 Hz target that is *consistent* with the chosen `target_reduce`:
      - "center": value placed at the center sample of each block, others filled with that value
      - "last":   value placed at the last sample of each block, others filled with that value
      - "mean"/"max": broadcast across the whole block (so re-framing reproduces df50)

    Assumptions:
      - df50 contains complete blocks only (as guaranteed by trim_to_full_pulse_blocks upstream).
      - If raw_idx_col exists, raw_idx is reconstructed as raw_idx + k*raw_step within each block.
        Otherwise a contiguous RangeIndex is used.
    """
    if factor <= 0:
        raise ValueError("factor must be positive")

    # ignore passthrough columns that are not present, e.g. dummy puls omitted in framing
    passthrough_50hz_cols = tuple(c for c in passthrough_50hz_cols if c in df50.columns)

    # --- Validate ECG framed columns ---
    framed_names = [f"{c}_t{k}" for c in ecg_cols for k in range(factor)]
    missing = [c for c in framed_names if c not in df50.columns]
    if missing:
        raise KeyError(f"Missing framed ECG columns in df50: {missing}")

    for c in passthrough_50hz_cols:
        if c not in df50.columns:
            raise KeyError(f"Missing passthrough col in df50: {c}")
    for c in target_cols:
        if c not in df50.columns:
            raise KeyError(f"Missing target col in df50: {c}")

    T50 = len(df50)
    if T50 == 0:
        cols = (
            list(ecg_cols)
            + list(passthrough_50hz_cols)
            + list(target_cols)
            + [raw_idx_col]
        )
        if keep_50hz_block_index:
            cols.append(block_index_col)
        return pd.DataFrame(columns=cols)

    # --- ECG: (T50, factor*C) -> (T50*factor, C) ---
    ecg_framed = df50.loc[:, framed_names].to_numpy(dtype=np.float32)  # (T50, factor*C)
    C = len(ecg_cols)
    # Reverse of: ecg_blocks.transpose(0,2,1).reshape(T50, factor*C)
    ecg_raw = (
        ecg_framed.reshape(T50, C, factor).transpose(0, 2, 1).reshape(T50 * factor, C)
    )

    out = pd.DataFrame(ecg_raw, columns=list(ecg_cols))

    # --- Passthrough 50 Hz features: repeat within each block ---
    if passthrough_50hz_cols:
        low_50 = df50.loc[:, passthrough_50hz_cols].to_numpy()
        low_raw = np.repeat(low_50, factor, axis=0)
        out.loc[:, list(passthrough_50hz_cols)] = low_raw

    # --- Targets: reconstruct a block-wise consistent 400 Hz version ---
    if target_cols:
        targ_50 = df50.loc[:, target_cols].to_numpy()
        targ_raw = np.repeat(targ_50, factor, axis=0)

        # Optionally “place” the value at a representative sample; we still fill the whole
        # block with the same value to keep targets well-defined on 400 Hz.
        # (Broadcasting keeps re-framing with mean/max/last/center consistent.)
        if target_reduce not in {"center", "last", "mean", "max"}:
            raise ValueError(f"Unknown target_reduce: {target_reduce}")

        # If you prefer strictly placing only one sample and leaving others NaN, replace the
        # broadcast above by a NaN array and set only the chosen positions.
        out.loc[:, list(target_cols)] = targ_raw

    # --- Reconstruct 400 Hz sort/index column ---
    source_frame_idx_col = block_index_col
    if (
        source_frame_idx_col not in df50.columns
        and source_frame_idx_col == "frame_idx"
        and "idx" in df50.columns
    ):
        source_frame_idx_col = "idx"

    if raw_idx_col in df50.columns:
        base = df50[raw_idx_col].to_numpy()
        k = (np.arange(factor, dtype=np.int64) * raw_step)[None, :]  # (1, factor)
        raw_idxs = (base[:, None] + k).reshape(T50 * factor)
    else:
        raw_idxs = np.arange(T50 * factor, dtype=np.int64)

    out[raw_idx_col] = raw_idxs

    # --- Optionally keep the 50 Hz block index repeated on 400 Hz grid ---
    if keep_50hz_block_index:
        if source_frame_idx_col not in df50.columns:
            raise KeyError(
                f"keep_50hz_block_index=True but df50 lacks '{block_index_col}'"
            )
        out[block_index_col] = np.repeat(
            df50[source_frame_idx_col].to_numpy(),
            factor,
        )

    return out.reset_index(drop=True)


def process_parquets(in_dir, out_dir, fn=frame_df_400hz_to_50hz):
    in_dir, out_dir = Path(in_dir), Path(out_dir)
    for p in in_dir.rglob("*.parquet"):
        rel = p.relative_to(in_dir)  # keep subdir structure
        out_path = out_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)

        df = pd.read_parquet(p)
        df2 = fn(df)  # your transform
        df2.to_parquet(out_path, index=False, compression="snappy")
