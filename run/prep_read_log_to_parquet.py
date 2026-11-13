#!/usr/bin/env python3
# Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Developed at the German Cancer Research Center (DKFZ).

"""Convert ECG and pulse PMU logs to synchronized sample-level parquet."""

from __future__ import annotations

import argparse
import glob
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dkfz.core.casting_helpers import (
    cast_columns_to_int16,
    cast_columns_to_uint8,
    subtract_offset_float64,
)
from dkfz.core.data_processing import reorder_columns
from dkfz.core.frame_to_50hz import trim_to_full_pulse_blocks
from dkfz.core.read_logs import (
    LoggedArrayMS,
    extract_ecg_and_trigger,
    prune_to_common_samples3,
    prune_to_common_samples4,
    read_log_to_array,
    read_mdh_times,
)
from dkfz.live.triggers import clean_siemens_trigger_pairs


LOG_FILE_SUFFIXES = {".ecg", ".puls"}
SAMPLE_INTERVAL_MS = 2.5
ECG_LOW_OFFSET = 2048
ECG_HIGH_OFFSET = 10240
ECG_OFFSET_MODES = ("auto", "standard", "swapped")


def strip_log_suffix(input_path: str | Path) -> str:
    path = Path(input_path)
    if path.suffix in LOG_FILE_SUFFIXES:
        path = path.with_suffix("")
    elif path.name.endswith("."):
        path = path.with_name(path.name[:-1])
    return str(path)


def expand_input_paths(inputs: list[str]) -> list[str]:
    in_files: list[str] = []
    seen: set[str] = set()

    def add_input(input_path: str | Path) -> None:
        in_file = strip_log_suffix(input_path)
        if in_file not in seen:
            seen.add(in_file)
            in_files.append(in_file)

    for input_arg in inputs:
        if glob.has_magic(input_arg):
            matches = sorted(glob.glob(input_arg, recursive=True))
            if not matches:
                raise FileNotFoundError(f"No files matched glob pattern: {input_arg}")
        else:
            matches = [input_arg]

        for match in matches:
            path = Path(match)
            if path.is_dir():
                ecg_files = sorted(path.glob("*.ecg"))
                if not ecg_files:
                    raise FileNotFoundError(f"No .ecg files found in directory: {path}")
                for ecg_file in ecg_files:
                    add_input(ecg_file)
            else:
                add_input(path)

    return in_files


def resolve_ecg_offsets(
    ecg1_raw: np.ndarray,
    ecg2_raw: np.ndarray,
    mode: str = "auto",
) -> tuple[int, int]:
    if mode == "standard":
        return ECG_LOW_OFFSET, ECG_HIGH_OFFSET
    if mode == "swapped":
        return ECG_HIGH_OFFSET, ECG_LOW_OFFSET
    if mode != "auto":
        raise ValueError(f"Unknown ECG offset mode: {mode}")
    if len(ecg1_raw) == 0 or len(ecg2_raw) == 0:
        raise ValueError("Cannot infer ECG offsets from empty ECG channels.")

    ecg1_median = float(np.median(ecg1_raw))
    ecg2_median = float(np.median(ecg2_raw))
    if not np.isfinite(ecg1_median) or not np.isfinite(ecg2_median):
        raise ValueError(
            "Cannot infer ECG offsets from non-finite ECG channel medians: "
            f"ecg1={ecg1_median}, ecg2={ecg2_median}"
        )
    standard_score = abs(ecg1_median - ECG_LOW_OFFSET) + abs(
        ecg2_median - ECG_HIGH_OFFSET
    )
    swapped_score = abs(ecg1_median - ECG_HIGH_OFFSET) + abs(
        ecg2_median - ECG_LOW_OFFSET
    )

    if standard_score <= swapped_score:
        ecg1_offset, ecg2_offset = ECG_LOW_OFFSET, ECG_HIGH_OFFSET
    else:
        ecg1_offset, ecg2_offset = ECG_HIGH_OFFSET, ECG_LOW_OFFSET

    print(
        "ECG offset mode auto: "
        f"ecg1 median={ecg1_median:.1f}, ecg2 median={ecg2_median:.1f}, "
        f"subtract ecg1={ecg1_offset}, ecg2={ecg2_offset}"
    )
    return ecg1_offset, ecg2_offset


def read_log_to_parquet(
    inFile: str = "data/raw/file_name",
    extract_trigger: bool = True,
    snap_to_grid: bool = True,
    ecg_offset_mode: str = "auto",
    min_pair_gap_samples: int = 1,
    max_pair_gap_samples: int = 50,
) -> Path:
    """Read one ECG log and its pulse companion into parquet."""
    inFile = strip_log_suffix(inFile)
    outFile = Path(inFile + ".parquet")

    ecg_file = inFile + ".ecg"
    puls_file = inFile + ".puls"

    ecg_arr = read_log_to_array(ecg_file)
    ecg_start, ecg_end = read_mdh_times(ecg_file)

    if not Path(puls_file).is_file():
        raise FileNotFoundError(
            f"Pulse log not found: {puls_file}. "
            "Use prep_read_available_logs_to_parquet.py for partial captures."
        )
    puls_arr = read_log_to_array(puls_file)
    puls_start, puls_end = read_mdh_times(puls_file)
    puls_filtered = puls_arr[~np.isin(puls_arr, (5000, 6000))]

    if extract_trigger:
        ecg_1, ecg_2, ecg_triggers = extract_ecg_and_trigger(ecg_arr)
        trigger_array = LoggedArrayMS(
            data=ecg_triggers,
            start_ms=ecg_start,
            end_ms=ecg_end,
            dt_ms=SAMPLE_INTERVAL_MS,
            snap_to_grid=snap_to_grid,
        )
    else:
        ecg_filtered = ecg_arr[~np.isin(ecg_arr, (5000, 6000))]
        ecg_1 = ecg_filtered[::2]
        ecg_2 = ecg_filtered[1::2]

    pulse_array = LoggedArrayMS(
        data=puls_filtered,
        start_ms=puls_start,
        end_ms=puls_end,
        dt_ms=SAMPLE_INTERVAL_MS,
        snap_to_grid=snap_to_grid,
    )
    ecg1_array = LoggedArrayMS(
        data=ecg_1,
        start_ms=ecg_start,
        end_ms=ecg_end,
        dt_ms=SAMPLE_INTERVAL_MS,
        snap_to_grid=snap_to_grid,
    )
    ecg2_array = LoggedArrayMS(
        data=ecg_2,
        start_ms=ecg_start,
        end_ms=ecg_end,
        dt_ms=SAMPLE_INTERVAL_MS,
        snap_to_grid=snap_to_grid,
    )

    if extract_trigger:
        (
            pulse_pruned,
            ecg1_pruned,
            ecg2_pruned,
            trigger_pruned,
            common_times,
        ) = prune_to_common_samples4(
            pulse_array,
            ecg1_array,
            ecg2_array,
            trigger_array,
        )
        trigger_pruned = np.asarray(trigger_pruned)
    else:
        pulse_pruned, ecg1_pruned, ecg2_pruned, common_times = (
            prune_to_common_samples3(
                pulse_array,
                ecg1_array,
                ecg2_array,
            )
        )

    pulse_pruned = np.asarray(pulse_pruned)
    ecg1_pruned = np.asarray(ecg1_pruned)
    ecg2_pruned = np.asarray(ecg2_pruned)
    ecg1_offset, ecg2_offset = resolve_ecg_offsets(
        ecg1_pruned,
        ecg2_pruned,
        ecg_offset_mode,
    )
    pulse_pruned = subtract_offset_float64(
        pulse_pruned, 2048, name="PMU pulse"
    )
    ecg1_pruned = subtract_offset_float64(
        ecg1_pruned, ecg1_offset, name="PMU ECG1"
    )
    ecg2_pruned = subtract_offset_float64(
        ecg2_pruned, ecg2_offset, name="PMU ECG2"
    )

    df = pd.DataFrame(
        {
            "time_ms": common_times - common_times[0],
            "puls_raw": pulse_pruned,
            "ecg1_raw": ecg1_pruned,
            "ecg2_raw": ecg2_pruned,
            "ecg3_raw": ecg2_pruned - ecg1_pruned,
            "pred": 0,
            "prob": 0.0,
            "target": 0,
            "siemens_tr": trigger_pruned if extract_trigger else 0,
            "raw_idx": np.arange(len(common_times), dtype=np.int64),
            "puls_peak": 0,
            "theta": 0.0,
        }
    )
    df = trim_to_full_pulse_blocks(df, fix_time=False)
    df["time_ms"] = (
        np.arange(len(df), dtype=np.float64) * SAMPLE_INTERVAL_MS
    )

    if extract_trigger:
        cleaned, stats = clean_siemens_trigger_pairs(
            df["siemens_tr"].to_numpy(),
            min_pair_gap_samples=min_pair_gap_samples,
            max_pair_gap_samples=max_pair_gap_samples,
        )
        df["siemens_tr"] = cleaned
        print(
            "Siemens trigger cleanup: "
            f"events={stats.complete_events}, "
            f"removed_incomplete={stats.removed_incomplete}, "
            f"changed_samples={stats.changed_samples}"
        )

    df = reorder_columns(df)
    print(f"DataFrame Size: {len(df):_}")

    int16_columns = ("puls_raw", "ecg1_raw", "ecg2_raw", "ecg3_raw")
    uint8_columns = ("target", "siemens_tr")
    if len(df):
        df = cast_columns_to_int16(df, int16_columns, copy=False)
        df = cast_columns_to_uint8(df, uint8_columns, copy=False)
    else:
        for column in int16_columns:
            df[column] = df[column].astype(np.int16)
        for column in uint8_columns:
            df[column] = df[column].astype(np.uint8)

    df.to_parquet(outFile, compression="snappy", index=False)
    return outFile


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "inputs",
        nargs="+",
        help=(
            "PMU log base path, .ecg/.puls file, directory, or glob pattern to convert."
        ),
    )
    parser.add_argument(
        "--no-extract-trigger",
        action="store_false",
        dest="extract_trigger",
        help="Do not extract the Siemens MR-system trigger from ECG markers.",
    )
    parser.add_argument(
        "--no-snap-to-grid",
        action="store_true",
        help="Do not snap MDH timestamps to the nearest sample grid.",
    )
    parser.add_argument(
        "--ecg-offset-mode",
        choices=ECG_OFFSET_MODES,
        default="auto",
        help=(
            "ECG baseline assignment: auto, standard "
            "(ecg1=2048/ecg2=10240), or swapped."
        ),
    )
    parser.add_argument(
        "--min-pair-gap-samples",
        type=int,
        default=1,
        help="Minimum Siemens marker gap for one complete event.",
    )
    parser.add_argument(
        "--max-pair-gap-samples",
        type=int,
        default=50,
        help="Maximum Siemens marker gap for one complete event.",
    )
    args = parser.parse_args(argv)

    try:
        in_files = expand_input_paths(args.inputs)
        for in_file in in_files:
            print(f"Converting {in_file} -> {in_file}.parquet")
            read_log_to_parquet(
                inFile=in_file,
                extract_trigger=args.extract_trigger,
                snap_to_grid=not args.no_snap_to_grid,
                ecg_offset_mode=args.ecg_offset_mode,
                min_pair_gap_samples=args.min_pair_gap_samples,
                max_pair_gap_samples=args.max_pair_gap_samples,
            )
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
