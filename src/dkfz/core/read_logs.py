# Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Developed at the German Cancer Research Center (DKFZ).

from __future__ import annotations
import numpy as np
import pandas as pd
import re
from dataclasses import dataclass
from typing import Sequence, Optional
from datetime import datetime, timezone
from typing import List, Tuple


def keep_only_numbers(s, allow_decimal=False):
    """
    Removes all characters from string `s` except digits.
    If allow_decimal=True, keeps one decimal point as well.
    """
    if allow_decimal:
        # Keep digits and decimal points, but only one decimal point allowed
        # First, remove everything except digits and dots
        s_filtered = re.sub(r"[^0-9.\s]", "", s)
        # Then, keep only the first decimal point (if any)
        parts = s_filtered.split(".")
        if len(parts) == 1:
            # No decimal points, just digits
            return parts[0]
        else:
            # Join the first part with one decimal point and the rest without dots
            return parts[0] + "." + "".join(parts[1:])
    else:
        # Keep only digits
        return re.sub(r"[^0-9\s]", "", s)


def read_mdh_times(filepath):
    start_time = None
    stop_time = None

    with open(filepath, "r") as f:
        for line in f:
            if line.startswith("LogStartMDHTime:"):
                start_time = int(line.split(":")[1].strip())
            elif line.startswith("LogStopMDHTime:"):
                stop_time = int(line.split(":")[1].strip())

    return start_time, stop_time


def read_log_to_array(filename):
    """
    Reads a log file with a preamble and comments after data.
    Assumes data lines contain only floats separated by whitespace.

    Parameters:
        filename (str): Path to the log file.
        comment_char (str): Character indicating start of comment lines.

    Returns:
        np.ndarray: NumPy array containing the data.
    """
    data = []
    with open(filename, "r") as file:
        reading_data = False
        for line in file:
            line = line.strip()
            if not line:
                # Skip empty lines anywhere
                continue

            line = keep_only_numbers(line)
            if not line.strip():
                continue

            # only read first line
            if reading_data:
                break

            line = re.sub(r"1.*?6002", "", line, count=1)

            line = re.sub(r"5002.*?6002", "", line)

            line = re.sub(r"5003", "", line)

            # Try to parse line as floats
            parts = line.split()

            try:
                floats = [float(p) for p in parts]
            except ValueError:
                continue
            # If we reach here, this line is data
            reading_data = True
            data.append(floats)

    reading_data = False
    data_arr = np.array(data).flatten()
    return data_arr


@dataclass
class LoggedArrayMS:
    data: Sequence[float] | Sequence[int]
    start_ms: int | float
    end_ms: int | float
    dt_ms: int | float = 2.5  # sampling interval in ms
    snap_to_grid: bool = True  # set True to snap timestamps to dt grid

    def _make_times(self) -> np.ndarray:
        n = len(self.data)
        times = self.start_ms + np.arange(n, dtype=np.float64) * float(self.dt_ms)
        if self.snap_to_grid:
            times = np.round(times / self.dt_ms) * self.dt_ms
        return np.round(times, 3)

    def to_series(self, name: str) -> pd.Series:
        n = len(self.data)
        inferred_end = self.start_ms + (n - 1) * self.dt_ms
        tol = 0.51 * self.dt_ms
        if abs(self.end_ms - inferred_end) > tol:
            raise ValueError(
                f"[{name}] end_ms mismatch: provided {self.end_ms}, "
                f"inferred {inferred_end} from len={n}, dt_ms={self.dt_ms}"
            )
        idx = pd.Index(self._make_times(), name="t_ms")  # float index
        return pd.Series(self.data, index=idx, name=name)


def prune_to_common_samples(
    arr1: LoggedArrayMS, arr2: LoggedArrayMS
) -> Tuple[List[float], List[float], np.ndarray]:
    """
    Return (data1_pruned, data2_pruned, common_times_ms), keeping ONLY samples whose
    timestamps exist in BOTH arrays. No interpolation, exact timestamp match.
    """
    t1 = arr1._make_times()
    t2 = arr2._make_times()
    common = np.intersect1d(t1, t2, assume_unique=False)

    # Build lookup from time -> value for each stream
    map1 = dict(zip(t1, arr1.data))
    map2 = dict(zip(t2, arr2.data))

    data1_p = [map1[t] for t in common]
    data2_p = [map2[t] for t in common]
    return data1_p, data2_p, common


def prune_to_common_samples3(
    arr1: LoggedArrayMS, arr2: LoggedArrayMS, arr3: LoggedArrayMS
) -> Tuple[List[float], List[float], List[float], np.ndarray]:
    """
    Return (data1_pruned, data2_pruned, data3_pruned, common_times_ms),
    keeping ONLY samples whose timestamps exist in ALL THREE arrays.
    No interpolation, exact timestamp match.
    """
    t1 = arr1._make_times()
    t2 = arr2._make_times()
    t3 = arr3._make_times()

    common12 = np.intersect1d(t1, t2, assume_unique=False)
    common = np.intersect1d(common12, t3, assume_unique=False)

    # Lookup: time -> value
    map1 = dict(zip(t1, arr1.data))
    map2 = dict(zip(t2, arr2.data))
    map3 = dict(zip(t3, arr3.data))

    data1_p = [map1[t] for t in common]
    data2_p = [map2[t] for t in common]
    data3_p = [map3[t] for t in common]

    return data1_p, data2_p, data3_p, common


def prune_to_common_samples4(
    arr1: LoggedArrayMS, arr2: LoggedArrayMS, arr3: LoggedArrayMS, arr4: LoggedArrayMS
) -> Tuple[List[float], List[float], List[float], List[float], np.ndarray]:
    """
    Return (data1_pruned, data2_pruned, data3_pruned, data4_pruned, common_times_ms),
    keeping ONLY samples whose timestamps exist in ALL FOUR arrays.
    No interpolation, exact timestamp match.
    """
    t1 = arr1._make_times()
    t2 = arr2._make_times()
    t3 = arr3._make_times()
    t4 = arr4._make_times()

    common12 = np.intersect1d(t1, t2, assume_unique=False)
    common123 = np.intersect1d(common12, t3, assume_unique=False)
    common = np.intersect1d(common123, t4, assume_unique=False)

    # Lookup: time -> value
    map1 = dict(zip(t1, arr1.data))
    map2 = dict(zip(t2, arr2.data))
    map3 = dict(zip(t3, arr3.data))
    map4 = dict(zip(t4, arr4.data))

    data1_p = [map1[t] for t in common]
    data2_p = [map2[t] for t in common]
    data3_p = [map3[t] for t in common]
    data4_p = [map4[t] for t in common]

    return data1_p, data2_p, data3_p, data4_p, common


# needs to create one trigger signal array for both ecg1 and ecg2
def extract_ecg_and_trigger(raw):
    """extract triggers from raw ecg signal and split ecg signal into the two signal strands.

    Args:
        raw (np.ndarray): raw ecg file from ideacmdtool logging

    Returns:
        np.ndarrays: (ecg1, ecg2, triggers)
    """
    raw = np.asarray(raw)

    is_trigger = np.isin(raw, [5000, 6000])
    ecg = raw[~is_trigger]

    trigger_raw_idx = np.where(is_trigger)[0]

    # map trigger positions to cleaned signal index space
    trigger_idx = trigger_raw_idx - np.cumsum(is_trigger)[trigger_raw_idx]

    # keep only valid ones (ignore trigger before first sample)
    trigger_idx = trigger_idx[trigger_idx >= 0]

    ecg_1 = ecg[::2]
    ecg_2 = ecg[1::2]

    trigger = np.zeros(max(len(ecg_1), len(ecg_2)))
    trigger[trigger_idx // 2] = 1

    return ecg_1, ecg_2, trigger
