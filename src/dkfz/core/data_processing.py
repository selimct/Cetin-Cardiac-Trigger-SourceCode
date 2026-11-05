# Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Developed at the German Cancer Research Center (DKFZ).

from collections.abc import Sequence
import warnings

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, find_peaks

LIVE_DATA_COLUMN_ORDER = (
    "time_ms",
    "puls_raw",
    "ecg1_raw",
    "ecg2_raw",
    "ecg3_raw",
    "ecg4_raw",
    "prob",
    "pred",
    "target",
    "siemens_tr",
    "raw_idx",
    "puls_peak",
    "mhd_peak",
    "theta",
    "ecg1_rotated",
    "ecg2_rotated",
    "ecg3_rotated",
)


def rotate(vector, angle, rad=True):
    if not rad:
        angle = np.pi / 180 * angle
    s = np.sin(angle)
    c = np.cos(angle)
    rotation = np.array([[c, -s], [s, c]], dtype=float)
    return np.asarray(vector, dtype=float) @ rotation.T


def to_vector(x, y):
    return np.vstack((x, y)).T


def from_vector(v):
    return np.vsplit(np.array(v).T, 2)


def reorder_columns(
    df: pd.DataFrame,
    column_order: Sequence[str] = LIVE_DATA_COLUMN_ORDER,
    drop_unnamed: bool = False,
    create_missing_columns: bool = False,
) -> pd.DataFrame:
    """Return a dataframe with requested columns first and all extras last."""
    if "idx" in df.columns:
        warnings.warn('Found "idx" column. Changed name to "raw_idx".', stacklevel=2)
        df.rename(columns={"idx": "raw_idx"}, inplace=True)

    missing_columns = [
        col
        for col in column_order
        if col not in df.columns and create_missing_columns
    ]
    if missing_columns:
        df = df.assign(**{col: 0 for col in missing_columns})

    order_set = set(column_order)
    ordered_columns = [col for col in column_order if col in df.columns]
    if not drop_unnamed:
        ordered_columns += [col for col in df.columns if col not in order_set]

    return df.loc[:, ordered_columns]


def peak_find_in_window(
    ecg_signal,
    puls_signal,
    ecg_threshold=0.8,
    ppg_threshold=0.5,
    ppg_prominence=0.3,
    ppg_distance=200,
    ecg_distance=200,
    min_dist=100,
    max_dist=250,
    select_highest=False,
):
    """Find ECG R-peak within a window before the PPG peak.

    Args:
        ecg_signal (Pandas Dataframe Column): ECG dataframe entries
        puls_signal (Pandas Dataframe Column): PPG dataframe entries
        ecg_threshold (float, optional): ECG threshold above which the peaks have to be. Defaults to 0.8.
        ppg_threshold (float, optional): PPG threshold below which the PPG value has to be for an R peak to be detected. Defaults to 0.5.
        ppg_prominence (float, optional): PPG prominence threshold. Defaults to 0.3.
        ppg_distance (int, optional): Distance between PPG peaks. Defaults to 300.
        ecg_distance (int, optional): Distance between ECG peaks. Defaults to 200.
        min_dist (int, optional): Closest distance from R-peak to PPG peak. Defaults to 100.
        max_dist (int, optional): Furthest distance from R-peak to PPG peak. Defaults to 250.
        select_highest (bool, optional): Go through entire window of possible peak location to select highest point in window. Defaults to False.

    Returns:
        list: ECG R-peak indices.
    """
    peaks = []  # store all detected peaks
    in_peak = False
    peak_idx = None
    peak_val = -np.inf
    last_peak = -np.inf

    puls_filtered = puls_signal

    peak_pos, props = find_peaks(
        puls_filtered,
        distance=ppg_distance,
        prominence=ppg_prominence,
    )

    pre_flag = np.zeros(len(puls_signal), dtype=int)

    for f in peak_pos:
        start = max(f - max_dist, 0)
        end = max(f - min_dist, 0)
        pre_flag[start:end] = 1

    ecg_filtered = ecg_signal

    for i, val in enumerate(ecg_filtered):
        if (
            val >= ecg_threshold
            and pre_flag[i] == 1
            and (i - last_peak) >= ecg_distance
        ):
            if not in_peak:
                # start of a new peak region
                in_peak = True
                peak_idx = i
                peak_val = val
            elif val > peak_val:
                # found a higher point within the same region
                peak_val = val
                peak_idx = i
        elif (
            val < ecg_threshold
            and pre_flag[i] == 1
            and (i - last_peak) >= ecg_distance
            and select_highest == True
        ):
            # dropped below threshold but is still within window of possible peak location
            continue
        else:
            if in_peak:
                # ecg_signal dropped below threshold -> save this peak
                if puls_signal[peak_idx] < ppg_threshold:
                    peaks.append(peak_idx)
                in_peak = False
                peak_val = -np.inf
                last_peak = peak_idx
                peak_idx = None

    # handle last peak if ecg_signal ends above threshold
    if in_peak and peak_idx is not None:
        peaks.append(peak_idx)

    return peaks
