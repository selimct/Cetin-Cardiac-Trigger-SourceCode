# Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Developed at the German Cancer Research Center (DKFZ).

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from dkfz.core.data_processing import reorder_columns
from dkfz.live.io import require_raw_signal_column

DEFAULT_OUT_DIR = Path("data/flagged")


# =============================================================================
# Annotation settings
# =============================================================================

SETTINGS = {
    # Input
    "inFile": None,

    # Signal columns
    "ecg_col": "ecg1_rotated",
    "puls_col": "puls_raw",

    # ECG peak detection
    "ecg_min": None,
    "ecg_prominence": None,
    "ecg_distance": 200,
    "ecg_threshold_quantile": None,
    "ecg_prominence_fraction": None,
    "ecg_polarity": "positive",

    # Pulse peak detection
    "puls_min": None,
    "puls_prominence": None,
    "puls_distance": 200,
    "puls_threshold_quantile": 80.0,
    "puls_prominence_fraction": 0.3,

    # Adaptive pulse normalisation
    "adaptive_puls": True,
    "puls_adaptive_window": 4000,
    "puls_adaptive_min_periods": 200,

    # ECG-to-pulse timing constraints
    "min_dist": 110,
    "max_dist": 160,

    # R-peak selection
    "select_highest": False,

    # Legacy / hidden
    "ppg_max": None,
}



def _rolling_robust_normalize(
    values: np.ndarray,
    *,
    window_samples: int,
    min_periods: int,
    eps: float = 1e-8,
) -> np.ndarray:
    if window_samples <= 1:
        raise ValueError("puls_adaptive_window must be greater than 1.")
    if min_periods <= 0:
        raise ValueError("puls_adaptive_min_periods must be positive.")

    series = pd.Series(values, dtype=float)
    rolling = series.rolling(
        window=window_samples,
        center=True,
        min_periods=min(min_periods, len(series)),
    )
    median = rolling.median()
    q25 = rolling.quantile(0.25)
    q75 = rolling.quantile(0.75)
    scale = q75 - q25

    fallback_scale = _iqr(values)
    if fallback_scale <= eps:
        fallback_scale = 1.0

    scale = scale.where(scale > eps, fallback_scale)
    normalized = (series - median) / scale
    return normalized.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()


def _percentile_threshold(values: np.ndarray, quantile: float) -> float:
    if not 0 <= quantile <= 100:
        raise ValueError("threshold quantiles must be between 0 and 100.")
    return float(np.percentile(values, quantile))


def _iqr(values: np.ndarray) -> float:
    return float(np.percentile(values, 75) - np.percentile(values, 25))


def _auto_prominence(values: np.ndarray, fraction: float) -> float | None:
    if fraction < 0:
        raise ValueError("prominence fraction must be non-negative.")
    prominence = _iqr(values) * fraction
    return float(prominence) if prominence > 0 else None


def _default_output_path(in_file: str | Path) -> Path:
    input_path = Path(in_file)
    input_path = input_path if input_path.is_absolute() else ROOT / input_path
    data_dir = (ROOT / "data").resolve()

    try:
        data_relative = input_path.resolve().relative_to(data_dir)
    except ValueError as exc:
        raise ValueError(
            f"Input file must be under {data_dir}/<source>/..., got {in_file!r}."
        ) from exc

    if len(data_relative.parts) < 2:
        raise ValueError(
            f"Input file must be under {data_dir}/<source>/..., got {in_file!r}."
        )

    source_relative = Path(*data_relative.parts[1:])
    return ROOT / DEFAULT_OUT_DIR / source_relative


def _find_pulse_peaks(
    puls_signal: np.ndarray,
    *,
    puls_min: float | None,
    puls_prominence: float | None,
    puls_distance: int,
) -> np.ndarray:
    kwargs: dict[str, float | int] = {"distance": puls_distance}
    if puls_min is not None:
        kwargs["height"] = puls_min
    if puls_prominence is not None and puls_prominence > 0:
        kwargs["prominence"] = puls_prominence

    pulse_peaks, _ = find_peaks(puls_signal, **kwargs)
    return pulse_peaks.astype(int, copy=False)


def _find_r_peaks_in_pulse_windows(
    ecg_signal: np.ndarray,
    pulse_peaks: np.ndarray,
    *,
    ecg_min: float | None,
    ecg_prominence: float | None,
    ecg_distance: int,
    min_dist: int,
    max_dist: int,
    select_highest: bool,
) -> list[int]:
    r_peaks: list[int] = []
    peak_kwargs: dict[str, float] = {}
    if ecg_min is not None:
        peak_kwargs["height"] = ecg_min
    if ecg_prominence is not None and ecg_prominence > 0:
        peak_kwargs["prominence"] = ecg_prominence

    for pulse_peak in pulse_peaks:
        start = max(int(pulse_peak) - max_dist, 0)
        stop = max(int(pulse_peak) - min_dist + 1, 0)
        if start >= stop:
            continue

        local_ecg = ecg_signal[start:stop]
        candidates, _ = find_peaks(local_ecg, **peak_kwargs)
        if len(candidates) == 0:
            continue

        if select_highest:
            local_values = local_ecg[candidates]
            selected = int(candidates[int(np.argmax(local_values))]) + start
        else:
            selected = int(candidates[0]) + start

        if r_peaks and selected - r_peaks[-1] < ecg_distance:
            continue
        r_peaks.append(selected)

    return r_peaks


def window_annotate_data(
    inFile: str,
    outFile: str | None = None,
    outDir: str | None = None,
    ecg_col: str = "ecg1_raw",
    puls_col: str = "puls_raw",
    ecg_min: float | None = None,
    ecg_prominence: float | None = None,
    puls_min: float | None = None,
    puls_prominence: float | None = None,
    puls_distance: int = 200,
    ecg_distance: int = 200,
    min_dist: int = 100,
    max_dist: int = 200,
    select_highest: bool = False,
    ecg_threshold_quantile: float | None = None,
    ecg_prominence_fraction: float | None = None,
    puls_threshold_quantile: float = 80.0,
    puls_prominence_fraction: float = 0.3,
    adaptive_puls: bool = True,
    puls_adaptive_window: int = 4000,
    puls_adaptive_min_periods: int = 200,
    ecg_polarity: str = "positive",
    ppg_max: float | None = None,
) -> Path:
    """Annotate raw-rate parquet rows with candidate ECG R peaks.

    Detection uses temporary pulse-preceding windows in memory. The saved parquet
    preserves all input columns and values except for the target and puls_peak
    columns, then applies the standard live-data column order.
    """
    if ppg_max is not None:
        print("--ppg-max is ignored; pulse thresholding uses --puls-min.")
    if min_dist < 0 or max_dist <= min_dist:
        raise ValueError("Require 0 <= min_dist < max_dist.")
    if puls_distance <= 0 or ecg_distance <= 0:
        raise ValueError("puls_distance and ecg_distance must be positive.")
    if ecg_polarity not in {"positive", "negative"}:
        raise ValueError("ecg_polarity must be 'positive' or 'negative'.")
    if ecg_prominence is not None and ecg_prominence < 0:
        raise ValueError("ecg_prominence must be non-negative.")
    if ecg_prominence_fraction is not None and ecg_prominence_fraction < 0:
        raise ValueError("ecg_prominence_fraction must be non-negative.")

    df = pd.read_parquet(inFile)
    ecg_col = require_raw_signal_column(df, ecg_col, "ECG")
    puls_col = require_raw_signal_column(df, puls_col, "pulse")

    raw_ecg = df[ecg_col].to_numpy(dtype=float)
    ecg_signal = -raw_ecg if ecg_polarity == "negative" else raw_ecg
    puls_signal = df[puls_col].to_numpy(dtype=float)
    puls_detection_signal = (
        _rolling_robust_normalize(
            puls_signal,
            window_samples=puls_adaptive_window,
            min_periods=puls_adaptive_min_periods,
        )
        if adaptive_puls
        else puls_signal
    )

    use_ecg_prominence_without_height = (
        ecg_min is None
        and ecg_threshold_quantile is None
        and (
            (ecg_prominence is not None and ecg_prominence > 0)
            or (
                ecg_prominence_fraction is not None
                and ecg_prominence_fraction > 0
            )
        )
    )
    actual_ecg_min = None
    if not use_ecg_prominence_without_height:
        actual_ecg_threshold_quantile = (
            80.0 if ecg_threshold_quantile is None else ecg_threshold_quantile
        )
        actual_ecg_min = (
            _percentile_threshold(ecg_signal, actual_ecg_threshold_quantile)
            if ecg_min is None
            else float(ecg_min)
        )
    actual_ecg_prominence = (
        _auto_prominence(ecg_signal, ecg_prominence_fraction)
        if ecg_prominence is None and ecg_prominence_fraction is not None
        else (None if ecg_prominence is None else float(ecg_prominence))
    )
    actual_puls_min = (
        _percentile_threshold(puls_detection_signal, puls_threshold_quantile)
        if puls_min is None
        else float(puls_min)
    )
    actual_puls_prominence = (
        _auto_prominence(puls_detection_signal, puls_prominence_fraction)
        if puls_prominence is None
        else float(puls_prominence)
    )

    pulse_peaks = _find_pulse_peaks(
        puls_detection_signal,
        puls_min=actual_puls_min,
        puls_prominence=actual_puls_prominence,
        puls_distance=puls_distance,
    )
    r_peaks = _find_r_peaks_in_pulse_windows(
        ecg_signal,
        pulse_peaks,
        ecg_min=actual_ecg_min,
        ecg_prominence=actual_ecg_prominence,
        ecg_distance=ecg_distance,
        min_dist=min_dist,
        max_dist=max_dist,
        select_highest=select_highest,
    )

    target = np.zeros(len(df), dtype=np.uint8)
    target[r_peaks] = 1
    df["target"] = target

    pulse_target = np.zeros(len(df), dtype=np.uint8)
    pulse_target[pulse_peaks] = 1
    df["puls_peak"] = pulse_target

    if outDir is not None and outFile is None:
        print("--out-dir is ignored; output path is derived from the input data path.")
    output_path = Path(outFile) if outFile else _default_output_path(inFile)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = reorder_columns(df)
    df.to_parquet(output_path, compression="snappy")

    print(
        f"Detected {len(pulse_peaks)} pulse peaks and annotated {len(r_peaks)} "
        f"R peaks in {len(df)} rows."
    )
    pulse_threshold_label = (
        "Adaptive pulse detector" if adaptive_puls else "Raw pulse detector"
    )
    ecg_min_label = "off" if actual_ecg_min is None else f"{actual_ecg_min:.3g}"
    ecg_prominence_label = (
        "0" if actual_ecg_prominence is None else f"{actual_ecg_prominence:.3g}"
    )
    print(
        f"{pulse_threshold_label}: "
        f"{puls_col} >= {actual_puls_min:.3g}, "
        f"{puls_col} prominence >= "
        f"{0.0 if actual_puls_prominence is None else actual_puls_prominence:.3g}; "
        f"raw ECG threshold {ecg_col}({ecg_polarity}) >= "
        f"{ecg_min_label}, prominence >= {ecg_prominence_label}."
    )
    print(f"Wrote {output_path}")

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Annotate raw parquet data with ECG R-peak targets using pulse-window "
            "candidate regions."
        )
    )

    parser.add_argument(
        "inFile",
        type=str,
        nargs="?",
        default=SETTINGS["inFile"],
        help=(
            "Input parquet under data/<source>/... . Output is written to the "
            "same relative path under data/flagged without renaming the file."
        ),
    )

    parser.add_argument(
        "--ecg-col",
        type=str,
        default=SETTINGS["ecg_col"],
        help="Raw ECG column used for annotation.",
    )

    parser.add_argument(
        "--puls-col",
        type=str,
        default=SETTINGS["puls_col"],
        help="Raw pulse column used for annotation.",
    )

    parser.add_argument(
        "--ecg-min",
        type=float,
        default=SETTINGS["ecg_min"],
        help=(
            "Raw ECG threshold after polarity selection. Defaults to the "
            "--ecg-threshold-quantile value."
        ),
    )

    parser.add_argument(
        "--ecg-prominence",
        type=float,
        default=SETTINGS["ecg_prominence"],
        help=(
            "Raw ECG prominence threshold after polarity selection. When set "
            "without --ecg-min, ECG peak detection uses prominence without a "
            "height threshold."
        ),
    )

    parser.add_argument(
        "--puls-min",
        type=float,
        default=SETTINGS["puls_min"],
        help=(
            "Pulse peak height threshold. Uses adaptive-normalized units by "
            "default, or raw units with --no-adaptive-puls. Defaults to the "
            "--puls-threshold-quantile value in the active pulse detection signal."
        ),
    )

    parser.add_argument(
        "--puls-prominence",
        "--ppg-prominence",
        dest="puls_prominence",
        type=float,
        default=SETTINGS["puls_prominence"],
        help=(
            "Pulse peak prominence threshold. Uses adaptive-normalized units by "
            "default, or raw units with --no-adaptive-puls. Defaults to "
            "--puls-prominence-fraction times the active pulse detection signal IQR."
        ),
    )

    parser.add_argument(
        "--puls-distance",
        "--ppg-distance",
        dest="puls_distance",
        type=int,
        default=SETTINGS["puls_distance"],
        help="Minimum distance between pulse peaks in raw samples.",
    )

    parser.add_argument(
        "--ecg-distance",
        type=int,
        default=SETTINGS["ecg_distance"],
        help="Minimum distance between selected ECG R peaks in raw samples.",
    )

    parser.add_argument(
        "--min-dist",
        type=int,
        default=SETTINGS["min_dist"],
        help="Minimum allowed R-to-pulse delay in raw samples.",
    )

    parser.add_argument(
        "--max-dist",
        type=int,
        default=SETTINGS["max_dist"],
        help="Maximum allowed R-to-pulse delay in raw samples.",
    )

    parser.add_argument(
        "--ecg-threshold-quantile",
        type=float,
        default=SETTINGS["ecg_threshold_quantile"],
        help=(
            "ECG quantile used as threshold when --ecg-min is omitted. Defaults "
            "to 80.0 unless ECG prominence is set without an ECG height option."
        ),
    )

    parser.add_argument(
        "--ecg-prominence-fraction",
        type=float,
        default=SETTINGS["ecg_prominence_fraction"],
        help=(
            "ECG IQR fraction used as prominence when --ecg-prominence is "
            "omitted. When set without --ecg-min, ECG peak detection uses "
            "prominence without a height threshold."
        ),
    )

    parser.add_argument(
        "--puls-threshold-quantile",
        type=float,
        default=SETTINGS["puls_threshold_quantile"],
        help=(
            "Pulse quantile used as peak-height threshold when --puls-min is "
            "omitted. Applied to the active pulse detection signal."
        ),
    )

    parser.add_argument(
        "--puls-prominence-fraction",
        type=float,
        default=SETTINGS["puls_prominence_fraction"],
        help="Pulse IQR fraction used as prominence when --puls-prominence is omitted.",
    )

    parser.add_argument(
        "--adaptive-puls",
        action=argparse.BooleanOptionalAction,
        default=SETTINGS["adaptive_puls"],
        help=(
            "Use rolling median/IQR normalized pulse values for peak detection. "
            "Pass --no-adaptive-puls to disable."
        ),
    )

    parser.add_argument(
        "--puls-adaptive-window",
        type=int,
        default=SETTINGS["puls_adaptive_window"],
        help="Rolling window size in raw samples for adaptive pulse normalization.",
    )

    parser.add_argument(
        "--puls-adaptive-min-periods",
        type=int,
        default=SETTINGS["puls_adaptive_min_periods"],
        help="Minimum samples required for adaptive pulse rolling statistics.",
    )

    parser.add_argument(
        "--ecg-polarity",
        choices=("positive", "negative"),
        default=SETTINGS["ecg_polarity"],
        help="Use positive or negative ECG peaks for detection.",
    )

    parser.add_argument(
        "--select-highest",
        action=argparse.BooleanOptionalAction,
        default=SETTINGS["select_highest"],
        help="Select the highest ECG peak within each pulse-preceding window.",
    )

    parser.add_argument(
        "--ppg-max",
        type=float,
        default=SETTINGS["ppg_max"],
        help=argparse.SUPPRESS,
    )

    window_annotate_data(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
