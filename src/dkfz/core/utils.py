# Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Developed at the German Cancer Research Center (DKFZ).

import numpy as np
import pandas as pd

from dkfz.core.data_processing import (
    from_vector,
    reorder_columns,
    rotate,
    to_vector,
)
from scipy.optimize import minimize_scalar
from dataclasses import dataclass
from typing import Optional, Sequence, Union
from pathlib import Path
import datetime
import os

from dkfz.core.colors import C, PALETTE, SIGNAL_COLORS

_PLOTTING_TEXT_STYLE = {
    "font.size": 20.0,
    "axes.titlesize": 28.0,
    "axes.labelsize": 26.0,
    "xtick.labelsize": 24.0,
    "ytick.labelsize": 24.0,
    "legend.fontsize": 20.0,
    "figure.titlesize": 28.0,
}
_PLOTTING_STYLE_APPLIED = False


def _repo_relative_path(path: str | Path) -> str:
    root = _find_repo_root()
    path = Path(path)
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return os.path.relpath(path, root)


def _looks_like_repo_root(path: Path) -> bool:
    return (path / "pyproject.toml").is_file() and (path / "src" / "dkfz").is_dir()


def _find_repo_root() -> Path:
    for start in (Path.cwd().resolve(), Path(__file__).resolve()):
        current = start if start.is_dir() else start.parent
        for parent in (current, *current.parents):
            if _looks_like_repo_root(parent):
                return parent
    return Path.cwd().resolve()


def getpyplot():
    """Return pyplot with the shared DKFZ/mplhep plotting style applied."""
    global _PLOTTING_STYLE_APPLIED

    from cycler import cycler
    import matplotlib.pyplot as plt
    import mplhep

    if not _PLOTTING_STYLE_APPLIED:
        mplhep.style.use(["LHCb2"])
        plt.rcParams.update(_PLOTTING_TEXT_STYLE)
        plt.rcParams["axes.prop_cycle"] = cycler(color=PALETTE)
        plt.rcParams["pdf.fonttype"] = 42
        plt.rcParams["ps.fonttype"] = 42
        plt.rcParams["svg.fonttype"] = "none"
        plt.rcParams["legend.frameon"] = True
        plt.rcParams["legend.framealpha"] = 0.8
        plt.rcParams["savefig.dpi"] = 600
        plt.rcParams["savefig.format"] = "pdf"
        _PLOTTING_STYLE_APPLIED = True

    return plt


def _get_pyplot():
    """Backward-compatible private alias for :func:`getpyplot`."""
    return getpyplot()


def cm_to_inch(*values):
    return tuple(v / 2.54 for v in values)


@dataclass
class ECGAverageResult:
    """
    Container for beat-aligned averages computed around R-peaks.

    Attributes
    ----------
    t : np.ndarray, shape (L,)
        Relative sample axis: [-pre, ..., 0, ..., +post].
    mean1, mean2 : np.ndarray, shape (L,)
        Mean beat waveform for ecg1 and ecg2.
    std1, std2 : np.ndarray, shape (L,) or None
        Beat-to-beat standard deviation for ecg1 and ecg2.
    segments1, segments2 : np.ndarray, shape (M, L) or None
        Extracted beat windows (M beats kept) for ecg1 and ecg2.
    mean_puls : np.ndarray, shape (L,) or None
        Mean beat waveform for puls.
    std_puls : np.ndarray, shape (L,) or None
        Beat-to-beat standard deviation for puls.
    segments3 : np.ndarray, shape (M, L) or None
        Extracted beat windows (M beats kept) for puls.
    """

    t: np.ndarray
    mean1: np.ndarray
    mean2: np.ndarray
    std1: Optional[np.ndarray] = None
    std2: Optional[np.ndarray] = None
    segments1: Optional[np.ndarray] = None
    segments2: Optional[np.ndarray] = None
    mean_puls: Optional[np.ndarray] = None
    std_puls: Optional[np.ndarray] = None
    segments3: Optional[np.ndarray] = None


def mean_window_around_rpeaks(
    ecg1: Sequence[float] | np.ndarray,
    ecg2: Sequence[float] | np.ndarray,
    r_peaks: Sequence[int] | np.ndarray,
    pre: int,
    post: int,
    puls: Sequence[float] | np.ndarray,
    drop_incomplete: bool = True,
) -> ECGAverageResult:
    """
    Compute mean (and standard deviation) of windows around each R-peak for two ECG leads
    and an additional signal (puls), returning results in a single structured object.

    Parameters
    ----------
    ecg1, ecg2 : array-like, shape (N,)
        ECG signals (must have identical length).
    r_peaks : array-like, shape (K,)
        R-peak sample indices (integers).
    pre, post : int
        Number of samples before and after each R-peak to include.
        Window length is L = pre + post + 1.
    puls : array-like, shape (N,)
        Additional signal sampled synchronously with ecg1/ecg2.
    drop_incomplete : bool, default=True
        If True, discard beats whose [rp-pre, rp+post] window exceeds signal bounds.
        If False, pad out-of-bounds parts with NaN and compute nanmean/nanstd.

    Returns
    -------
    ECGAverageResult
        Dataclass holding the relative axis, mean/std waveforms, and extracted segments.
    """
    ecg1 = np.asarray(ecg1, dtype=float)
    ecg2 = np.asarray(ecg2, dtype=float)
    puls = np.asarray(puls, dtype=float)
    r_peaks = np.asarray(r_peaks, dtype=int)

    if ecg1.shape != ecg2.shape or ecg1.shape != puls.shape:
        raise ValueError("ecg1, ecg2, and puls must have the same shape")

    N = ecg1.size
    L = pre + post + 1
    t = np.arange(-pre, post + 1)

    segments1 = []
    segments2 = []
    segments3 = []

    for rp in r_peaks:
        start = rp - pre
        end = rp + post + 1  # python slice end is exclusive

        if start >= 0 and end <= N:
            segments1.append(ecg1[start:end])
            segments2.append(ecg2[start:end])
            segments3.append(puls[start:end])
        elif not drop_incomplete:
            seg1 = np.full(L, np.nan)
            seg2 = np.full(L, np.nan)
            seg3 = np.full(L, np.nan)

            s0 = max(start, 0)
            e0 = min(end, N)

            # indices inside the segment arrays where real data will go
            seg_start = s0 - start
            seg_end = seg_start + (e0 - s0)

            seg1[seg_start:seg_end] = ecg1[s0:e0]
            seg2[seg_start:seg_end] = ecg2[s0:e0]
            seg3[seg_start:seg_end] = puls[s0:e0]

            segments1.append(seg1)
            segments2.append(seg2)
            segments3.append(seg3)

    if len(segments1) == 0:
        raise ValueError(
            "No valid R-peak windows found. Check pre/post and r_peaks indices."
        )

    segments1 = np.vstack(segments1)
    segments2 = np.vstack(segments2)
    segments3 = np.vstack(segments3)

    if drop_incomplete:
        mean1 = segments1.mean(axis=0)
        mean2 = segments2.mean(axis=0)
        mean_puls = segments3.mean(axis=0)

        std1 = segments1.std(axis=0, ddof=1) if segments1.shape[0] > 1 else np.zeros(L)
        std2 = segments2.std(axis=0, ddof=1) if segments2.shape[0] > 1 else np.zeros(L)
        std_puls = (
            segments3.std(axis=0, ddof=1) if segments3.shape[0] > 1 else np.zeros(L)
        )

    else:
        mean1 = np.nanmean(segments1, axis=0)
        mean2 = np.nanmean(segments2, axis=0)
        mean_puls = np.nanmean(segments3, axis=0)

        std1 = np.nanstd(segments1, axis=0, ddof=1)
        std2 = np.nanstd(segments2, axis=0, ddof=1)
        std_puls = np.nanstd(segments3, axis=0, ddof=1)

    return ECGAverageResult(
        t=t,
        mean1=mean1,
        mean2=mean2,
        std1=std1,
        std2=std2,
        segments1=segments1,
        segments2=segments2,
        mean_puls=mean_puls,
        std_puls=std_puls,
        segments3=segments3,
    )


def plot_mean_around_rpeaks(
    t,
    mean1,
    mean2,
    std1=None,
    std2=None,
    fs=None,
    segments1=None,
    segments2=None,
    show_individual=False,
    max_individual=50,
    mean_puls=None,
    std_puls=None,
    segments3=None,
):
    """
    Plot beat-aligned mean waveforms around R-peaks for two ECG leads in one figure.

    Parameters
    ----------
    t : array-like, shape (L,)
        Relative axis for the extracted window (typically sample offsets
        ``[-pre, ..., 0, ..., +post]``). If `fs` is given, `t/fs` is used as the x-axis
        in seconds; otherwise `t` is used directly (samples).
    mean1, mean2 : array-like, shape (L,)
        Mean beat waveform for ECG lead 1 and ECG lead 2.
    std1, std2 : array-like, shape (L,), optional
        Beat-to-beat standard deviation for `mean1`/`mean2`. If provided, a ±1 sigma band
        is drawn around the corresponding mean curve.
    fs : float, optional
        Sampling rate in Hz. If provided, the x-axis is labeled in seconds; if None,
        the x-axis is labeled in samples.
    segments1, segments2 : array-like, shape (M, L), optional
        Individual beat windows for lead 1 and lead 2. If `show_individual=True`,
        up to `max_individual` beats are overlaid as semi-transparent traces.
    show_individual : bool, default False
        If True, overlay individual beat windows from `segments1`/`segments2` (and
        `segments3` if provided).
    max_individual : int, default 50
        Maximum number of individual beats to overlay (limits visual clutter).
    mean_puls : array-like, shape (L,), optional
        Mean beat waveform of an additional synchronous signal (e.g., pulse/PPG).
        If provided, it is plotted on both subplots using the project palette.
    std_puls : array-like, shape (L,), optional
        Beat-to-beat standard deviation for `mean_puls`. If provided, a ±1 sigma band is
        drawn around `mean_puls` on both subplots.
    segments3 : array-like, shape (M, L), optional
        Individual beat windows for the additional signal. If `show_individual=True`,
        up to `max_individual` windows are overlaid on both subplots.

    Notes
    -----
    - Creates two vertically stacked subplots with a shared x-axis.
    - Draws a vertical reference line at x = 0 (the R-peak alignment point).
    - Calls `plt.show()` and returns nothing.
    """
    plt = _get_pyplot()

    t = np.asarray(t)
    mean1 = np.asarray(mean1)
    mean2 = np.asarray(mean2)

    if fs is not None:
        x = t / fs
        xlabel = "Time relative to R-peak (s)"
    else:
        x = t
        xlabel = "Samples relative to R-peak"

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, constrained_layout=True)

    # Optional: overlay individual beats (thin lines)
    if show_individual and segments1 is not None and segments2 is not None:
        seg1 = np.asarray(segments1)
        seg2 = np.asarray(segments2)
        m = min(seg1.shape[0], max_individual)

        for i in range(m):
            ax1.plot(x, seg1[i], linewidth=0.7, alpha=0.35)
            ax2.plot(x, seg2[i], linewidth=0.7, alpha=0.35)

    if show_individual and segments3 is not None:
        seg3 = np.asarray(segments3)
        m = min(seg3.shape[0], max_individual)

        for i in range(m):
            ax1.plot(x, seg3[i], linewidth=0.7, alpha=0.35)
            ax2.plot(x, seg3[i], linewidth=0.7, alpha=0.35)

    # Mean curves
    ax1.plot(x, mean1, linewidth=2.0)
    ax2.plot(x, mean2, linewidth=2.0)

    if mean_puls is not None:
        mean_puls = np.asarray(mean_puls)
        ax1.plot(x, mean_puls, linewidth=1.0, color=C["steel_blue"])
        ax2.plot(x, mean_puls, linewidth=1.0, color=C["steel_blue"])

    # Optional: ±1σ shading
    if std1 is not None:
        std1 = np.asarray(std1)
        ax1.fill_between(x, mean1 - std1, mean1 + std1, alpha=0.3)
    if std2 is not None:
        std2 = np.asarray(std2)
        ax2.fill_between(x, mean2 - std2, mean2 + std2, alpha=0.3)
    if std_puls is not None:
        std_puls = np.asarray(std_puls)
        ax1.fill_between(
            x,
            mean_puls - std_puls,
            mean_puls + std_puls,
            alpha=0.3,
            color=C["steel_blue"],
        )
        ax2.fill_between(
            x,
            mean_puls - std_puls,
            mean_puls + std_puls,
            alpha=0.3,
            color=C["steel_blue"],
        )

    # Visual marker at R-peak (x=0)
    ax1.axvline(0, linewidth=1.0, alpha=0.6)
    ax2.axvline(0, linewidth=1.0, alpha=0.6)

    # ax1.set_title("ECG1: mean window around R-peaks")
    # ax2.set_title("ECG2: mean window around R-peaks")
    ax1.set_ylabel("ECG 1")
    ax2.set_ylabel("ECG 2")
    # ax2.set_xlabel(xlabel)

    ax1.grid(True)
    ax2.grid(True)

    plt.show()


def plot_mean_vcg_and_ecg(
    df: pd.DataFrame,
    vcglim=[-500, 1000],
    use_rot: bool = False,
    only_ecg1: bool = False,
    prepost_ms=450,
    step_size: int | None = 500,
    filepath: str | None = None,
    return_ax: bool = False,
    return_data: bool = False,
    show_rotated_vcg: bool = False,
    vcg_filepath: str | Path | None = None,
    ecg_filepath: str | Path | None = None,
):
    """Plot the mean VCG next to two mean ECG traces.

    Set ``only_ecg1=True`` for raw ECG1 on the top right and rotated ECG1 on
    the bottom right. With a raw VCG (``use_rot=False``),
    ``show_rotated_vcg=True`` overlays the rotated VCG and its R-peak in
    grayscale on the left axis. Use ``vcg_filepath`` and ``ecg_filepath`` to
    save the left VCG and the two right ECG axes as separate PDF files.
    """
    from matplotlib.ticker import MultipleLocator

    plt = getpyplot()
    if use_rot and not only_ecg1:
        ECG_COLUMNS = ("ecg1_rotated", "ecg2_rotated")
        VCG_COLS = ECG_COLUMNS
    if only_ecg1:
        ECG_COLUMNS = ("ecg1_raw", "ecg1_rotated")
        if use_rot:
            VCG_COLS = ("ecg1_rotated", "ecg2_rotated")
        else:
            VCG_COLS = ("ecg1_raw", "ecg2_raw")
    if not use_rot and not only_ecg1:
        ECG_COLUMNS = ("ecg1_raw", "ecg2_raw")
        VCG_COLS = ("ecg1_raw", "ecg2_raw")

    TARGET_COLUMN = "target"
    SAMPLE_RATE_HZ = 400.0
    PRE_MS = prepost_ms
    POST_MS = prepost_ms

    r_peaks_out = np.flatnonzero(df[TARGET_COLUMN].to_numpy() != 0)

    mean_vcg = mean_window_around_rpeaks(
        ecg1=df[VCG_COLS[0]],
        ecg2=df[VCG_COLS[1]],
        puls=df["puls_raw"],
        r_peaks=r_peaks_out,
        pre=round(PRE_MS * SAMPLE_RATE_HZ / 1000),
        post=round(POST_MS * SAMPLE_RATE_HZ / 1000),
    )

    mean_rotated_vcg = None
    if show_rotated_vcg:
        mean_rotated_vcg = mean_window_around_rpeaks(
            ecg1=df["ecg1_rotated"],
            ecg2=df["ecg2_rotated"],
            puls=df["puls_raw"],
            r_peaks=r_peaks_out,
            pre=round(PRE_MS * SAMPLE_RATE_HZ / 1000),
            post=round(POST_MS * SAMPLE_RATE_HZ / 1000),
        )

    mean_ecg = mean_window_around_rpeaks(
        ecg1=df[ECG_COLUMNS[0]],
        ecg2=df[ECG_COLUMNS[1]],
        puls=df["puls_raw"],
        r_peaks=r_peaks_out,
        pre=round(PRE_MS * SAMPLE_RATE_HZ / 1000),
        post=round(POST_MS * SAMPLE_RATE_HZ / 1000),
    )

    vcg1 = mean_vcg.mean1
    vcg2 = mean_vcg.mean2
    vcg_std1 = mean_vcg.std1
    vcg_std2 = mean_vcg.std2

    ecg1 = mean_ecg.mean1
    ecg2 = mean_ecg.mean2
    std1 = mean_ecg.std1
    std2 = mean_ecg.std2
    segments1 = mean_ecg.segments1
    segments2 = mean_ecg.segments2

    t = mean_ecg.t / SAMPLE_RATE_HZ * 1000  # ms

    fig = plt.figure(figsize=(16, 10))

    # ---------------------------------------------------------
    # 1) Create the VCG axis first
    # ---------------------------------------------------------
    ax_vcg = fig.add_axes([0.08, 0.12, 0.38, 0.78])  # [left, bottom, width, height]

    ax_vcg.plot(
        vcg1,
        vcg2,
        color=C["red"],
        linewidth=1.5,
        zorder=2,
        label="Rotated VCG" if use_rot else "Raw VCG",
    )

    if mean_rotated_vcg is not None:
        rotated_vcg_color = "0.5"
        ax_vcg.plot(
            mean_rotated_vcg.mean1,
            mean_rotated_vcg.mean2,
            color=rotated_vcg_color,
            linewidth=1.5,
            zorder=1,
            label="Rotated VCG",
        )

        rotated_r_idx = np.argmin(np.abs(mean_rotated_vcg.t))
        ax_vcg.errorbar(
            x=mean_rotated_vcg.mean1[rotated_r_idx],
            y=mean_rotated_vcg.mean2[rotated_r_idx],
            xerr=mean_rotated_vcg.std1[rotated_r_idx],
            yerr=mean_rotated_vcg.std2[rotated_r_idx],
            capsize=3,
            fmt="o",
            mfc=ax_vcg.get_facecolor(),
            mec=rotated_vcg_color,
            ecolor=rotated_vcg_color,
            markersize=7,
            mew=2,
            lw=0.8,
            zorder=3,
            label="Rotated R-peak",
        )

    r_idx = np.argmin(np.abs(t))

    ax_vcg.errorbar(
        x=vcg1[r_idx],
        y=vcg2[r_idx],
        xerr=vcg_std1[r_idx],
        yerr=vcg_std2[r_idx],
        capsize=3,
        fmt="o",
        mfc=ax_vcg.get_facecolor(),
        mec=C["deep_violet"],
        ecolor=C["deep_violet"],
        markersize=7,
        lw=0.8,
        zorder=1,
    )
    ax_vcg.errorbar(
        x=vcg1[r_idx],
        y=vcg2[r_idx],
        fmt="o",
        mfc="none",
        mec=C["deep_violet"],
        ecolor=C["deep_violet"],
        markersize=7,
        capsize=3,
        mew=2,
        lw=0.8,
        zorder=3,
    )
    # Proxy artist for legend
    ax_vcg.errorbar(
        [np.nan],
        [np.nan],
        xerr=[1],
        yerr=[1],
        fmt="o",
        ms=7,
        mfc="white",
        mec=C["deep_violet"],
        # mew=2.5,
        ecolor=C["deep_violet"],
        elinewidth=0.8,
        capsize=3,
        label="R-peak",
    )

    ax_vcg.set_xlabel("ECG 1")
    ax_vcg.set_ylabel("ECG 2")
    # ax_vcg.set_box_aspect(1)  # make the VCG axis box square
    # ax_vcg.grid()

    ax_vcg.axhline(
        0,
        color="black",
        linestyle="--",
        linewidth=0.8,
        zorder=0,
    )

    ax_vcg.axvline(
        0,
        color="black",
        linestyle="--",
        linewidth=0.8,
        zorder=0,
    )

    ax_vcg.set_xlim(vcglim)
    ax_vcg.set_ylim(vcglim)

    ax_vcg.xaxis.set_major_locator(MultipleLocator(step_size))
    ax_vcg.yaxis.set_major_locator(MultipleLocator(step_size))
    ax_vcg.legend(loc="upper left")

    # Important: draw once so the final axis geometry is known
    fig.canvas.draw()

    # ---------------------------------------------------------
    # 2) Read the final VCG axis position
    # ---------------------------------------------------------
    vcg_pos = ax_vcg.get_position()

    # ---------------------------------------------------------
    # 3) Define ECG layout based on VCG height
    # ---------------------------------------------------------
    gap_x = 0.05  # horizontal gap between VCG and ECG column
    gap_y = 0.04  # vertical gap between the two ECG plots
    right_margin = 0.05

    ecg_left = vcg_pos.x1 + gap_x
    ecg_bottom = vcg_pos.y0
    ecg_total_height = vcg_pos.height
    ecg_width = 1.0 - ecg_left - right_margin
    ecg_height = (ecg_total_height - gap_y) / 2.0

    # Top ECG
    ax_ecg1 = fig.add_axes(
        [ecg_left, ecg_bottom + ecg_height + gap_y, ecg_width, ecg_height]
    )

    # Bottom ECG
    ax_ecg2 = fig.add_axes(
        [ecg_left, ecg_bottom, ecg_width, ecg_height], sharex=ax_ecg1
    )

    # ---------------------------------------------------------
    # 4) Plot ECGs
    # ---------------------------------------------------------
    ax_ecg1.plot(
        t, ecg1, color=SIGNAL_COLORS["ecg1_raw"], linewidth=1.2, zorder=2, label="ECG1"
    )
    ax_ecg1.fill_between(
        t,
        ecg1 - std1,
        ecg1 + std1,
        color=SIGNAL_COLORS["ecg1_raw"],
        alpha=0.4,
        linewidth=0,
        zorder=2,
        label=r"$1\sigma$",
    )

    ax_ecg1.axvline(0, color=C["deep_violet"], linestyle="--", linewidth=0.8, zorder=1)
    ax_ecg1.fill_betweenx(
        vcglim,
        -20,
        20,
        color=C["deep_violet"],
        alpha=0.2,
        linewidth=0,
        label="R-peak",
        zorder=1,
    )
    ax_ecg1.axhline(
        0,
        color="black",
        linestyle="-",
        linewidth=1,
        zorder=0,
    )

    ax_ecg1.set_xticks(
        [-400, 0, 400],
        minor=False,
    )
    ax_ecg1.set_yticks([0], minor=False)
    ax_ecg1.set_ylim(vcglim)
    ax_ecg1.set_ylabel("ECG 1")
    ax_ecg1.legend(loc="upper left")

    # ax_ecg1.grid()
    ax_ecg1.tick_params(labelbottom=False, labelleft=True, labelright=False)

    ax_ecg2.plot(
        t,
        ecg2,
        color=SIGNAL_COLORS["ecg2_raw"],
        linewidth=1.2,
        zorder=2,
        label="ECG1 Rotated" if only_ecg1 else "ECG2",
    )
    ax_ecg2.fill_between(
        t,
        ecg2 - std2,
        ecg2 + std2,
        color=SIGNAL_COLORS["ecg2_raw"],
        alpha=0.4,
        linewidth=0,
        zorder=2,
        label=r"$1\sigma$",
    )

    ax_ecg2.axvline(0, color=C["deep_violet"], linestyle="--", linewidth=0.8, zorder=1)
    ax_ecg2.fill_betweenx(
        vcglim,
        -20,
        20,
        color=C["deep_violet"],
        alpha=0.2,
        linewidth=0,
        label="R-peak",
        zorder=1,
    )
    ax_ecg2.axhline(
        0,
        color="black",
        linestyle="-",
        linewidth=1,
        zorder=0,
    )

    ax_ecg2.set_xticks(
        [-400, 0, 400] if prepost_ms > 400 else [-200, 0, 200],
        minor=False,
    )
    ax_ecg2.set_yticks([0])  # ,minor=False)
    ax_ecg2.set_ylim(vcglim)
    ax_ecg2.set_xlabel("Time (ms)")
    if only_ecg1:
        ax_ecg2.set_ylabel("ECG 1")
    else:
        ax_ecg2.set_ylabel("ECG 2")
    ax_ecg2.legend(loc="upper left")
    ax_ecg2.tick_params(labelbottom=True, labelleft=True, labelright=False)

    # ax_ecg2.grid()

    output_saved = False

    if filepath is not None:
        fig.savefig(filepath, format="pdf")
        output_saved = True

    if vcg_filepath is not None or ecg_filepath is not None:
        from matplotlib.transforms import Bbox

        def save_axes_region(axes, output_path):
            exported_axes = set(axes)
            original_visibility = {axis: axis.get_visible() for axis in fig.axes}

            try:
                for axis in fig.axes:
                    axis.set_visible(axis in exported_axes)

                fig.canvas.draw()
                renderer = fig.canvas.get_renderer()
                bbox = Bbox.union([axis.get_tightbbox(renderer) for axis in axes])
                bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
                padding_inches = 0.1
                bbox = Bbox.from_extents(
                    bbox.x0 - padding_inches,
                    bbox.y0 - padding_inches,
                    bbox.x1 + padding_inches,
                    bbox.y1 + padding_inches,
                )
                fig.savefig(output_path, format="pdf", bbox_inches=bbox)
            finally:
                for axis, was_visible in original_visibility.items():
                    axis.set_visible(was_visible)

        if vcg_filepath is not None:
            save_axes_region([ax_vcg], vcg_filepath)
            output_saved = True

        if ecg_filepath is not None:
            save_axes_region([ax_ecg1, ax_ecg2], ecg_filepath)
            output_saved = True

    if not output_saved:
        plt.show()

    rt = ()

    if return_ax:
        rt += (ax_vcg, ax_ecg1, ax_ecg2)

    if return_data:
        rt += (mean_vcg, mean_ecg)

    return rt


def rotate_ecg(
    df_gauge,
    df=None,
):

    idx = np.where(df_gauge["target"] != 0)[0]
    vector = to_vector(df_gauge["ecg1_raw"].iloc[idx], df_gauge["ecg2_raw"].iloc[idx])

    fun = lambda theta: -np.mean(from_vector(rotate(vector, theta))[0])

    res = minimize_scalar(fun)
    theta = np.round(res.x, 5)
    print(f"Rotation angle {theta = :.4f} [rad] = {np.rad2deg(theta):.2f} [deg]")

    p1 = np.mean(df_gauge["ecg1_raw"].iloc[idx])
    print(f"Mean ECG R-peak amplitude for gauge signal before rotation: {p1:.2f}")

    e1_gauge_raw, e2_gauge_raw = from_vector(
        rotate(to_vector(df_gauge["ecg1_raw"], df_gauge["ecg2_raw"]), theta)
    )
    df_gauge["ecg1_rotated"] = e1_gauge_raw.ravel().astype(np.float32)
    df_gauge["ecg2_rotated"] = e2_gauge_raw.ravel().astype(np.float32)
    df_gauge["ecg3_rotated"] = (
        df_gauge["ecg2_rotated"] - df_gauge["ecg1_rotated"]
    ).astype(np.float32)
    df_gauge["theta"] = theta

    p2 = np.mean(df_gauge["ecg1_rotated"].iloc[idx])
    print(f"Mean ECG R-peak amplitude for gauge signal after rotation: {p2:.2f}")

    if df is not None:
        e1_raw, e2_raw = from_vector(
            rotate(to_vector(df["ecg1_raw"], df["ecg2_raw"]), theta)
        )
        df["theta"] = theta
        df["ecg1_rotated"] = e1_raw.ravel().astype(np.float32)
        df["ecg2_rotated"] = e2_raw.ravel().astype(np.float32)
        df["ecg3_rotated"] = (df["ecg2_rotated"] - df["ecg1_rotated"]).astype(
            np.float32
        )


def apply_temporal_tolerance(y_true, y_pred, window=3):
    """
    Shift tolerance: for each true label, if a correct label appears within ±w steps in y_pred,
    assign that label at the true index.
    """
    y_true = np.asarray(y_true)  # .astype(int).ravel()
    y_pred = np.asarray(y_pred)  # .astype(int).ravel()
    n = len(y_true)
    y_pred_tol = y_pred.copy()

    for t in range(n):
        a = max(0, t - window)
        b = min(n, t + window + 1)
        if y_true[t] in y_pred[a:b]:
            y_pred_tol[t] = y_true[t]  # mark it as correct in this window

    return y_pred_tol


def corr_annotated_data(filename: str, false_pos: list, false_neg: list):
    """Correct automatically annotated data by manually overriding the targets.

    Args:
        filename (str): input file.
        false_pos (list): annotated as 1, but actually 0.
        false_neg (list): annotated as 0, but actually 1.
    """
    df = pd.read_parquet(filename)
    false_pos = [x - 1 for x in false_pos]
    false_neg = [x - 1 for x in false_neg]
    df.loc[false_pos, "target"] = 0
    df.loc[false_neg, "target"] = 1
    return reorder_columns(df)


def read_all_parquet_files(root_dir):
    """
    Recursively reads all .parquet files under root_dir and
    prints their filenames in a single print statement.

    Parameters
    ----------
    root_dir : str or Path
        Root directory to search (including subdirectories).

    Returns
    -------
    list[pd.DataFrame]
        List of DataFrames loaded from the parquet files.
    """
    root_dir = Path(root_dir)
    parquet_files = list(root_dir.rglob("*.parquet"))

    # Read all parquet files
    dataframes = [pd.read_parquet(f) for f in parquet_files]

    # Print all filenames in ONE print command
    print(
        "Parquet files found:\n"
        + "\n".join(_repo_relative_path(f) for f in parquet_files)
    )

    return dataframes


def add_prediction_column(
    recording: pd.DataFrame | str | Path,
    all_events: pd.DataFrame | str | Path,
    *,
    out_file: str | Path | None = None,
) -> pd.DataFrame:
    """Add binary ``pred`` flags from an all-events comparison parquet.

    Inputs may be parquet paths or dataframes. Use the original recording
    passed to prediction and its corresponding ``prediction_all_events_compare``
    file. Both must have ``time_ms`` in milliseconds on the same time origin.
    Match exact timestamps, including ``sid`` when both inputs provide it.
    Without a shared ``sid``, each input must describe at most one recording.

    The prediction log timestamps refer to frame ends; its ``raw_idx`` refers
    to frame starts and must not be used as the timestamp's sample position.
    Each flag is placed on one raw row at the logged time, without expanding
    it across the frame or shifting it to the later trigger emission time.
    Other rows (including model warmup) receive zero; this does not indicate
    whether a prediction was available on those rows.

    Preserve row order, index labels, and other columns. Replace an existing
    ``pred`` column in the returned copy. Reject missing/duplicate time keys
    and any log rows that do not match the recording. Optionally save the
    result to ``out_file``; input files are never written implicitly.
    """
    raw = recording if isinstance(recording, pd.DataFrame) else pd.read_parquet(recording)
    log = all_events if isinstance(all_events, pd.DataFrame) else pd.read_parquet(all_events)
    for name, frame, required in (
        ("Recording", raw, ("time_ms",)),
        ("All-events", log, ("time_ms", "pred")),
    ):
        missing = set(required) - set(frame.columns)
        if missing:
            raise ValueError(f"{name} is missing columns: {sorted(missing)}")
        if "sid" in frame and frame["sid"].isna().any():
            raise ValueError(f"{name} contains missing recording IDs.")
        if not pd.api.types.is_numeric_dtype(frame["time_ms"]):
            raise ValueError(f"{name} time_ms must be numeric milliseconds.")
        if not np.isfinite(frame["time_ms"].to_numpy(dtype=float)).all():
            raise ValueError(f"{name} time_ms must contain only finite values.")

    if not log["pred"].isin([0, 1]).all():
        raise ValueError("All-events pred must contain only binary 0/1 flags.")

    keys = ["time_ms"]
    if "sid" in raw and "sid" in log:
        keys.insert(0, "sid")
    else:
        for frame in (raw, log):
            if "sid" in frame and frame["sid"].nunique() > 1:
                raise ValueError("Multiple recordings require 'sid' in both inputs.")

    raw_keys = pd.MultiIndex.from_frame(raw[keys])
    log_keys = pd.MultiIndex.from_frame(log[keys])
    if not raw_keys.is_unique or not log_keys.is_unique:
        raise ValueError("Time keys must be unique within each recording in both inputs.")
    positions = raw_keys.get_indexer(log_keys)
    if (positions < 0).any():
        raise ValueError(
            f"{int((positions < 0).sum())} all-events timestamps do not match the "
            "recording; use the original prediction input and the same time origin."
        )

    result = raw.copy()
    flags = np.zeros(len(raw), dtype=np.uint8)
    flags[positions] = log["pred"].to_numpy(dtype=np.uint8)
    result["pred"] = flags
    if out_file is not None:
        result.to_parquet(out_file)
    return result


def chop_microseconds(delta):
    return delta - datetime.timedelta(microseconds=delta.microseconds)


PathLike = Union[str, Path]


def ensure_dir(dir_path: PathLike) -> Path:
    """
    Ensure a directory exists.

    - If it does not exist: create it (including parents).
    - If it exists and is a directory: do nothing.
    - If it exists but is a file: raise an error.

    Returns
    -------
    pathlib.Path
        The resolved Path to the directory.
    """
    p = Path(dir_path)

    if p.exists() and not p.is_dir():
        raise NotADirectoryError(f"Path exists but is not a directory: {p}")

    p.mkdir(parents=True, exist_ok=True)
    return p.resolve()
