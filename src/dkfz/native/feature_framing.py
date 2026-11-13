# Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Developed at the German Cancer Research Center (DKFZ).

"""Python-compatible adapters for the canonical native feature framer."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from dkfz.live.io import require_raw_signal_columns
from dkfz.native import _live_core

PREPROCESSING_CONTRACT_VERSION = _live_core.PREPROCESSING_CONTRACT_VERSION
ECG_SLOPE_MODES = ("none", "fast", "slow", "fast_slow")
SIGNAL_QUALITY_MODES = ("none", "ecg", "ecg_puls")
SIGNAL_QUALITY_FEATURE_COL = "signal_quality"

_SQ_ECG_COMPONENT_COLS = (
    "_signal_quality_ecg_log_rough",
    "_signal_quality_ecg_log_curvature",
    "_signal_quality_ecg_flat",
    "_signal_quality_ecg_log_scale_jump",
)
_SQ_PULS_COMPONENT_COLS = (
    "_signal_quality_puls_log_step",
    "_signal_quality_puls_log_curvature",
    "_signal_quality_puls_flat",
    "_signal_quality_puls_log_outlier",
)
_SQ_DEFAULT_Z_CLIP = 6.0
_SQ_DEFAULT_MIN_IQR = 0.05
_SQ_COMPONENT_MIN_IQR = {
    "_signal_quality_ecg_flat": 0.25,
    "_signal_quality_ecg_log_scale_jump": 0.1,
    "_signal_quality_puls_flat": 0.25,
}


@dataclass(frozen=True)
class LiveFeatureFrame:
    """One completed 50 Hz model feature frame."""

    frame_idx_50hz: int
    start_sample_idx_400hz: int
    end_sample_idx_400hz: int
    start_time_ms: float | None
    end_time_ms: float | None
    features: np.ndarray
    auxiliary_features: dict[str, float] = field(default_factory=dict)


RollingPairNormQuantileNormalizer = _live_core.NativeRollingPairNormQuantileNormalizer
RollingScalarQuantileNormalizer = _live_core.NativeRollingScalarQuantileNormalizer
SignalQualityTracker = _live_core.NativeSignalQualityTracker


def live_feature_columns(
    frame_size_raw: int = 8,
    include_puls: bool = True,
    ecg_cols: tuple[str, ...] = ("ecg1", "ecg2"),
    ecg_slope_mode: str = "none",
    signal_quality_mode: str = "none",
) -> list[str]:
    """Return feature names using the canonical C++ ordering."""
    return list(
        _live_core.live_feature_columns(
            frame_size_raw=int(frame_size_raw),
            include_puls=bool(include_puls),
            ecg_columns=list(ecg_cols),
            ecg_slope_mode=str(ecg_slope_mode),
            signal_quality_mode=str(signal_quality_mode),
        )
    )


def signal_quality_component_columns(
    mode: str,
    *,
    include_puls: bool = True,
) -> list[str]:
    """Return signal-quality component names from the C++ contract."""
    return list(
        _live_core.signal_quality_component_columns(
            str(mode),
            bool(include_puls),
        )
    )


def signal_quality_feature_from_components(
    components: dict[str, float],
    *,
    stats: dict | None,
    eps: float = 1e-6,
) -> float:
    """Apply signal-quality calibration in the shared C++ core."""
    return float(
        _live_core.signal_quality_feature_from_components(
            components,
            stats,
            float(eps),
        )
    )


class LiveFeatureFramer:
    """Python wrapper around the canonical native feature framer."""

    def __init__(
        self,
        raw_hz: float = 400.0,
        frame_size_raw: int = 8,
        include_puls: bool = True,
        ecg1_offset: float = 0.0,
        ecg2_offset: float = 0.0,
        ecg3_offset: float = 0.0,
        include_ecg3: bool = False,
        ecg_slope_mode: str = "none",
        ecg_slope_slow_window_samples: int = 40,
        signal_quality_mode: str = "none",
        signal_quality_stats: dict | None = None,
        ecg_rotation_angle: float = 0.0,
        puls_offset: float = 0.0,
        ecg_normalizer=None,
        puls_normalizer=None,
        align_to_puls_blocks: bool = True,
        verify_puls_constant_within_block: bool = False,
        puls_constant_tol: float = 1e-12,
        signal_quality_puls_flat_window_frames: int = 10,
        signal_quality_puls_median_window_frames: int = 200,
        reset_on_discontinuity: bool = False,
    ):
        ecg_normalizer = ecg_normalizer or RollingPairNormQuantileNormalizer()
        puls_normalizer = puls_normalizer or RollingScalarQuantileNormalizer()
        self.raw_hz = float(raw_hz)
        self.frame_size_raw = int(frame_size_raw)
        self.frame_hz = self.raw_hz / self.frame_size_raw
        self.include_puls = bool(include_puls)
        self.include_ecg3 = bool(include_ecg3)
        self.ecg_slope_mode = str(ecg_slope_mode)
        self.signal_quality_mode = str(signal_quality_mode)
        self.align_to_puls_blocks = bool(align_to_puls_blocks)
        self.verify_puls_constant_within_block = bool(verify_puls_constant_within_block)
        self.feature_cols = live_feature_columns(
            frame_size_raw=self.frame_size_raw,
            include_puls=self.include_puls,
            ecg_cols=(
                ("ecg1", "ecg2", "ecg3") if self.include_ecg3 else ("ecg1", "ecg2")
            ),
            ecg_slope_mode=self.ecg_slope_mode,
            signal_quality_mode=self.signal_quality_mode,
        )

        self._native = _live_core.NativeFeatureFramer(
            {
                "signal_quality_puls_flat_window_frames": int(
                    signal_quality_puls_flat_window_frames
                ),
                "signal_quality_puls_median_window_frames": int(
                    signal_quality_puls_median_window_frames
                ),
                "reset_on_discontinuity": bool(reset_on_discontinuity),
                "raw_hz": self.raw_hz,
                "frame_size_raw": self.frame_size_raw,
                "include_puls": self.include_puls,
                "ecg1_offset": float(ecg1_offset),
                "ecg2_offset": float(ecg2_offset),
                "ecg3_offset": float(ecg3_offset),
                "include_ecg3": self.include_ecg3,
                "ecg_slope_mode": self.ecg_slope_mode,
                "ecg_slope_slow_window_samples": int(ecg_slope_slow_window_samples),
                "signal_quality_mode": self.signal_quality_mode,
                "signal_quality_stats": signal_quality_stats,
                "ecg_rotation_angle": float(ecg_rotation_angle),
                "puls_offset": float(puls_offset),
                "ecg_quantile": float(ecg_normalizer.q),
                "ecg_norm_window_samples": int(ecg_normalizer.window_samples),
                "ecg_norm_eps": float(ecg_normalizer.eps),
                "ecg_norm_warmup_scale": float(ecg_normalizer.warmup_scale),
                "puls_quantile_low": float(puls_normalizer.q_low),
                "puls_quantile_high": float(puls_normalizer.q_high),
                "puls_norm_window_frames": int(puls_normalizer.window_frames),
                "puls_norm_eps": float(puls_normalizer.eps),
                "puls_norm_warmup_value": float(puls_normalizer.warmup_value),
                "puls_clip": bool(puls_normalizer.clip),
                "align_to_puls_blocks": self.align_to_puls_blocks,
                "verify_puls_constant_within_block": (
                    self.verify_puls_constant_within_block
                ),
                "puls_constant_tol": float(puls_constant_tol),
            }
        )

    def reset(self) -> None:
        self._native.reset()

    def _convert_native_frame(self, native_frame) -> LiveFeatureFrame:
        native_auxiliary = dict(native_frame.auxiliary_features)
        ordered_auxiliary = {
            name: native_auxiliary[name]
            for name in signal_quality_component_columns(
                self.signal_quality_mode,
                include_puls=True,
            )
            if name in native_auxiliary
        }
        ordered_auxiliary.update(
            {
                name: value
                for name, value in native_auxiliary.items()
                if name not in ordered_auxiliary
            }
        )
        return LiveFeatureFrame(
            frame_idx_50hz=int(native_frame.frame_idx_50hz),
            start_sample_idx_400hz=int(native_frame.start_sample_idx_400hz),
            end_sample_idx_400hz=int(native_frame.end_sample_idx_400hz),
            start_time_ms=native_frame.start_time_ms,
            end_time_ms=native_frame.end_time_ms,
            features=np.asarray(native_frame.features, dtype=np.float32),
            auxiliary_features=ordered_auxiliary,
        )

    def process_sample(
        self,
        ecg1: float,
        ecg2: float,
        puls: float | None = None,
        sample_idx_400hz: int | None = None,
        time_ms: float | None = None,
        ecg3: float | None = None,
    ) -> LiveFeatureFrame | None:
        native_frame = self._native.process_sample(
            ecg1=float(ecg1),
            ecg2=float(ecg2),
            puls=puls,
            sample_idx_400hz=sample_idx_400hz,
            time_ms=time_ms,
            ecg3=ecg3,
        )
        if native_frame is None:
            return None
        return self._convert_native_frame(native_frame)

    def process_arrays(
        self,
        ecg1: np.ndarray,
        ecg2: np.ndarray,
        puls: np.ndarray | None = None,
        sample_indices: np.ndarray | None = None,
        times_ms: np.ndarray | None = None,
        ecg3: np.ndarray | None = None,
    ) -> list[LiveFeatureFrame]:
        """Process one contiguous stream through a single native call."""
        native_frames = self._native.process_arrays(
            ecg1=np.asarray(ecg1, dtype=np.float64),
            ecg2=np.asarray(ecg2, dtype=np.float64),
            puls=None if puls is None else np.asarray(puls, dtype=np.float64),
            sample_indices=(
                None
                if sample_indices is None
                else np.asarray(sample_indices, dtype=np.int64)
            ),
            times_ms=(
                None if times_ms is None else np.asarray(times_ms, dtype=np.float64)
            ),
            ecg3=None if ecg3 is None else np.asarray(ecg3, dtype=np.float64),
        )
        return [self._convert_native_frame(frame) for frame in native_frames]


def make_live_framer(
    raw_hz: float = 400.0,
    frame_size_raw: int = 8,
    include_puls: bool = True,
    ecg_quantile: float = 0.999,
    ecg_norm_window_samples: int = 4000,
    ecg1_offset: float = 0.0,
    ecg2_offset: float = 0.0,
    ecg3_offset: float = 0.0,
    include_ecg3: bool = False,
    ecg_slope_mode: str = "none",
    ecg_slope_slow_window_samples: int = 40,
    signal_quality_mode: str = "none",
    signal_quality_stats: dict | None = None,
    ecg_rotation_angle: float = 0.0,
    puls_offset: float = 0.0,
    puls_quantile_low: float = 0.05,
    puls_quantile_high: float = 0.95,
    puls_norm_window_frames: int = 500,
    puls_norm_warmup_value: float = 0.5,
    puls_clip: bool = True,
    align_to_puls_blocks: bool = True,
    verify_puls_constant_within_block: bool = False,
    signal_quality_puls_flat_window_frames: int = 10,
    signal_quality_puls_median_window_frames: int = 200,
    reset_on_discontinuity: bool = False,
) -> LiveFeatureFramer:
    """Construct the native framer with the existing Python API."""
    return LiveFeatureFramer(
        signal_quality_puls_flat_window_frames=signal_quality_puls_flat_window_frames,
        signal_quality_puls_median_window_frames=signal_quality_puls_median_window_frames,
        reset_on_discontinuity=reset_on_discontinuity,
        raw_hz=raw_hz,
        frame_size_raw=frame_size_raw,
        include_puls=include_puls,
        ecg1_offset=ecg1_offset,
        ecg2_offset=ecg2_offset,
        ecg3_offset=ecg3_offset,
        include_ecg3=include_ecg3,
        ecg_slope_mode=ecg_slope_mode,
        ecg_slope_slow_window_samples=ecg_slope_slow_window_samples,
        signal_quality_mode=signal_quality_mode,
        signal_quality_stats=signal_quality_stats,
        ecg_rotation_angle=ecg_rotation_angle,
        puls_offset=puls_offset,
        ecg_normalizer=RollingPairNormQuantileNormalizer(
            q=ecg_quantile,
            window_samples=ecg_norm_window_samples,
        ),
        puls_normalizer=RollingScalarQuantileNormalizer(
            q_low=puls_quantile_low,
            q_high=puls_quantile_high,
            window_frames=puls_norm_window_frames,
            warmup_value=puls_norm_warmup_value,
            clip=puls_clip,
        ),
        align_to_puls_blocks=align_to_puls_blocks,
        verify_puls_constant_within_block=verify_puls_constant_within_block,
    )


def _validate_signal_quality_mode(mode: str) -> str:
    mode = str(mode)
    if mode not in SIGNAL_QUALITY_MODES:
        choices = ", ".join(SIGNAL_QUALITY_MODES)
        raise ValueError(f"signal_quality_mode must be one of: {choices}.")
    return mode


def _default_signal_quality_min_iqrs() -> dict[str, float]:
    min_iqrs = {
        col: _SQ_DEFAULT_MIN_IQR
        for col in list(_SQ_ECG_COMPONENT_COLS) + list(_SQ_PULS_COMPONENT_COLS)
    }
    min_iqrs.update(_SQ_COMPONENT_MIN_IQR)
    return min_iqrs


def _concat_frame_dfs(
    frame_dfs: list[pd.DataFrame] | pd.DataFrame,
) -> pd.DataFrame:
    if isinstance(frame_dfs, pd.DataFrame):
        return frame_dfs.copy()
    if not frame_dfs:
        raise ValueError("No frame data available.")
    return pd.concat(frame_dfs, ignore_index=True)


def fit_signal_quality_stats(
    frame_dfs: list[pd.DataFrame] | pd.DataFrame,
    *,
    mode: str,
    split_col: str = "split",
    train_split: str = "train",
    eps: float = 1e-6,
) -> dict | None:
    """Fit robust signal-quality normalization constants on training frames."""
    mode = _validate_signal_quality_mode(mode)
    if mode == "none":
        return None

    df = _concat_frame_dfs(frame_dfs)
    fit_df = df
    if split_col in df.columns:
        fit_df = df[df[split_col] == train_split]
    if fit_df.empty:
        raise ValueError("Cannot fit signal quality stats on an empty training split.")

    ecg_cols = list(_SQ_ECG_COMPONENT_COLS)
    missing_ecg = [col for col in ecg_cols if col not in fit_df.columns]
    if missing_ecg:
        raise ValueError(
            f"Signal quality components are missing from framed data: {missing_ecg}."
        )

    puls_cols = []
    if mode == "ecg_puls":
        candidate_puls_cols = list(_SQ_PULS_COMPONENT_COLS)
        if all(col in fit_df.columns for col in candidate_puls_cols):
            has_values = fit_df[candidate_puls_cols].notna().any().all()
            if bool(has_values):
                puls_cols = candidate_puls_cols

    medians: dict[str, float] = {}
    iqrs: dict[str, float] = {}
    min_iqrs = _default_signal_quality_min_iqrs()
    for col in ecg_cols + puls_cols:
        values = pd.to_numeric(fit_df[col], errors="coerce")
        values = values[np.isfinite(values.to_numpy(dtype=float))]
        values = values.dropna()
        if values.empty:
            raise ValueError(f"Cannot fit signal quality stats for empty {col!r}.")
        q25 = float(values.quantile(0.25))
        q50 = float(values.median())
        q75 = float(values.quantile(0.75))
        medians[col] = q50
        iqrs[col] = max(q75 - q25, eps)

    uses_puls = bool(puls_cols)
    return {
        "mode": mode,
        "feature_col": SIGNAL_QUALITY_FEATURE_COL,
        "transform": "exp_neg_half_weighted_chi2_iqr",
        "ecg_component_cols": ecg_cols,
        "puls_component_cols": puls_cols,
        "median": medians,
        "iqr": iqrs,
        "min_iqr": {col: min_iqrs[col] for col in ecg_cols + puls_cols},
        "z_clip": _SQ_DEFAULT_Z_CLIP,
        "ecg_weight": 0.6 if uses_puls else 1.0,
        "puls_weight": 0.4 if uses_puls else 0.0,
        "eps": float(eps),
    }


def apply_signal_quality_stats(
    frame_dfs: list[pd.DataFrame] | pd.DataFrame,
    *,
    stats: dict | None,
) -> list[pd.DataFrame]:
    """Apply fitted calibration through the C++ scalar implementation."""
    if stats is None:
        return frame_dfs if isinstance(frame_dfs, list) else [frame_dfs]

    frames = frame_dfs if isinstance(frame_dfs, list) else [frame_dfs]
    component_cols = list(stats.get("ecg_component_cols") or []) + list(
        stats.get("puls_component_cols") or []
    )
    outputs: list[pd.DataFrame] = []
    for frame in frames:
        output = frame.copy()
        missing = [
            col for col in stats.get("ecg_component_cols", []) if col not in output
        ]
        if missing:
            raise ValueError(
                "Signal quality components are missing from framed data: "
                f"{missing}. Regenerate the preprocessed cache from raw data."
            )
        component_values = output.reindex(columns=component_cols).to_numpy(
            dtype=np.float64
        )
        output[SIGNAL_QUALITY_FEATURE_COL] = _live_core.signal_quality_features(
            component_values,
            component_cols,
            stats,
            float(stats.get("eps", 1e-6)),
        )
        outputs.append(output)
    return outputs


def fit_apply_signal_quality_feature(
    frame_dfs: list[pd.DataFrame],
    *,
    mode: str,
    stats: dict | None = None,
) -> tuple[list[pd.DataFrame], dict | None]:
    """Fit in Python and apply the resulting feature in the native core."""
    if mode == "none":
        return frame_dfs, None
    fitted = stats or fit_signal_quality_stats(frame_dfs, mode=mode)
    return apply_signal_quality_stats(frame_dfs, stats=fitted), fitted


def _reduce_block(values: list[float], how: str) -> float:
    """Reduce raw-sample target values to one target value per 50 Hz frame."""
    arr = np.asarray(values, dtype=np.float32)
    if how == "max":
        return float(np.max(arr))
    if how == "last":
        return float(arr[-1])
    if how == "center":
        return float(arr[len(arr) // 2])
    if how == "mean":
        return float(np.mean(arr))
    raise ValueError(f"Unknown target_reduce: {how}")


def make_live_feature_dataframe(
    df: pd.DataFrame,
    ecg1_col: str = "ecg1_raw",
    ecg2_col: str = "ecg2_raw",
    ecg3_col: str | None = None,
    puls_col: str | None = "puls_raw",
    target_col: str | None = "target",
    time_col: str | None = "time_ms",
    time_unit: str = "ms",
    target_reduce: str = "max",
    sid_col: str = "sid",
    framer: LiveFeatureFramer | None = None,
    raw_idx_col: str = "raw_idx",
    frame_idx_col: str = "frame_idx",
    **framer_kwargs,
) -> pd.DataFrame:
    """Use Python DataFrame orchestration with the native streaming framer."""
    ecg1_col, ecg2_col, ecg3_col, puls_col = require_raw_signal_columns(
        df,
        ecg1_col=ecg1_col,
        ecg2_col=ecg2_col,
        ecg3_col=ecg3_col,
        puls_col=puls_col,
        include_puls=puls_col is not None,
    )
    if framer is None:
        framer = LiveFeatureFramer(
            include_puls=puls_col is not None,
            include_ecg3=ecg3_col is not None,
            **framer_kwargs,
        )

    rows: list[dict] = []
    grouped = df.groupby(sid_col, sort=False) if sid_col in df.columns else [(0, df)]
    for sid, group in grouped:
        framer.reset()
        times_ms = None
        if time_col is not None and time_col in group.columns:
            times_ms = group[time_col].to_numpy(dtype=float)
            if time_unit == "s":
                times_ms = times_ms * 1000.0
            elif time_unit != "ms":
                raise ValueError("time_unit must be 'ms' or 's'.")

        source_idx_col = raw_idx_col
        if source_idx_col not in group.columns and "idx" in group.columns:
            source_idx_col = "idx"
        sample_indices = (
            group[source_idx_col].to_numpy(dtype=np.int64)
            if source_idx_col in group.columns
            else np.arange(len(group), dtype=np.int64)
        )
        puls_values = (
            None if puls_col is None else group[puls_col].to_numpy(dtype=float)
        )
        ecg3_values = (
            None if ecg3_col is None else group[ecg3_col].to_numpy(dtype=float)
        )
        target_values = None
        if target_col is not None and target_col in group.columns:
            target_values = group[target_col].to_numpy(dtype=float)

        frames = framer.process_arrays(
            ecg1=group[ecg1_col].to_numpy(dtype=float),
            ecg2=group[ecg2_col].to_numpy(dtype=float),
            ecg3=ecg3_values,
            puls=puls_values,
            sample_indices=sample_indices,
            times_ms=times_ms,
        )
        positions_by_idx = {
            int(sample_idx): position
            for position, sample_idx in enumerate(sample_indices)
        }
        for frame in frames:
            row = {
                col: float(value)
                for col, value in zip(framer.feature_cols, frame.features)
            }
            row.update(frame.auxiliary_features)
            row["sid"] = sid
            row[frame_idx_col] = frame.frame_idx_50hz
            row["raw_idx"] = frame.start_sample_idx_400hz
            if frame.end_time_ms is not None:
                row["time_ms"] = frame.end_time_ms
            if target_values is not None:
                end_position = positions_by_idx[frame.end_sample_idx_400hz]
                start_position = max(0, end_position - framer.frame_size_raw + 1)
                row[target_col] = _reduce_block(
                    target_values[start_position : end_position + 1]
                    .astype(float)
                    .tolist(),
                    target_reduce,
                )
            rows.append(row)
    return pd.DataFrame(rows)


__all__ = [
    "RollingScalarQuantileNormalizer",
    "ECG_SLOPE_MODES",
    "LiveFeatureFrame",
    "LiveFeatureFramer",
    "PREPROCESSING_CONTRACT_VERSION",
    "RollingPairNormQuantileNormalizer",
    "SIGNAL_QUALITY_FEATURE_COL",
    "SIGNAL_QUALITY_MODES",
    "SignalQualityTracker",
    "apply_signal_quality_stats",
    "fit_apply_signal_quality_feature",
    "fit_signal_quality_stats",
    "live_feature_columns",
    "make_live_feature_dataframe",
    "make_live_framer",
    "signal_quality_component_columns",
    "signal_quality_feature_from_components",
]
