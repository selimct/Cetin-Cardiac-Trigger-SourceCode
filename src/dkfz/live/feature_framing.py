# Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Developed at the German Cancer Research Center (DKFZ).

"""Stable Python API for the canonical native live feature implementation."""

from __future__ import annotations

import os

_BACKEND_ENV = "DKFZ_FEATURE_BACKEND"
_REQUESTED_BACKEND = os.environ.get(_BACKEND_ENV, "native").strip().lower()
if _REQUESTED_BACKEND == "python":
    raise RuntimeError(
        "DKFZ_FEATURE_BACKEND=python is no longer supported; the canonical "
        "C++ live feature implementation is required."
    )
if _REQUESTED_BACKEND not in {"native", "auto"}:
    raise ValueError(
        f"{_BACKEND_ENV} must be native or auto; got {_REQUESTED_BACKEND!r}."
    )

try:
    from dkfz.native import feature_framing as _native
except ImportError as exc:
    raise ImportError(
        "The compiled dkfz.native._live_core extension is required. Reinstall "
        "the project to build its native extension."
    ) from exc


FEATURE_BACKEND = "native"
PREPROCESSING_CONTRACT_VERSION = _native.PREPROCESSING_CONTRACT_VERSION


def feature_backend_name() -> str:
    """Return the sole supported feature-framing backend name."""
    return FEATURE_BACKEND


ECG_SLOPE_MODES = _native.ECG_SLOPE_MODES
SIGNAL_QUALITY_MODES = _native.SIGNAL_QUALITY_MODES
SIGNAL_QUALITY_FEATURE_COL = _native.SIGNAL_QUALITY_FEATURE_COL
LiveFeatureFrame = _native.LiveFeatureFrame
RollingScalarQuantileNormalizer = _native.RollingScalarQuantileNormalizer
LiveFeatureFramer = _native.LiveFeatureFramer
RollingPairNormQuantileNormalizer = _native.RollingPairNormQuantileNormalizer
SignalQualityTracker = _native.SignalQualityTracker
apply_signal_quality_stats = _native.apply_signal_quality_stats
fit_apply_signal_quality_feature = _native.fit_apply_signal_quality_feature
fit_signal_quality_stats = _native.fit_signal_quality_stats
live_feature_columns = _native.live_feature_columns
make_live_feature_dataframe = _native.make_live_feature_dataframe
make_live_framer = _native.make_live_framer
signal_quality_component_columns = _native.signal_quality_component_columns
signal_quality_feature_from_components = _native.signal_quality_feature_from_components


__all__ = [
    "RollingScalarQuantileNormalizer",
    "ECG_SLOPE_MODES",
    "FEATURE_BACKEND",
    "LiveFeatureFrame",
    "LiveFeatureFramer",
    "PREPROCESSING_CONTRACT_VERSION",
    "RollingPairNormQuantileNormalizer",
    "SIGNAL_QUALITY_FEATURE_COL",
    "SIGNAL_QUALITY_MODES",
    "SignalQualityTracker",
    "apply_signal_quality_stats",
    "feature_backend_name",
    "fit_apply_signal_quality_feature",
    "fit_signal_quality_stats",
    "live_feature_columns",
    "make_live_feature_dataframe",
    "make_live_framer",
    "signal_quality_component_columns",
    "signal_quality_feature_from_components",
]
