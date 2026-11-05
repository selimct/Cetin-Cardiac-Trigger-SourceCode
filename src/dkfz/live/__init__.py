# Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Developed at the German Cancer Research Center (DKFZ).

from dkfz.live.feature_framing import make_live_feature_dataframe, make_live_framer
from dkfz.live.feature_framing import (
    RollingScalarQuantileNormalizer,
    PREPROCESSING_CONTRACT_VERSION,
    LiveFeatureFrame,
    LiveFeatureFramer,
    RollingPairNormQuantileNormalizer,
    SIGNAL_QUALITY_FEATURE_COL,
    SIGNAL_QUALITY_MODES,
    feature_backend_name,
    fit_apply_signal_quality_feature,
    live_feature_columns,
)
from dkfz.live.io import load_json_config
from dkfz.live.prediction_logs import add_trigger_peak_flags
from dkfz.live.raw_bit_frames import (
    GpioEdgeConversionStats,
    convert_gpio_edges_to_parquet,
    default_gpio_edge_parquet_path,
)
from dkfz.live.streaming_predictor import LiveStreamingPredictor
from dkfz.live.triggers import (
    StreamingHoldoffLocalMaxTrigger,
    TriggerEvent,
    trigger_backend_name,
)
from dkfz.live.unframing import read_unframe_live_parquet, unframe_live_dataframe

__all__ = [
    "RollingScalarQuantileNormalizer",
    "GpioEdgeConversionStats",
    "LiveFeatureFrame",
    "LiveFeatureFramer",
    "LiveLSTMPredictor",
    "LiveStreamingPredictor",
    "LiveTensorRTPredictor",
    "PREPROCESSING_CONTRACT_VERSION",
    "RollingPairNormQuantileNormalizer",
    "SIGNAL_QUALITY_FEATURE_COL",
    "SIGNAL_QUALITY_MODES",
    "StreamingHoldoffLocalMaxTrigger",
    "TensorRTLogitEngine",
    "TriggerEvent",
    "add_trigger_peak_flags",
    "convert_gpio_edges_to_parquet",
    "default_gpio_edge_parquet_path",
    "default_torch_device",
    "fit_apply_signal_quality_feature",
    "feature_backend_name",
    "load_json_config",
    "live_feature_columns",
    "make_live_framer",
    "make_live_feature_dataframe",
    "read_unframe_live_parquet",
    "unframe_live_dataframe",
    "trigger_backend_name",
]

_LAZY_EXPORTS = {
    "LiveLSTMPredictor": ("dkfz.live.torch_streaming_predictor", "LiveLSTMPredictor"),
    "LiveTensorRTPredictor": (
        "dkfz.live.tensorrt_streaming_predictor",
        "LiveTensorRTPredictor",
    ),
    "TensorRTLogitEngine": ("dkfz.live.tensorrt_engine", "TensorRTLogitEngine"),
    "default_torch_device": ("dkfz.live.torch_streaming_predictor", "default_torch_device"),
}


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _LAZY_EXPORTS[name]
    from importlib import import_module

    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
