# Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Developed at the German Cancer Research Center (DKFZ).

"""Shared runtime config loading for Torch and TensorRT live predictors."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dkfz.live.feature_framing import make_live_framer
from dkfz.live.io import load_json_config
from dkfz.live.prediction_contract import (
    prediction_delay, PREDICTION_CONFIG_KEYS, supported_preprocessing_contract,
)
from dkfz.live.model_variants import (
    DEFAULT_MODEL_VARIANT,
    default_config_path,
    default_trt_engine_path as default_variant_trt_engine_path,
)

BASE_RUNTIME_DEFAULTS = {
    "modelname": "target_r",
    "model_variant": DEFAULT_MODEL_VARIANT,
    "ecg1_col": "ecg1_raw",
    "ecg2_col": "ecg2_raw",
    "ecg3_col": None,
    "ecg3_offset": 0.0,
    "ecg_slope_mode": "none",
    "ecg_slope_slow_window_samples": 40,
    "signal_quality_mode": "none",
    "signal_quality_stats": None,
    "puls_col": "puls_raw",
    "time_col": "time_ms",
    "sid_col": "sid",
    "raw_idx_col": "raw_idx",
}

COMMON_RUNTIME_REQUIRED_KEYS = {
    "input_dim",
    "window_size",
    "not_use_puls",
    "raw_hz",
    "frame_size_raw",
    "ecg_quantile",
    "ecg_norm_window_samples",
    "ecg1_offset",
    "ecg2_offset",
    "ecg_rotation_angle",
    "puls_offset",
    "puls_quantile_low",
    "puls_quantile_high",
    "puls_norm_window_frames",
    "puls_norm_warmup_value",
    "puls_clip",
    "align_to_puls_blocks",
    "threshold",
}

TORCH_RUNTIME_REQUIRED_KEYS = COMMON_RUNTIME_REQUIRED_KEYS | {
    "hidden_dim",
    "num_layers",
}

COMMON_RUNTIME_OPTIONAL_KEYS = PREDICTION_CONFIG_KEYS | {
    "recovery_head",
    "recovery_enabled",
    "recovery_threshold",
    "feature_cols",
    "holdoff_frames",
    "holdoff_ms",
    "verify_puls_constant_within_block",
    "ecg3_offset",
    "ecg_slope_mode",
    "ecg_slope_slow_window_samples",
    "signal_quality_mode",
    "signal_quality_stats",
    "preprocessing_contract_version",
}

RUNTIME_COLUMN_KEYS = (
    "ecg1_col",
    "ecg2_col",
    "ecg3_col",
    "puls_col",
    "time_col",
    "sid_col",
    "raw_idx_col",
)


def effective_recovery_threshold(settings: dict, *, disable_recovery: bool = False):
    """Disable recovery events while preserving the two-output architecture."""
    if disable_recovery or not settings.get("recovery_enabled", True):
        return None
    return settings.get("recovery_threshold")


def default_train_config_path(
    modelname: str,
    model_variant: str | None = None,
) -> Path:
    return default_config_path(modelname, model_variant)


def default_trt_engine_path(
    modelname: str,
    model_variant: str | None = None,
) -> Path:
    return default_variant_trt_engine_path(modelname, model_variant)


def load_runtime_settings(
    values: dict,
    *,
    defaults: dict,
    required_keys: set[str],
    optional_keys: set[str] | None = None,
    error_label: str,
) -> dict:
    """Load a runtime config contract and merge CLI/function overrides."""
    optional_keys = optional_keys or set()
    modelname = values.get("modelname") or defaults["modelname"]
    model_variant = (
        values.get("model_variant")
        or defaults.get("model_variant")
        or DEFAULT_MODEL_VARIANT
    )
    cfg_path = Path(
        values.get("config") or default_train_config_path(modelname, model_variant)
    )
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    cfg = {
        key: value
        for key, value in load_json_config(cfg_path).items()
        if not key.startswith("_")
    }
    if "include_puls" in cfg and "not_use_puls" not in cfg:
        cfg["not_use_puls"] = not bool(cfg["include_puls"])

    missing = sorted(key for key in required_keys if key not in cfg)
    if missing:
        joined = ", ".join(missing)
        raise ValueError(
            f"Config file {cfg_path} is missing required {error_label} key(s): "
            f"{joined}"
        )

    settings = dict(defaults)
    config_keys = required_keys | optional_keys
    settings.update({key: cfg[key] for key in config_keys if key in cfg})
    if "modelname" in cfg and values.get("modelname") is None:
        settings["modelname"] = cfg["modelname"]
    for key in RUNTIME_COLUMN_KEYS:
        if key in cfg and cfg[key] is not None:
            settings[key] = cfg[key]

    for key in defaults:
        if values.get(key) is not None:
            settings[key] = values[key]

    settings["config"] = str(cfg_path)
    settings["recovery_threshold"] = effective_recovery_threshold(settings)
    settings["verify_puls_constant_within_block"] = bool(
        settings.get("verify_puls_constant_within_block", False)
    )
    supported_preprocessing_contract(settings)
    prediction_delay(settings)
    return settings


def holdoff_frames_from_settings(settings: dict) -> int:
    """Return holdoff frames using explicit frames or millisecond fallback."""
    holdoff_frames = settings.get("holdoff_frames")
    if holdoff_frames is not None:
        return int(holdoff_frames)

    raw_hz = float(settings["raw_hz"])
    frame_size_raw = int(settings["frame_size_raw"])
    frame_hz = raw_hz / frame_size_raw
    holdoff_ms = float(settings.get("holdoff_ms", 200.0))
    return int(np.ceil(holdoff_ms * frame_hz / 1000.0))


def make_checked_runtime_framer(settings: dict):
    """Build the shared live framer and validate it against input_dim."""
    supported_preprocessing_contract(settings)
    delay = prediction_delay(settings)
    raw_hz = float(settings["raw_hz"])
    frame_size_raw = int(settings["frame_size_raw"])
    include_puls = not bool(settings["not_use_puls"])
    signal_quality_mode = str(settings.get("signal_quality_mode", "none"))
    signal_quality_stats = settings.get("signal_quality_stats")
    if signal_quality_mode != "none" and signal_quality_stats is None:
        raise ValueError(
            "signal_quality_mode is enabled but signal_quality_stats are missing "
            "from the runtime config."
        )

    framer = make_live_framer(
        reset_on_discontinuity=bool(delay),
        signal_quality_puls_flat_window_frames=int(
            settings.get("signal_quality_puls_flat_window_frames", 10)
        ),
        signal_quality_puls_median_window_frames=int(
            settings.get("signal_quality_puls_median_window_frames", 200)
        ),
        raw_hz=raw_hz,
        frame_size_raw=frame_size_raw,
        include_puls=include_puls,
        ecg_quantile=float(settings["ecg_quantile"]),
        ecg_norm_window_samples=int(settings["ecg_norm_window_samples"]),
        ecg1_offset=float(settings["ecg1_offset"]),
        ecg2_offset=float(settings["ecg2_offset"]),
        ecg3_offset=float(settings.get("ecg3_offset", 0.0)),
        include_ecg3=settings.get("ecg3_col") is not None,
        ecg_slope_mode=str(settings.get("ecg_slope_mode", "none")),
        ecg_slope_slow_window_samples=int(
            settings.get("ecg_slope_slow_window_samples", 40)
        ),
        signal_quality_mode=signal_quality_mode,
        signal_quality_stats=signal_quality_stats,
        ecg_rotation_angle=float(settings["ecg_rotation_angle"]),
        puls_offset=float(settings["puls_offset"]),
        puls_quantile_low=float(settings["puls_quantile_low"]),
        puls_quantile_high=float(settings["puls_quantile_high"]),
        puls_norm_window_frames=int(settings["puls_norm_window_frames"]),
        puls_norm_warmup_value=float(settings["puls_norm_warmup_value"]),
        puls_clip=bool(settings["puls_clip"]),
        align_to_puls_blocks=bool(settings["align_to_puls_blocks"]),
        verify_puls_constant_within_block=bool(
            settings["verify_puls_constant_within_block"]
        ),
    )

    input_dim = int(settings["input_dim"])
    feature_dim = len(framer.feature_cols)
    if feature_dim != input_dim:
        raise ValueError(
            f"Configured input_dim={input_dim} does not match live framer feature "
            f"dimension {feature_dim}."
        )
    expected_feature_cols = settings.get("feature_cols")
    if expected_feature_cols is not None:
        expected_feature_cols = list(expected_feature_cols)
        if framer.feature_cols != expected_feature_cols:
            raise ValueError(
                "Configured feature_cols do not match live framer feature order. "
                f"Expected {expected_feature_cols}, got {framer.feature_cols}."
            )
    return framer
