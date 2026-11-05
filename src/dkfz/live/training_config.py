# Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Developed at the German Cancer Research Center (DKFZ).

"""Shared configuration resolution for live training entrypoints."""

from __future__ import annotations

from typing import Any

from dkfz.live.io import load_json_config

TRAINING_DEFAULTS = {
    "inputs": ["data/live/p*/*"],
    "threshold_inputs": ["data/live/sp*/*"],
    "modelname": "target_r",
    "lr": 0.002884760125018128,
    "hidden_dim": 64,
    "num_layers": 2,
    "window_size": 50,
    "num_epochs": 35,
    "batch_size": 256,
    "seed": 42,
    "test_size": 0.2,
    "long_sequence_split_seconds": 720.0,
    "long_sequence_split_parts": 3,
    "target_col": "target",
    "ecg1_col": "ecg1_raw",
    "ecg2_col": "ecg2_raw",
    "ecg3_col": "ecg3_raw",
    "puls_col": "puls_raw",
    "not_use_puls": False,
    "time_col": "time_ms",
    "time_unit": "ms",
    "raw_hz": 400.0,
    "frame_size_raw": 8,
    "target_reduce": "max",
    "ecg_quantile": 0.999,
    "ecg_norm_window_samples": 4000,
    "ecg1_offset": 0.0,
    "ecg2_offset": 0.0,
    "ecg3_offset": 0.0,
    "ecg_slope_mode": "none",
    "ecg_slope_slow_window_samples": 40,
    "signal_quality_mode": "none",
    "signal_quality_stats": None,
    "ecg_rotation_angle": 0.0,
    "puls_offset": 0.0,
    "puls_quantile_low": 0.05,
    "puls_quantile_high": 0.95,
    "puls_norm_window_frames": 500,
    "puls_norm_warmup_value": 0.5,
    "puls_clip": True,
    "align_to_puls_blocks": True,
    "verify_puls_constant_within_block": False,
    "min_distance": 20,
    "error_association_frames": 1,
    "threshold_objective": "min_error",
    "manual_threshold": None,
    "threshold_sensitivity_efficiency": 0.95,
    "use_amp": True,
    "save_last": False,
    "save_prepped": True,
    "use_prepped": False,
    "prepped_file": None,
    "prep_workers": 8,
    "debug_timing": False,
}


def load_training_config_values(config: str | None) -> dict[str, Any]:
    """Load compatible training settings from a saved JSON configuration."""
    cfg = {
        key: value
        for key, value in load_json_config(config).items()
        if not key.startswith("_")
    }
    if "include_puls" in cfg and "not_use_puls" not in cfg:
        cfg["not_use_puls"] = not bool(cfg["include_puls"])
    # Absolute recall/sensitivity floors have different semantics and must not
    # be silently reinterpreted as an attainable-sensitivity efficiency.
    cfg.pop("threshold_min_exact_sensitivity", None)
    cfg.pop("threshold_min_recall", None)
    cfg.pop("temp_tolerance", None)
    return cfg


def training_kwargs_from_config_and_values(
    values: dict[str, Any],
    config_values: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge defaults, config values, and explicit entrypoint values.

    Precedence is explicit value, then config file, then the training default.
    """
    cfg = (
        load_training_config_values(values.get("config"))
        if config_values is None
        else config_values
    )

    kwargs = dict(TRAINING_DEFAULTS)
    kwargs.update(
        {
            key: cfg[key]
            for key in TRAINING_DEFAULTS
            if key in cfg and cfg[key] is not None
        }
    )

    for key in TRAINING_DEFAULTS:
        if values.get(key) is not None:
            kwargs[key] = values[key]

    inputs = values.get("inputs") or cfg.get("inputs") or TRAINING_DEFAULTS["inputs"]
    kwargs["inputs"] = inputs
    kwargs["config"] = values.get("config")
    return kwargs
