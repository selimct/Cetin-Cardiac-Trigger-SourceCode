# Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Developed at the German Cancer Research Center (DKFZ).

"""Export the stable key/value contract consumed by the C++ live runtime."""

from __future__ import annotations

import math
import os
import zlib
from pathlib import Path
from typing import Any

from dkfz.live.feature_framing import live_feature_columns
from dkfz.live.model_variants import DEFAULT_MODEL_VARIANT, get_model_variant
from dkfz.live.runtime_config import effective_recovery_threshold
from dkfz.live.prediction_contract import prediction_delay, supported_preprocessing_contract

DEPLOYMENT_MANIFEST_SCHEMA = "dkfz-live-deployment-v1"
DELAYED_DEPLOYMENT_MANIFEST_SCHEMA = "dkfz-live-deployment-v2"


def default_native_manifest_path(
    modelname: str,
    model_variant: str | None = None,
    model_dir: str | Path | None = None,
) -> Path:
    """Return the default native manifest path for a trained variant."""
    variant = get_model_variant(model_variant)
    artifact_root = variant.model_dir(modelname, model_dir)
    return artifact_root / f"{modelname}_{variant.name}_native.conf"


def _require(cfg: dict[str, Any], key: str) -> Any:
    if key not in cfg or cfg[key] is None:
        raise ValueError(f"Training config is missing required native key: {key}")
    return cfg[key]


def _bool_text(value: Any) -> str:
    return "true" if bool(value) else "false"


def _number_text(value: Any, key: str) -> str:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{key} must be finite, got {value!r}.")
    return format(number, ".17g")


def _integer_text(value: Any, key: str) -> str:
    integer = int(value)
    if float(value) != integer:
        raise ValueError(f"{key} must be an integer, got {value!r}.")
    return str(integer)


def _safe_scalar(value: Any, key: str) -> str:
    text = str(value)
    if any(character in text for character in ("\n", "\r", "#")):
        raise ValueError(f"{key} contains a character unsupported by the manifest.")
    return text


def _csv(values: Any, key: str) -> str:
    items = [_safe_scalar(value, key) for value in values]
    if any("," in item or not item for item in items):
        raise ValueError(f"{key} values must be non-empty and may not contain commas.")
    return ",".join(items)


def _include_puls(cfg: dict[str, Any]) -> bool:
    if "not_use_puls" in cfg:
        return not bool(cfg["not_use_puls"])
    return bool(_require(cfg, "include_puls"))


def _holdoff_frames(cfg: dict[str, Any]) -> int:
    if cfg.get("holdoff_frames") is not None:
        value = int(cfg["holdoff_frames"])
    else:
        raw_hz = float(_require(cfg, "raw_hz"))
        frame_size = int(_require(cfg, "frame_size_raw"))
        holdoff_ms = float(cfg.get("holdoff_ms", 200.0))
        value = int(math.ceil(holdoff_ms * (raw_hz / frame_size) / 1000.0))
    if value < 0:
        raise ValueError("holdoff_frames may not be negative.")
    return value


def native_manifest_values(
    cfg: dict[str, Any],
    *,
    engine_path: str | Path,
    model_variant: str = DEFAULT_MODEL_VARIANT,
    source_model_dir: str | Path | None = None,
    engine_size_bytes: int | None = None,
    engine_crc32: str | None = None,
) -> dict[str, str]:
    """Translate one training config to validated native manifest values."""
    contract = supported_preprocessing_contract(cfg)

    include_puls = _include_puls(cfg)
    include_ecg3 = cfg.get("ecg3_col") is not None
    slope_mode = str(cfg.get("ecg_slope_mode", "none"))
    quality_mode = str(cfg.get("signal_quality_mode", "none"))
    ecg_cols = ("ecg1", "ecg2", "ecg3") if include_ecg3 else ("ecg1", "ecg2")
    expected_features = live_feature_columns(
        frame_size_raw=int(_require(cfg, "frame_size_raw")),
        include_puls=include_puls,
        ecg_cols=ecg_cols,
        ecg_slope_mode=slope_mode,
        signal_quality_mode=quality_mode,
    )
    configured_features = list(_require(cfg, "feature_cols"))
    if configured_features != expected_features:
        raise ValueError(
            "feature_cols do not match the native preprocessing contract. "
            f"Expected {expected_features}, got {configured_features}."
        )
    input_dim = int(_require(cfg, "input_dim"))
    if input_dim != len(expected_features):
        raise ValueError(
            f"input_dim={input_dim} does not match feature count "
            f"{len(expected_features)}."
        )

    quality_stats = cfg.get("signal_quality_stats")
    if quality_mode != "none" and not isinstance(quality_stats, dict):
        raise ValueError(
            "signal_quality_mode is enabled but signal_quality_stats are missing."
        )

    delay = prediction_delay(cfg)
    recovery_threshold = effective_recovery_threshold(cfg)
    values = {
        "prediction_mode": cfg.get("prediction_mode", "current"),
        "prediction_delay_frames": str(delay),
        "signal_quality_puls_flat_window_frames": str(
            int(cfg.get("signal_quality_puls_flat_window_frames", 10))
        ),
        "signal_quality_puls_median_window_frames": str(
            int(cfg.get("signal_quality_puls_median_window_frames", 200))
        ),
        "recovery_head": _bool_text(cfg.get("recovery_head", False)),
        "recovery_threshold": (
            ""
            if recovery_threshold is None
            else _number_text(recovery_threshold, "recovery_threshold")
        ),
        "schema": (
            DELAYED_DEPLOYMENT_MANIFEST_SCHEMA
            if (
                delay
                or int(cfg.get("signal_quality_puls_flat_window_frames", 10)) != 10
                or int(cfg.get("signal_quality_puls_median_window_frames", 200)) != 200
            )
            else DEPLOYMENT_MANIFEST_SCHEMA
        ),
        "preprocessing_contract_version": str(contract),
        "model_name": _safe_scalar(cfg.get("modelname") or "target_r", "model_name"),
        "model_variant": _safe_scalar(model_variant, "model_variant"),
        "engine_path": _safe_scalar(engine_path, "engine_path"),
        "input_tensor": _safe_scalar(cfg.get("input_tensor") or "x", "input_tensor"),
        "output_tensor": _safe_scalar(
            cfg.get("output_tensor") or "logit", "output_tensor"
        ),
        "window_size": _integer_text(_require(cfg, "window_size"), "window_size"),
        "input_dim": str(input_dim),
        "threshold": _number_text(_require(cfg, "threshold"), "threshold"),
        "manual_threshold": (
            ""
            if cfg.get("manual_threshold") is None
            else _number_text(cfg["manual_threshold"], "manual_threshold")
        ),
        "holdoff_frames": str(_holdoff_frames(cfg)),
        "model_output_is_probability": _bool_text(
            cfg.get("model_output_is_probability", False)
        ),
        "raw_hz": _number_text(_require(cfg, "raw_hz"), "raw_hz"),
        "frame_size_raw": _integer_text(
            _require(cfg, "frame_size_raw"), "frame_size_raw"
        ),
        "include_puls": _bool_text(include_puls),
        "include_ecg3": _bool_text(include_ecg3),
        "ecg1_offset": _number_text(_require(cfg, "ecg1_offset"), "ecg1_offset"),
        "ecg2_offset": _number_text(_require(cfg, "ecg2_offset"), "ecg2_offset"),
        "ecg3_offset": _number_text(cfg.get("ecg3_offset", 0.0), "ecg3_offset"),
        "ecg_slope_mode": _safe_scalar(slope_mode, "ecg_slope_mode"),
        "ecg_slope_slow_window_samples": _integer_text(
            cfg.get("ecg_slope_slow_window_samples", 40),
            "ecg_slope_slow_window_samples",
        ),
        "signal_quality_mode": _safe_scalar(quality_mode, "signal_quality_mode"),
        "signal_quality_stats_present": _bool_text(isinstance(quality_stats, dict)),
        "ecg_rotation_angle": _number_text(
            _require(cfg, "ecg_rotation_angle"), "ecg_rotation_angle"
        ),
        "puls_offset": _number_text(_require(cfg, "puls_offset"), "puls_offset"),
        "ecg_quantile": _number_text(_require(cfg, "ecg_quantile"), "ecg_quantile"),
        "ecg_norm_window_samples": _integer_text(
            _require(cfg, "ecg_norm_window_samples"), "ecg_norm_window_samples"
        ),
        "ecg_norm_eps": _number_text(cfg.get("ecg_norm_eps", 1e-8), "ecg_norm_eps"),
        "ecg_norm_warmup_scale": _number_text(
            cfg.get("ecg_norm_warmup_scale", 1.0), "ecg_norm_warmup_scale"
        ),
        "puls_quantile_low": _number_text(
            _require(cfg, "puls_quantile_low"), "puls_quantile_low"
        ),
        "puls_quantile_high": _number_text(
            _require(cfg, "puls_quantile_high"), "puls_quantile_high"
        ),
        "puls_norm_window_frames": _integer_text(
            _require(cfg, "puls_norm_window_frames"),
            "puls_norm_window_frames",
        ),
        "puls_norm_eps": _number_text(cfg.get("puls_norm_eps", 1e-8), "puls_norm_eps"),
        "puls_norm_warmup_value": _number_text(
            _require(cfg, "puls_norm_warmup_value"),
            "puls_norm_warmup_value",
        ),
        "puls_clip": _bool_text(_require(cfg, "puls_clip")),
        "align_to_puls_blocks": _bool_text(_require(cfg, "align_to_puls_blocks")),
        "verify_puls_constant_within_block": _bool_text(
            cfg.get("verify_puls_constant_within_block", False)
        ),
        "puls_constant_tol": _number_text(
            cfg.get("puls_constant_tol", 1e-12), "puls_constant_tol"
        ),
        "feature_columns": _csv(configured_features, "feature_cols"),
        "ecg1_field": _safe_scalar(cfg.get("ecg1_col") or "ecg1_raw", "ecg1_field"),
        "ecg2_field": _safe_scalar(cfg.get("ecg2_col") or "ecg2_raw", "ecg2_field"),
        "ecg3_field": (
            "" if not include_ecg3 else _safe_scalar(cfg["ecg3_col"], "ecg3_field")
        ),
        "puls_field": _safe_scalar(cfg.get("puls_col") or "puls_raw", "puls_field"),
        "time_field": _safe_scalar(cfg.get("time_col") or "time_ms", "time_field"),
        "sample_index_field": _safe_scalar(
            cfg.get("raw_idx_col") or "raw_idx", "sample_index_field"
        ),
        "sid_field": _safe_scalar(cfg.get("sid_col") or "sid", "sid_field"),
    }
    for key in ("fp_threshold", "der_threshold"):
        value = cfg.get(key)
        if value is not None and not (
            math.isfinite(float(value)) and 0 <= float(value) <= 1
        ):
            raise ValueError(f"{key} must be null or in [0, 1].")
        values[key] = "" if value is None else _number_text(value, key)
    if (
        cfg.get("manual_threshold") is not None
        and not 0 <= float(cfg["manual_threshold"]) <= 1
    ):
        raise ValueError("manual_threshold must be null or in [0, 1].")
    if source_model_dir is not None:
        values["source_model_dir"] = _safe_scalar(
            Path(source_model_dir).as_posix(), "source_model_dir"
        )
    recovery = cfg.get("recovery_threshold")
    if recovery is not None and (
        not cfg.get("recovery_head") or not 0 < float(recovery) <= 1
    ):
        raise ValueError(
            "recovery_threshold requires recovery_head and a value in (0, 1]"
        )
    if cfg.get("recovery_head") and int(cfg["window_size"]) < 2:
        raise ValueError("recovery_head requires window_size >= 2")
    if engine_size_bytes is not None or engine_crc32 is not None:
        if engine_size_bytes is None or engine_crc32 is None:
            raise ValueError(
                "engine_size_bytes and engine_crc32 must be provided together."
            )
        engine_size_bytes = int(engine_size_bytes)
        engine_crc32 = str(engine_crc32)
        if engine_size_bytes < 0:
            raise ValueError("engine_size_bytes may not be negative.")
        if len(engine_crc32) != 8 or any(
            character not in "0123456789abcdef" for character in engine_crc32
        ):
            raise ValueError("engine_crc32 must contain 8 lowercase hex digits.")
        values["engine_size_bytes"] = str(engine_size_bytes)
        values["engine_crc32"] = engine_crc32

    if isinstance(quality_stats, dict):
        values.update(
            {
                "signal_quality_transform": _safe_scalar(
                    quality_stats.get("transform", ""), "signal_quality_transform"
                ),
                "signal_quality_ecg_component_cols": _csv(
                    quality_stats.get("ecg_component_cols", []),
                    "signal_quality_ecg_component_cols",
                ),
                "signal_quality_puls_component_cols": _csv(
                    quality_stats.get("puls_component_cols", []),
                    "signal_quality_puls_component_cols",
                ),
                "signal_quality_ecg_weight": _number_text(
                    quality_stats.get("ecg_weight", 1.0),
                    "signal_quality_ecg_weight",
                ),
                "signal_quality_puls_weight": _number_text(
                    quality_stats.get("puls_weight", 0.0),
                    "signal_quality_puls_weight",
                ),
                "signal_quality_eps": _number_text(
                    quality_stats.get("eps", 1e-6), "signal_quality_eps"
                ),
            }
        )
        if quality_stats.get("z_clip") is not None:
            values["signal_quality_z_clip"] = _number_text(
                quality_stats["z_clip"], "signal_quality_z_clip"
            )
        for group in ("median", "iqr", "min_iqr"):
            component_values = quality_stats.get(group)
            if not isinstance(component_values, dict):
                raise ValueError(f"signal_quality_stats.{group} must be a mapping.")
            for component, value in component_values.items():
                component = _safe_scalar(component, f"signal_quality_{group}")
                values[f"signal_quality_{group}.{component}"] = _number_text(
                    value, f"signal_quality_{group}.{component}"
                )
    return values


def fingerprint_engine(path: str | Path) -> tuple[int, str]:
    """Return the size and streaming CRC32 fingerprint of an engine."""
    size = 0
    fingerprint = 0
    with Path(path).open("rb") as source:
        while chunk := source.read(1024 * 1024):
            size += len(chunk)
            fingerprint = zlib.crc32(chunk, fingerprint)
    return size, f"{fingerprint & 0xFFFFFFFF:08x}"


def write_native_deployment_manifest(
    cfg: dict[str, Any],
    output_path: str | Path,
    *,
    engine_path: str | Path,
    model_variant: str = DEFAULT_MODEL_VARIANT,
    source_model_dir: str | Path | None = None,
) -> Path:
    """Write an atomic native deployment manifest next to model artifacts."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_parent = output_path.parent.resolve()
    engine_path = Path(engine_path)
    if not engine_path.is_absolute():
        engine_path = Path.cwd() / engine_path
    engine_size_bytes = None
    engine_crc32 = None
    if engine_path.is_file():
        engine_size_bytes, engine_crc32 = fingerprint_engine(engine_path)
    relative_engine = os.path.relpath(engine_path.resolve(), output_parent)
    values = native_manifest_values(
        cfg,
        engine_path=Path(relative_engine).as_posix(),
        model_variant=model_variant,
        source_model_dir=source_model_dir,
        engine_size_bytes=engine_size_bytes,
        engine_crc32=engine_crc32,
    )
    contents = (
        "# Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>\n"
        "# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0\n"
        "# Developed at the German Cancer Research Center (DKFZ).\n"
        "\n"
        "# Generated by dkfz.live.deployment_manifest; do not hand-mix models.\n"
    )
    contents += "".join(f"{key}={value}\n" for key, value in values.items())
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(contents, encoding="utf-8")
    temporary.replace(output_path)
    return output_path
