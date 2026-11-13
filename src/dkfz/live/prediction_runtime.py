# Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Developed at the German Cancer Research Center (DKFZ).

"""Shared helpers for parquet prediction entrypoints."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from dkfz.live.io import load_json_config, repo_relative_path
from dkfz.live.io import (
    optional_raw_signal_column,
    require_raw_signal_columns,
    resolve_input_paths,
)
from dkfz.live.runtime_config import make_checked_runtime_framer
from dkfz.live.triggers import TriggerEvent


@dataclass
class RuntimeConfigResolution:
    values: dict[str, Any]
    defaults: dict[str, Any]
    explicit_values: dict[str, Any]
    config_values: dict[str, Any]
    derived_values: dict[str, Any]
    title: str = "Resolved prediction configuration"


def safe_output_stem(in_file: str) -> str:
    """Return a filesystem-safe stem for input-specific output files."""
    stem = Path(in_file).stem
    safe = "".join(
        char if char.isalnum() or char in ("-", "_", ".") else "_" for char in stem
    )
    return safe.strip("._") or "input"


def default_prediction_metrics_path(model_dir: Path, out_log: str | Path) -> str:
    """Return the default text metrics path for a prediction log."""
    out_log_path = Path(out_log)
    stem = out_log_path.stem
    if not stem.startswith("prediction_"):
        stem = f"prediction_{stem}"
    return str(model_dir / "metrics" / f"{stem}_metrics.txt")


def load_config_contract(
    cfg_path: str | Path,
    *,
    required_keys: set[str],
    error_label: str,
) -> dict[str, Any]:
    """Load a train config and validate the keys needed by a runtime."""
    cfg_path = Path(cfg_path)
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    cfg = load_json_config(cfg_path)
    cfg = {key: value for key, value in cfg.items() if not key.startswith("_")}
    if "include_puls" in cfg and "not_use_puls" not in cfg:
        cfg["not_use_puls"] = not bool(cfg["include_puls"])

    missing = sorted(key for key in required_keys if key not in cfg)
    if missing:
        joined = ", ".join(missing)
        raise ValueError(
            f"Config file {cfg_path} is missing required {error_label} key(s): "
            f"{joined}"
        )
    return cfg


def resolve_config_and_runtime_values(
    *,
    cfg: dict[str, Any],
    values: dict[str, Any],
    defaults: dict[str, Any],
    config_keys: set[str],
    column_keys: tuple[str, ...],
) -> RuntimeConfigResolution:
    """Resolve config/default/runtime values and keep provenance for logging."""
    kwargs = dict(defaults)
    kwargs.update({key: cfg[key] for key in config_keys if key in cfg})
    if "modelname" in cfg and values.get("modelname") is None:
        kwargs["modelname"] = cfg["modelname"]
    for key in column_keys:
        if key in cfg and cfg[key] is not None:
            kwargs[key] = cfg[key]

    for key in defaults:
        if values.get(key) is not None:
            kwargs[key] = values[key]

    return RuntimeConfigResolution(
        values=kwargs,
        defaults=defaults,
        explicit_values=values,
        config_values=cfg,
        derived_values={},
    )


_PREDICTION_CONFIG_SECTIONS = (
    (
        "Paths",
        (
            "in_file",
            "model_dir",
            "config",
            "model_path",
            "engine",
            "out_events",
            "out_log",
            "out_metrics",
        ),
    ),
    (
        "Model",
        (
            "modelname",
            "model_variant",
            "input_dim",
            "hidden_dim",
            "num_layers",
            "window_size",
            "include_puls",
        ),
    ),
    (
        "Runtime",
        (
            "prediction_mode",
            "prediction_delay_frames",
            "exact_tolerance_frames",
            "error_association_frames",
            "outer_error_association_frames",
            "threshold",
            "holdoff_frames",
            "holdoff_ms",
            "skip_metrics",
            "include_input_name_in_outputs",
        ),
    ),
    (
        "Preprocessing",
        (
            "raw_hz",
            "frame_size_raw",
            "ecg_quantile",
            "ecg_norm_window_samples",
            "ecg_slope_mode",
            "ecg_slope_slow_window_samples",
            "signal_quality_mode",
            "ecg_rotation_angle",
            "puls_quantile_low",
            "puls_quantile_high",
            "puls_norm_window_frames",
            "puls_norm_warmup_value",
            "puls_clip",
            "align_to_puls_blocks",
            "verify_puls_constant_within_block",
        ),
    ),
    (
        "Columns",
        (
            "target_col",
            "ecg1_col",
            "ecg2_col",
            "ecg3_col",
            "puls_col",
            "time_col",
            "time_unit",
        ),
    ),
)

_PREDICTION_PATH_KEYS = {
    "in_file",
    "model_dir",
    "config",
    "model_path",
    "engine",
    "out_events",
    "out_log",
    "out_metrics",
}


def _short_value(value: Any, *, is_path: bool = False) -> str:
    if is_path:
        if isinstance(value, (list, tuple)):
            value = [repo_relative_path(item) for item in value]
        elif value is not None:
            value = repo_relative_path(value)
    if isinstance(value, Path):
        return repr(repo_relative_path(value))
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list):
        if len(value) <= 3:
            return repr(value)
        return f"{len(value)} entries, first={value[0]!r}"
    return repr(value)


def _explicit_value_provided(key: str, values: dict[str, Any]) -> bool:
    return key in values and values[key] is not None


def _prediction_config_source(
    key: str,
    *,
    defaults: dict[str, Any],
    explicit_values: dict[str, Any],
    config_values: dict[str, Any],
    derived_values: dict[str, Any],
) -> str:
    if key in derived_values:
        return "derived"
    if _explicit_value_provided(key, explicit_values):
        return "explicit"
    if key in config_values and config_values[key] is not None:
        return "config"
    if key in defaults:
        return "default"
    return ""


def format_resolved_prediction_configuration(
    resolution: RuntimeConfigResolution,
) -> str:
    """Format effective prediction settings and where they came from."""
    all_values = dict(resolution.values)
    all_values.update(resolution.derived_values)
    line_keys = [
        key
        for _, keys in _PREDICTION_CONFIG_SECTIONS
        for key in keys
        if key in all_values
    ]
    key_width = max((len(key) for key in line_keys), default=0)
    lines = [resolution.title]

    for section, keys in _PREDICTION_CONFIG_SECTIONS:
        section_lines = []
        for key in keys:
            if key not in all_values:
                continue

            source = _prediction_config_source(
                key,
                defaults=resolution.defaults,
                explicit_values=resolution.explicit_values,
                config_values=resolution.config_values,
                derived_values=resolution.derived_values,
            )
            details = []
            if (
                key in resolution.config_values
                and resolution.config_values[key] is not None
            ):
                config_value = _short_value(
                    resolution.config_values[key],
                    is_path=key in _PREDICTION_PATH_KEYS,
                )
                details.append(f"config: {config_value}")
            if key in resolution.defaults:
                default_value = _short_value(
                    resolution.defaults[key],
                    is_path=key in _PREDICTION_PATH_KEYS,
                )
                details.append(f"default: {default_value}")
            if source in {"explicit", "config"}:
                details.append(f"source: {source}")
            elif source == "derived":
                details.append("derived")

            suffix = f"  ({', '.join(details)})" if details else ""
            section_lines.append(
                f"  {key:<{key_width}} = "
                f"{_short_value(all_values[key], is_path=key in _PREDICTION_PATH_KEYS)}"
                f"{suffix}"
            )

        if section_lines:
            lines.append("")
            lines.append(section)
            lines.extend(section_lines)

    return "\n".join(lines)


def print_resolved_prediction_configuration(
    resolution: RuntimeConfigResolution,
) -> None:
    """Print the effective prediction settings in a log-friendly format."""
    print(format_resolved_prediction_configuration(resolution))


def input_output_prefix(values: dict[str, Any], kwargs: dict[str, Any]) -> str:
    if kwargs["include_input_name_in_outputs"]:
        return f"_{safe_output_stem(values['in_file'])}"
    return ""


def resolve_prediction_inputs(
    inputs: list[str],
    *,
    out_events: str | Path | None,
    out_log: str | Path | None,
    out_metrics: str | Path | None,
) -> tuple[list[Path], bool]:
    """Resolve prediction inputs and report whether batch-safe outputs are needed."""
    input_paths = resolve_input_paths(inputs)
    multiple_inputs = len(input_paths) > 1
    if multiple_inputs:
        explicit_outputs = [
            name
            for name, value in (
                ("--out-events", out_events),
                ("--out-log", out_log),
                ("--out-metrics", out_metrics),
            )
            if value is not None
        ]
        if explicit_outputs:
            joined = ", ".join(explicit_outputs)
            raise ValueError(f"{joined} can only be used with one resolved input file.")
    return input_paths, multiple_inputs


def make_framer_from_settings(settings: dict[str, Any]):
    """Construct and dimension-check a live framer from config settings."""
    return make_checked_runtime_framer(settings)


def resolve_prediction_columns(
    df: pd.DataFrame,
    *,
    ecg1_col: str,
    ecg2_col: str,
    ecg3_col: str | None = None,
    puls_col: str,
    include_puls: bool,
    optional_puls: bool = False,
) -> tuple[str, str, str | None, str | None]:
    actual_ecg1_col, actual_ecg2_col, actual_ecg3_col, actual_puls_col = (
        require_raw_signal_columns(
            df,
            ecg1_col=ecg1_col,
            ecg2_col=ecg2_col,
            ecg3_col=ecg3_col,
            puls_col=puls_col,
            include_puls=include_puls,
        )
    )
    if not include_puls and optional_puls:
        actual_puls_col = optional_raw_signal_column(df, puls_col, "pulse")
    return actual_ecg1_col, actual_ecg2_col, actual_ecg3_col, actual_puls_col


def events_to_dataframe(events: list[TriggerEvent]) -> pd.DataFrame:
    return pd.DataFrame(
        [asdict(event) for event in events],
        columns=list(TriggerEvent.__annotations__.keys()),
    )


def write_prediction_parquets(
    *,
    events_df: pd.DataFrame,
    log_df: pd.DataFrame,
    out_events: str | Path,
    out_log: str | Path,
) -> None:
    Path(out_events).parent.mkdir(parents=True, exist_ok=True)
    Path(out_log).parent.mkdir(parents=True, exist_ok=True)
    events_df.to_parquet(out_events, compression="snappy")
    log_df.to_parquet(out_log, compression="snappy")
