# Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Developed at the German Cancer Research Center (DKFZ).

"""Shared pieces for live training and fine-tuning entrypoints."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader

from dkfz.live import default_torch_device, live_feature_columns
from dkfz.live.feature_framing import (
    PREPROCESSING_CONTRACT_VERSION,
    feature_backend_name,
    fit_apply_signal_quality_feature,
)
from dkfz.live.io import repo_relative_path, resolve_input_paths
from dkfz.live.event_matching import (
    outer_association_frames,
    EXACT_TOLERANCE_FRAMES,
    METRICS_SCHEMA_VERSION,
)
from dkfz.live.triggers import trigger_backend_name
from dkfz.live.training_evaluation import (
    build_live_eval_outputs,
    write_live_eval_artifacts,
)
from dkfz.live.training_preprocessing import (
    SPLIT_COL,
    TRAIN_SPLIT,
    VAL_SPLIT,
    build_training_frame_dfs,
    load_prepped_training_frame_dfs,
    save_prepped_training_frame_dfs,
    separate_framed_train_val,
)
from dkfz.core.dataset import SlidingWindowDataset
from dkfz.core.model import ECG_RNN
from dkfz.core.network import eval_torch
from dkfz.core.network_helpers import ThresholdSelection


class DebugTimer:
    """Print elapsed phase timings when explicitly enabled."""

    def __init__(self, enabled: bool):
        self.enabled = bool(enabled)
        self.started = time.perf_counter()
        self.last = self.started

    def mark(self, label: str) -> None:
        if not self.enabled:
            return

        now = time.perf_counter()
        print(
            f"[debug timing] {label}: "
            f"+{now - self.last:.3f}s, total {now - self.started:.3f}s"
        )
        self.last = now


_TRAINING_CONFIG_SECTIONS = (
    (
        "Paths",
        (
            "config",
            "model_dir",
            "prepped_file",
            "checkpoint",
            "output_checkpoint",
        ),
    ),
    (
        "Inputs",
        (
            "inputs",
            "threshold_inputs",
            "use_prepped",
            "save_prepped",
            "prep_workers",
        ),
    ),
    (
        "Model",
        (
            "modelname",
            "hidden_dim",
            "num_layers",
            "window_size",
            "include_puls",
        ),
    ),
    (
        "Training",
        (
            "lr",
            "num_epochs",
            "batch_size",
            "seed",
            "test_size",
            "use_amp",
        ),
    ),
    (
        "Preprocessing",
        (
            "raw_hz",
            "frame_size_raw",
            "target_reduce",
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
    (
        "Evaluation",
        (
            "min_distance",
            "error_association_frames",
            "threshold_objective",
            "threshold_sensitivity_efficiency",
        ),
    ),
)

_TRAINING_PATH_KEYS = {
    "config",
    "model_dir",
    "prepped_file",
    "checkpoint",
    "output_checkpoint",
    "inputs",
    "threshold_inputs",
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
    if key not in values:
        return False
    value = values[key]
    if key == "inputs":
        return bool(value)
    return value is not None


def _resolved_config_source(
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


def format_resolved_training_configuration(
    *,
    kwargs: dict[str, Any],
    defaults: dict[str, Any],
    explicit_values: dict[str, Any] | None = None,
    config_values: dict[str, Any] | None = None,
    derived_values: dict[str, Any] | None = None,
    title: str = "Resolved training configuration",
) -> str:
    """Format the effective training settings and where they came from."""
    explicit_values = explicit_values or {}
    config_values = config_values or {}
    derived_values = derived_values or {}
    all_values = dict(kwargs)
    all_values.update(derived_values)

    line_keys = [
        key
        for _, keys in _TRAINING_CONFIG_SECTIONS
        for key in keys
        if key in all_values
    ]
    key_width = max((len(key) for key in line_keys), default=0)
    lines = [title]

    for section, keys in _TRAINING_CONFIG_SECTIONS:
        section_lines = []
        for key in keys:
            if key not in all_values:
                continue

            source = _resolved_config_source(
                key,
                defaults=defaults,
                explicit_values=explicit_values,
                config_values=config_values,
                derived_values=derived_values,
            )
            details = []
            if key in config_values and config_values[key] is not None:
                config_value = _short_value(
                    config_values[key],
                    is_path=key in _TRAINING_PATH_KEYS,
                )
                details.append(f"config: {config_value}")
            if key in defaults:
                default_value = _short_value(
                    defaults[key],
                    is_path=key in _TRAINING_PATH_KEYS,
                )
                details.append(f"default: {default_value}")
            if source in {"explicit", "config"}:
                details.append(f"source: {source}")
            elif source == "derived":
                details.append("derived")

            suffix = f"  ({', '.join(details)})" if details else ""
            section_lines.append(
                f"  {key:<{key_width}} = "
                f"{_short_value(all_values[key], is_path=key in _TRAINING_PATH_KEYS)}"
                f"{suffix}"
            )

        if section_lines:
            lines.append("")
            lines.append(section)
            lines.extend(section_lines)

    return "\n".join(lines)


def print_resolved_training_configuration(**kwargs: Any) -> None:
    """Print the effective training settings in a log-friendly format."""
    print(format_resolved_training_configuration(**kwargs))


_METADATA_PATH_KEYS = {
    "inputs",
    "threshold_inputs",
    "prepped_file",
    "source_config",
    "source_checkpoint",
    "tuned_checkpoint",
}


def _metadata_value_for_json(key: str, value: Any) -> Any:
    is_path_key = (
        key in _METADATA_PATH_KEYS
        or key.endswith("_file")
        or key.endswith("_checkpoint")
    )
    if not is_path_key:
        return value
    if isinstance(value, (list, tuple)):
        return [repo_relative_path(item) for item in value]
    if value is None:
        return None
    return repo_relative_path(value)


def _validate_prepped_input_roles(
    frame_dfs: list[pd.DataFrame],
    *,
    input_paths: list[Path],
    threshold_input_paths: list[Path],
) -> None:
    """Reject a cache whose source files do not match configured data roles."""
    if not threshold_input_paths:
        return

    frames = pd.concat(frame_dfs, ignore_index=True)
    expected = {
        TRAIN_SPLIT: {repo_relative_path(path) for path in input_paths},
        VAL_SPLIT: {repo_relative_path(path) for path in threshold_input_paths},
    }
    actual = {
        split: set(
            frames.loc[frames[SPLIT_COL] == split, "source_file"].astype(str).unique()
        )
        for split in expected
    }
    if actual == expected:
        return

    details = []
    for split in (TRAIN_SPLIT, VAL_SPLIT):
        missing = sorted(expected[split] - actual[split])
        unexpected = sorted(actual[split] - expected[split])
        if missing:
            details.append(f"{split} missing={missing}")
        if unexpected:
            details.append(f"{split} unexpected={unexpected}")
    raise ValueError(
        "Preprocessed cache does not match the configured training and "
        "threshold-selection inputs ("
        + "; ".join(details)
        + "). Regenerate it without --use-prepped."
    )


def load_or_build_training_frames(
    kwargs: dict[str, Any],
    prepped_file: str | Path,
    timing,
) -> tuple[list[pd.DataFrame], list[Path], list[Path]]:
    """Load or build training and threshold-selection frames.

    Separate ``threshold_inputs`` assign all primary inputs to the training
    split and all threshold inputs to validation. When no threshold inputs are
    configured, the legacy within-recording split remains available.
    """
    inputs = kwargs["inputs"]
    threshold_inputs = kwargs.get("threshold_inputs") or []
    signal_quality_mode = str(kwargs.get("signal_quality_mode", "none"))
    signal_quality_stats = kwargs.get("signal_quality_stats")
    if kwargs["use_prepped"]:
        try:
            input_paths = resolve_input_paths(inputs)
        except FileNotFoundError:
            input_paths = [Path(item) for item in inputs]
        try:
            threshold_input_paths = (
                resolve_input_paths(threshold_inputs) if threshold_inputs else []
            )
        except FileNotFoundError:
            threshold_input_paths = [Path(item) for item in threshold_inputs]
        overlap = sorted(set(input_paths).intersection(threshold_input_paths))
        if overlap:
            display = ", ".join(repo_relative_path(path) for path in overlap)
            raise ValueError(
                "Training and threshold-selection inputs must be disjoint; "
                f"overlap: {display}"
            )
        timing.mark("resolved input paths")
        frame_dfs = load_prepped_training_frame_dfs(prepped_file)
        _validate_prepped_input_roles(
            frame_dfs,
            input_paths=input_paths,
            threshold_input_paths=threshold_input_paths,
        )
        timing.mark("loaded and validated preprocessed training frames")
        frame_dfs, signal_quality_stats = fit_apply_signal_quality_feature(
            frame_dfs,
            mode=signal_quality_mode,
            stats=signal_quality_stats,
        )
        if signal_quality_stats is not None:
            kwargs["signal_quality_stats"] = signal_quality_stats
            timing.mark("applied signal quality calibration")
        return frame_dfs, input_paths, threshold_input_paths

    input_paths = resolve_input_paths(inputs)
    threshold_input_paths = (
        resolve_input_paths(threshold_inputs) if threshold_inputs else []
    )
    overlap = sorted(set(input_paths).intersection(threshold_input_paths))
    if overlap:
        display = ", ".join(repo_relative_path(path) for path in overlap)
        raise ValueError(
            "Training and threshold-selection inputs must be disjoint; "
            f"overlap: {display}"
        )
    timing.mark("resolved training and threshold-selection input paths")

    build_kwargs = training_frame_build_kwargs(
        kwargs,
        signal_quality_stats=signal_quality_stats,
    )
    if threshold_input_paths:
        frame_dfs = build_training_frame_dfs(
            input_paths=input_paths,
            validation_fraction=None,
            split=TRAIN_SPLIT,
            **build_kwargs,
        )
        threshold_frame_dfs = build_training_frame_dfs(
            input_paths=threshold_input_paths,
            validation_fraction=None,
            split=VAL_SPLIT,
            **build_kwargs,
        )
        frame_dfs.extend(threshold_frame_dfs)
    else:
        frame_dfs = build_training_frame_dfs(
            input_paths=input_paths,
            validation_fraction=float(kwargs["test_size"]),
            split=None,
            **build_kwargs,
        )
    timing.mark("built training frame data")
    frame_dfs, signal_quality_stats = fit_apply_signal_quality_feature(
        frame_dfs,
        mode=signal_quality_mode,
        stats=signal_quality_stats,
    )
    if signal_quality_stats is not None:
        kwargs["signal_quality_stats"] = signal_quality_stats
        timing.mark("fit/applied signal quality calibration")
    if kwargs["save_prepped"]:
        save_prepped_training_frame_dfs(frame_dfs, prepped_file)
        timing.mark("saved preprocessed training frames")
    return frame_dfs, input_paths, threshold_input_paths


def training_frame_build_kwargs(
    kwargs: dict[str, Any],
    *,
    signal_quality_stats: dict | None,
) -> dict[str, Any]:
    """Return raw-to-frame arguments shared by training orchestrators."""
    signal_quality_mode = str(kwargs.get("signal_quality_mode", "none"))
    return dict(
        ecg1_col=kwargs["ecg1_col"],
        ecg2_col=kwargs["ecg2_col"],
        ecg3_col=kwargs.get("ecg3_col"),
        puls_col=kwargs["puls_col"],
        target_col=kwargs["target_col"],
        not_use_puls=bool(kwargs["not_use_puls"]),
        time_col=kwargs["time_col"],
        time_unit=kwargs["time_unit"],
        target_reduce=kwargs["target_reduce"],
        raw_hz=float(kwargs["raw_hz"]),
        frame_size_raw=int(kwargs["frame_size_raw"]),
        ecg_quantile=float(kwargs["ecg_quantile"]),
        ecg_norm_window_samples=int(kwargs["ecg_norm_window_samples"]),
        ecg1_offset=float(kwargs["ecg1_offset"]),
        ecg2_offset=float(kwargs["ecg2_offset"]),
        ecg3_offset=float(kwargs.get("ecg3_offset", 0.0)),
        ecg_slope_mode=str(kwargs.get("ecg_slope_mode", "none")),
        ecg_slope_slow_window_samples=int(
            kwargs.get("ecg_slope_slow_window_samples", 40)
        ),
        signal_quality_mode=signal_quality_mode,
        signal_quality_stats=signal_quality_stats,
        ecg_rotation_angle=float(kwargs["ecg_rotation_angle"]),
        puls_offset=float(kwargs["puls_offset"]),
        puls_quantile_low=float(kwargs["puls_quantile_low"]),
        puls_quantile_high=float(kwargs["puls_quantile_high"]),
        puls_norm_window_frames=int(kwargs["puls_norm_window_frames"]),
        puls_norm_warmup_value=float(kwargs["puls_norm_warmup_value"]),
        puls_clip=bool(kwargs["puls_clip"]),
        align_to_puls_blocks=bool(kwargs["align_to_puls_blocks"]),
        verify_puls_constant_within_block=bool(
            kwargs["verify_puls_constant_within_block"]
        ),
        long_sequence_split_seconds=float(kwargs["long_sequence_split_seconds"]),
        long_sequence_split_parts=int(kwargs["long_sequence_split_parts"]),
        prep_workers=int(kwargs.get("prep_workers", 1)),
    )


def training_feature_columns(
    frame_size_raw: int,
    not_use_puls: bool,
    include_ecg3: bool = False,
    ecg_slope_mode: str = "none",
    signal_quality_mode: str = "none",
) -> tuple[list[str], int]:
    feature_cols = live_feature_columns(
        frame_size_raw=frame_size_raw,
        include_puls=not not_use_puls,
        ecg_cols=("ecg1", "ecg2", "ecg3") if include_ecg3 else ("ecg1", "ecg2"),
        ecg_slope_mode=ecg_slope_mode,
        signal_quality_mode=signal_quality_mode,
    )
    return feature_cols, len(feature_cols)


def build_training_loaders(
    frame_dfs: list[pd.DataFrame],
    *,
    feature_cols: list[str],
    target_col: str,
    window_size: int,
    batch_size: int,
    timing,
) -> tuple[SlidingWindowDataset, SlidingWindowDataset, DataLoader, DataLoader]:
    train_data, val_data = separate_framed_train_val(frame_dfs)
    timing.mark("separated train and validation frames")
    train_dataset = SlidingWindowDataset(
        train_data,
        feature_cols=feature_cols,
        target_col=target_col,
        window_size=window_size,
    )
    val_dataset = SlidingWindowDataset(
        val_data,
        feature_cols=feature_cols,
        target_col=target_col,
        window_size=window_size,
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    timing.mark("built datasets and dataloaders")
    return train_dataset, val_dataset, train_loader, val_loader


def build_ecg_rnn(
    *,
    input_dim: int,
    hidden_dim: int,
    num_layers: int,
    device: torch.device | None = None,
    checkpoint_path: str | Path | None = None,
) -> ECG_RNN:
    device = device or default_torch_device()
    model = ECG_RNN(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    ).to(device)
    if checkpoint_path is not None:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return model


def evaluate_and_write_live_artifacts(
    *,
    model: ECG_RNN,
    loader: DataLoader,
    dataset: SlidingWindowDataset,
    model_dir: Path,
    holdoff_frames: int,
    error_association_frames: int,
    threshold_objective: str,
    threshold_sensitivity_efficiency: float,
    frame_size_raw: int,
    feature_cols: list[str],
    target_col: str,
    device: torch.device,
    use_amp: bool,
    timing,
    prefix: str = "train",
    threshold_label: str = "searched validation threshold",
    outputs_label: str = "built live evaluation outputs",
    artifacts_label: str = "wrote live evaluation artifacts",
    metrics_dir: Path | None = None,
    metrics_prefix: str | None = None,
    raw_hz: float = 400.0,
) -> ThresholdSelection:
    threshold_search_df, threshold_selection = eval_torch(
        model,
        loader,
        error_association_frames=error_association_frames,
        holdoff_frames=holdoff_frames,
        threshold_objective=threshold_objective,
        threshold_sensitivity_efficiency=threshold_sensitivity_efficiency,
        device=device,
        use_amp=use_amp,
        return_threshold_selection=True,
        outer_error_association_frames=outer_association_frames(raw_hz, frame_size_raw),
    )
    best_threshold = threshold_selection.threshold
    selected_metrics = threshold_selection.metrics
    print(
        f"Selected validation threshold {best_threshold:.6f} using "
        f"{threshold_selection.objective}: "
        f"PE={selected_metrics['position_error']}, "
        f"oPE={selected_metrics['outer_position_error']}, "
        f"FP={selected_metrics['unmatched_false_positive']}, "
        f"FN={selected_metrics['unmatched_false_negative']}, "
        f"Se={float(selected_metrics['exact_sensitivity']):.6f}, "
        "+P="
        f"{float(selected_metrics['exact_positive_predictive_value']):.6f}, "
        f"weighted_cost={selected_metrics['weighted_error_count']}, "
        f"weighted_DER={float(selected_metrics['weighted_detection_error']):.6f}"
    )
    print(
        "Sensitivity selection band: "
        f"best attainable={threshold_selection.best_attainable_sensitivity:.6f}, "
        f"efficiency={threshold_selection.sensitivity_efficiency:.2%}, "
        f"effective floor={threshold_selection.effective_sensitivity_floor:.6f}"
    )
    print(
        f"fp_threshold={threshold_selection.fp_threshold:.6f}, "
        f"der_threshold={threshold_selection.der_threshold:.6f}"
    )
    timing.mark(threshold_label)
    live_eval_predictions, live_eval_events = build_live_eval_outputs(
        model=model,
        loader=loader,
        dataset=dataset,
        threshold=float(best_threshold),
        holdoff_frames=holdoff_frames,
        frame_size_raw=frame_size_raw,
        feature_cols=feature_cols,
        target_col=target_col,
        device=device,
        use_amp=use_amp,
    )
    timing.mark(outputs_label)
    write_live_eval_artifacts(
        model_dir=model_dir,
        prediction_df=live_eval_predictions,
        events_df=live_eval_events,
        threshold_search_df=threshold_search_df,
        target_col=target_col,
        threshold=float(best_threshold),
        holdoff_frames=holdoff_frames,
        error_association_frames=error_association_frames,
        prefix=prefix,
        metrics_dir=metrics_dir,
        metrics_prefix=metrics_prefix,
        threshold_selection=threshold_selection,
        outer_error_association_frames=outer_association_frames(raw_hz, frame_size_raw),
    )
    timing.mark(artifacts_label)
    return threshold_selection


def base_training_metadata(
    *,
    kwargs: dict[str, Any],
    input_dim: int,
    feature_cols: list[str],
    best_threshold: float,
    threshold_selection: ThresholdSelection,
    input_paths: list[Path],
    threshold_input_paths: list[Path] | None,
    prepped_file: str | Path,
) -> dict[str, Any]:
    """Build metadata keys shared by training and fine-tuning configs."""
    return {
        "modelname": kwargs["modelname"],
        "metrics_schema_version": METRICS_SCHEMA_VERSION,
        "preprocessing_backend": feature_backend_name(),
        "preprocessing_contract_version": PREPROCESSING_CONTRACT_VERSION,
        "trigger_backend": trigger_backend_name(),
        "input_dim": input_dim,
        "feature_cols": feature_cols,
        "hidden_dim": int(kwargs["hidden_dim"]),
        "num_layers": int(kwargs["num_layers"]),
        "window_size": int(kwargs["window_size"]),
        "lr": float(kwargs["lr"]),
        "num_epochs": int(kwargs["num_epochs"]),
        "batch_size": int(kwargs["batch_size"]),
        "seed": int(kwargs.get("seed", 42)),
        "test_size": float(kwargs["test_size"]),
        "long_sequence_split_seconds": float(kwargs["long_sequence_split_seconds"]),
        "long_sequence_split_parts": int(kwargs["long_sequence_split_parts"]),
        "threshold": float(best_threshold),
        **threshold_selection.policy_metadata(),
        "manual_threshold": kwargs.get("manual_threshold"),
        "outer_error_association_ms": 40.0,
        "outer_error_association_frames": outer_association_frames(
            kwargs["raw_hz"], kwargs["frame_size_raw"]
        ),
        "threshold_objective": threshold_selection.objective,
        "threshold_sensitivity_efficiency": (
            threshold_selection.sensitivity_efficiency
        ),
        "threshold_best_attainable_exact_sensitivity": (
            threshold_selection.best_attainable_sensitivity
        ),
        "threshold_effective_exact_sensitivity_floor": (
            threshold_selection.effective_sensitivity_floor
        ),
        **{
            f"threshold_validation_{key}": value
            for key, value in threshold_selection.metrics.items()
        },
        "holdoff_frames": int(kwargs["min_distance"]),
        "min_distance": int(kwargs["min_distance"]),
        "exact_tolerance_frames": EXACT_TOLERANCE_FRAMES,
        "error_association_frames": int(kwargs["error_association_frames"]),
        "save_last": bool(kwargs["save_last"]),
        "raw_hz": float(kwargs["raw_hz"]),
        "frame_size_raw": int(kwargs["frame_size_raw"]),
        "ecg_quantile": float(kwargs["ecg_quantile"]),
        "ecg_norm_window_samples": int(kwargs["ecg_norm_window_samples"]),
        "ecg1_offset": float(kwargs["ecg1_offset"]),
        "ecg2_offset": float(kwargs["ecg2_offset"]),
        "ecg3_offset": float(kwargs.get("ecg3_offset", 0.0)),
        "ecg_slope_mode": str(kwargs.get("ecg_slope_mode", "none")),
        "ecg_slope_slow_window_samples": int(
            kwargs.get("ecg_slope_slow_window_samples", 40)
        ),
        "signal_quality_mode": str(kwargs.get("signal_quality_mode", "none")),
        "signal_quality_stats": kwargs.get("signal_quality_stats"),
        "ecg_rotation_angle": float(kwargs["ecg_rotation_angle"]),
        "puls_offset": float(kwargs["puls_offset"]),
        "puls_quantile_low": float(kwargs["puls_quantile_low"]),
        "puls_quantile_high": float(kwargs["puls_quantile_high"]),
        "puls_norm_window_frames": int(kwargs["puls_norm_window_frames"]),
        "puls_norm_warmup_value": float(kwargs["puls_norm_warmup_value"]),
        "puls_clip": bool(kwargs["puls_clip"]),
        "align_to_puls_blocks": bool(kwargs["align_to_puls_blocks"]),
        "verify_puls_constant_within_block": bool(
            kwargs["verify_puls_constant_within_block"]
        ),
        "include_puls": not bool(kwargs["not_use_puls"]),
        "target_reduce": kwargs["target_reduce"],
        "target_col": kwargs["target_col"],
        "ecg1_col": kwargs["ecg1_col"],
        "ecg2_col": kwargs["ecg2_col"],
        "ecg3_col": kwargs.get("ecg3_col"),
        "puls_col": kwargs["puls_col"],
        "time_col": kwargs["time_col"],
        "time_unit": kwargs["time_unit"],
        "use_amp": bool(kwargs["use_amp"]),
        "use_prepped": bool(kwargs["use_prepped"]),
        "save_prepped": bool(kwargs["save_prepped"]),
        "prep_workers": int(kwargs.get("prep_workers", 1)),
        "inputs": [repo_relative_path(p) for p in input_paths],
        "threshold_inputs": [
            repo_relative_path(p) for p in (threshold_input_paths or [])
        ],
        "prepped_file": repo_relative_path(prepped_file),
    }


def write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    metadata = {
        key: _metadata_value_for_json(key, value) for key, value in metadata.items()
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
