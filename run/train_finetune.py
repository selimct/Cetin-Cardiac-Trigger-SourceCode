# Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Developed at the German Cancer Research Center (DKFZ).

"""Fine-tune an already trained live ECG LSTM checkpoint.

This is the continuation-training sibling of `run/train.py`: it reads the
training contract from a config, loads `models/<modelname>/best.pth` by default,
trains for a few additional epochs on the configured or overridden data, and
writes:

- `models/<modelname>/best_tuned.pth`
- `models/<modelname>/tuning/tuned_config.json`
- `models/<modelname>/tuning/tuned_history.json`
- `models/<modelname>/tuning/tuned_trigger_events.parquet`
- `models/<modelname>/tuning/tuned_all_events.parquet`
- `models/<modelname>/metrics/tuned_metrics.json`
- `models/<modelname>/metrics/tuned_threshold_search.csv`
- `models/<modelname>/metrics/tuned_threshold_dashboard.pdf`
- `models/<modelname>/metrics/tuned_threshold_errors.pdf`

The model architecture and preprocessing settings must stay compatible with the
source checkpoint. After validation, export the fine-tuned checkpoint explicitly
with `run/jetson_export_onnx.py --checkpoint ...`.
"""

from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path
import sys
import time

import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dkfz.live import default_torch_device
from dkfz.live.training_pipeline import (
    DebugTimer,
    base_training_metadata,
    build_ecg_rnn,
    build_training_loaders,
    evaluate_and_write_live_artifacts,
    load_or_build_training_frames,
    print_resolved_training_configuration,
    training_feature_columns,
    write_metadata,
)
from train import (
    TRAINING_DEFAULTS,
    _load_training_config_values,
    _training_kwargs_from_config_and_values,
)
from dkfz.live.io import repo_relative_path
from dkfz.live.training_seed import set_training_seed
from dkfz.core.network import train_torch
from dkfz.core.utils import chop_microseconds, ensure_dir

sys.stdout.reconfigure(line_buffering=True, write_through=True)

FINE_TUNE_DEFAULT_LR = 1e-4
FINE_TUNE_DEFAULT_EPOCHS = 10
FINE_TUNE_DEFAULTS = dict(TRAINING_DEFAULTS)
FINE_TUNE_DEFAULTS.update(
    {
        "lr": FINE_TUNE_DEFAULT_LR,
        "num_epochs": FINE_TUNE_DEFAULT_EPOCHS,
        "save_prepped": False,
        "use_prepped": False,
        "prepped_file": None,
    }
)


def _default_config_path(modelname: str | None) -> Path:
    return (
        Path("models")
        / str(modelname or TRAINING_DEFAULTS["modelname"])
        / "train_config.json"
    )


def _artifact_path(model_dir: Path, value: str | None, default_name: str) -> Path:
    path = Path(value or default_name)
    if path.is_absolute():
        return path
    if path.parent == Path("."):
        return model_dir / path
    return path.resolve()


def _resolve_fine_tune_kwargs(values: dict) -> tuple[dict, dict, dict]:
    """Merge fine-tune CLI values with the train config and safe FT defaults."""
    values = dict(values)
    explicit_values = dict(values)
    explicit_save_prepped = values.get("save_prepped") is not None
    explicit_use_prepped = values.get("use_prepped") is not None
    explicit_prepped_file = bool(
        values.get("_explicit_prepped_file", values.get("prepped_file") is not None)
    )
    values["config"] = str(
        values.get("config") or _default_config_path(values.get("modelname"))
    )
    config_values = _load_training_config_values(values["config"])
    summary_config_values = dict(config_values)
    kwargs = _training_kwargs_from_config_and_values(
        values,
        config_values=config_values,
    )

    # Fine-tune training length/rate are intentionally independent from the
    # source training config. Only explicit caller/CLI values override these.
    if values.get("lr") is None:
        kwargs["lr"] = FINE_TUNE_DEFAULT_LR
        summary_config_values.pop("lr", None)
    if values.get("num_epochs") is None:
        kwargs["num_epochs"] = FINE_TUNE_DEFAULT_EPOCHS
        summary_config_values.pop("num_epochs", None)

    # The source training config may point at the original training cache.
    # Fine-tuning should rebuild from the chosen raw inputs unless the caller
    # explicitly opts into a preprocessed fine-tune cache.
    if not explicit_save_prepped:
        kwargs["save_prepped"] = False
        summary_config_values.pop("save_prepped", None)
    if not explicit_use_prepped:
        kwargs["use_prepped"] = False
        summary_config_values.pop("use_prepped", None)
    if not explicit_prepped_file:
        kwargs["prepped_file"] = None
        summary_config_values.pop("prepped_file", None)

    kwargs["checkpoint"] = values.get("checkpoint")
    kwargs["output_checkpoint"] = values.get("output_checkpoint")
    kwargs["_explicit_prepped_file"] = explicit_prepped_file
    return kwargs, summary_config_values, explicit_values


def live_fine_tuning(
    inputs: list[str] | None = None,
    config: str | None = None,
    modelname: str | None = None,
    checkpoint: str | None = None,
    output_checkpoint: str | None = None,
    lr: float | None = None,
    num_epochs: int | None = None,
    batch_size: int | None = None,
    seed: int | None = None,
    test_size: float | None = None,
    min_distance: int | None = None,
    error_association_frames: int | None = None,
    threshold_objective: str | None = None,
    threshold_sensitivity_efficiency: float | None = None,
    use_amp: bool | None = None,
    save_last: bool | None = None,
    save_prepped: bool | None = None,
    use_prepped: bool | None = None,
    prepped_file: str | None = None,
    debug_timing: bool | None = None,
    threshold_inputs: list[str] | None = None,
    **training_values,
) -> None:
    """Continue training a live ECG model from an existing checkpoint."""
    values = locals()
    values.update(values.pop("training_values"))
    kwargs, config_values, explicit_values = _resolve_fine_tune_kwargs(values)
    cfg = config_values

    inputs = kwargs["inputs"]
    modelname = kwargs["modelname"]
    lr = float(kwargs["lr"])
    hidden_dim = int(kwargs["hidden_dim"])
    num_layers = int(kwargs["num_layers"])
    window_size = int(kwargs["window_size"])
    num_epochs = int(kwargs["num_epochs"])
    batch_size = int(kwargs["batch_size"])
    seed = set_training_seed(int(kwargs["seed"]))
    target_col = kwargs["target_col"]
    ecg1_col = kwargs["ecg1_col"]
    ecg2_col = kwargs["ecg2_col"]
    ecg3_col = kwargs.get("ecg3_col")
    puls_col = kwargs["puls_col"]
    not_use_puls = bool(kwargs["not_use_puls"])
    time_col = kwargs["time_col"]
    time_unit = kwargs["time_unit"]
    raw_hz = float(kwargs["raw_hz"])
    frame_size_raw = int(kwargs["frame_size_raw"])
    target_reduce = kwargs["target_reduce"]
    ecg_quantile = float(kwargs["ecg_quantile"])
    ecg_norm_window_samples = int(kwargs["ecg_norm_window_samples"])
    ecg1_offset = float(kwargs["ecg1_offset"])
    ecg2_offset = float(kwargs["ecg2_offset"])
    ecg3_offset = float(kwargs.get("ecg3_offset", 0.0))
    ecg_slope_mode = str(kwargs.get("ecg_slope_mode", "none"))
    ecg_slope_slow_window_samples = int(kwargs.get("ecg_slope_slow_window_samples", 40))
    signal_quality_mode = str(kwargs.get("signal_quality_mode", "none"))
    ecg_rotation_angle = float(kwargs["ecg_rotation_angle"])
    puls_offset = float(kwargs["puls_offset"])
    puls_quantile_low = float(kwargs["puls_quantile_low"])
    puls_quantile_high = float(kwargs["puls_quantile_high"])
    puls_norm_window_frames = int(kwargs["puls_norm_window_frames"])
    puls_norm_warmup_value = float(kwargs["puls_norm_warmup_value"])
    puls_clip = bool(kwargs["puls_clip"])
    align_to_puls_blocks = bool(kwargs["align_to_puls_blocks"])
    verify_puls_constant_within_block = bool(
        kwargs["verify_puls_constant_within_block"]
    )
    min_distance = int(kwargs["min_distance"])
    error_association_frames = int(kwargs["error_association_frames"])
    threshold_objective = str(kwargs["threshold_objective"])
    threshold_sensitivity_efficiency = float(kwargs["threshold_sensitivity_efficiency"])
    if threshold_objective not in {"min_error", "min_fp"}:
        raise ValueError("threshold_objective must be 'min_error' or 'min_fp'.")
    if error_association_frames < 0:
        raise ValueError("error_association_frames must be zero or positive.")
    if not 0.0 < threshold_sensitivity_efficiency <= 1.0:
        raise ValueError(
            "threshold_sensitivity_efficiency must be greater than 0 and at most 1."
        )
    use_amp = bool(kwargs["use_amp"])
    save_last = bool(kwargs["save_last"])
    save_prepped = bool(kwargs["save_prepped"])
    use_prepped = bool(kwargs["use_prepped"])
    prepped_file = kwargs["prepped_file"]
    debug_timing = bool(kwargs["debug_timing"])

    start_time = time.time()
    timing = DebugTimer(debug_timing)
    if int(kwargs.get("prediction_delay_frames", 0)) != int(
        cfg.get("prediction_delay_frames", 0)
    ):
        raise ValueError(
            "Changing the primary target requires a separate training run/modelname, rather than overwriting a source model"
        )
    model_dir = ensure_dir(Path("models") / modelname)
    metrics_dir = ensure_dir(model_dir / "metrics")
    tuning_dir = ensure_dir(model_dir / "tuning")
    checkpoint_path = _artifact_path(model_dir, kwargs["checkpoint"], "best.pth")
    output_checkpoint_path = _artifact_path(
        model_dir, kwargs["output_checkpoint"], "best_tuned.pth"
    )
    last_checkpoint_path = model_dir / "last_tuned.pth"

    if prepped_file is None:
        prepped_file = str(tuning_dir / "tuned_prepped.parquet")
    elif (
        save_prepped
        and not use_prepped
        and not bool(kwargs.get("_explicit_prepped_file"))
    ):
        prepped_file = str(tuning_dir / "tuned_prepped.parquet")
    timing.mark("initialized fine-tune paths")
    derived_config_values = {
        "model_dir": str(model_dir),
        "prepped_file": str(prepped_file),
        "checkpoint": str(checkpoint_path),
        "output_checkpoint": str(output_checkpoint_path),
        "include_puls": not not_use_puls,
    }
    if explicit_values.get("config") is None:
        derived_config_values["config"] = kwargs["config"]
    print_resolved_training_configuration(
        kwargs=kwargs,
        defaults=FINE_TUNE_DEFAULTS,
        explicit_values=explicit_values,
        config_values=config_values,
        derived_values=derived_config_values,
        title="Resolved fine-tune configuration",
    )

    frame_dfs, input_paths, threshold_input_paths = load_or_build_training_frames(
        kwargs,
        prepped_file,
        timing,
    )
    feature_cols, input_dim = training_feature_columns(
        frame_size_raw,
        not_use_puls,
        include_ecg3=ecg3_col is not None,
        ecg_slope_mode=ecg_slope_mode,
        signal_quality_mode=signal_quality_mode,
    )
    if (
        cfg.get("feature_cols") is not None
        and list(cfg["feature_cols"]) != feature_cols
    ):
        raise ValueError(
            "Fine-tune feature columns do not match the source config. "
            f"config={cfg['feature_cols']}, current={feature_cols}"
        )

    if cfg.get("input_dim") is not None and int(cfg["input_dim"]) != input_dim:
        raise ValueError(
            f"Fine-tune input_dim={input_dim} does not match config "
            f"input_dim={cfg['input_dim']}."
        )

    print(f"{feature_cols = }")
    print(f"{input_dim = }, {window_size = }, {hidden_dim = }, {num_layers = }")
    print(
        f"Fine-tuning from {repo_relative_path(checkpoint_path)} "
        f"to {repo_relative_path(output_checkpoint_path)}"
    )
    timing.mark("prepared feature metadata")

    _, val_dataset, train_loader, val_loader = build_training_loaders(
        frame_dfs,
        feature_cols=feature_cols,
        target_col=target_col,
        window_size=window_size,
        batch_size=batch_size,
        timing=timing,
        recovery_head=bool(kwargs.get("recovery_head", False)),
        prediction_delay_frames=int(kwargs.get("prediction_delay_frames", 0)),
    )

    device = default_torch_device()
    model = build_ecg_rnn(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        device=device,
        checkpoint_path=checkpoint_path,
        initialize_recovery=bool(kwargs.get("recovery_head", False)),
        recovery_head=bool(kwargs.get("recovery_head", False)),
        prediction_delay_frames=int(kwargs.get("prediction_delay_frames", 0)),
    )
    criterion = nn.BCEWithLogitsLoss()
    if kwargs.get("recovery_only"):
        model.freeze_primary()
    timing.mark("loaded source checkpoint")

    train_torch(
        model,
        criterion,
        train_loader,
        val_loader,
        lr=lr,
        num_epochs=num_epochs,
        early_stopping=0,
        device=device,
        modelname=modelname,
        non_blocking=True,
        use_amp=use_amp,
        save_last=save_last,
        best_filename=output_checkpoint_path,
        last_filename=last_checkpoint_path,
        history_filename=tuning_dir / "tuned_history.json",
        cache_frozen_recovery=bool(kwargs.get("cache_frozen_recovery", True)),
    )
    timing.mark("fine-tuned model")

    saved_model = build_ecg_rnn(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        device=device,
        checkpoint_path=output_checkpoint_path,
        recovery_head=bool(kwargs.get("recovery_head", False)),
        prediction_delay_frames=int(kwargs.get("prediction_delay_frames", 0)),
    )
    timing.mark("loaded fine-tuned best checkpoint")

    threshold_selection = evaluate_and_write_live_artifacts(
        model=saved_model,
        loader=val_loader,
        dataset=val_dataset,
        model_dir=tuning_dir,
        holdoff_frames=min_distance,
        error_association_frames=error_association_frames,
        threshold_objective=threshold_objective,
        threshold_sensitivity_efficiency=threshold_sensitivity_efficiency,
        frame_size_raw=frame_size_raw,
        raw_hz=raw_hz,
        feature_cols=feature_cols,
        target_col=target_col,
        device=device,
        use_amp=use_amp,
        timing=timing,
        prefix="tuned",
        metrics_dir=metrics_dir,
        metrics_prefix="tuned",
        artifacts_label="wrote fine-tune evaluation artifacts",
        plots=kwargs["plots"],
        recovery_max_fp_increase=float(kwargs.get("recovery_max_fp_increase", 0.10)),
        recovery_min_extra_fp_per_minute=float(
            kwargs.get("recovery_min_extra_fp_per_minute", 0.5)
        ),
        primary_policy=(
            dict(
                threshold=(
                    cfg.get("manual_threshold")
                    if cfg.get("manual_threshold") is not None
                    else cfg["threshold"]
                ),
                fp_threshold=cfg.get("fp_threshold"),
                der_threshold=cfg.get("der_threshold"),
            )
            if kwargs.get("recovery_only")
            else None
        ),
    )
    best_threshold = threshold_selection.threshold

    metadata = base_training_metadata(
        kwargs=kwargs,
        input_dim=input_dim,
        feature_cols=feature_cols,
        best_threshold=best_threshold,
        threshold_selection=threshold_selection,
        input_paths=input_paths,
        threshold_input_paths=threshold_input_paths,
        prepped_file=prepped_file,
    )
    metadata.update(
        {
            "source_config": str(kwargs["config"]),
            "source_checkpoint": str(checkpoint_path),
            "tuned_checkpoint": str(output_checkpoint_path),
            "tuned_history_file": str(tuning_dir / "tuned_history.json"),
            "tuned_all_events_file": str(tuning_dir / "tuned_all_events.parquet"),
            "tuned_trigger_events_file": str(
                tuning_dir / "tuned_trigger_events.parquet"
            ),
            "tuned_metrics_file": str(metrics_dir / "tuned_metrics.json"),
            "tuned_threshold_search_file": str(
                metrics_dir / "tuned_threshold_search.csv"
            ),
        }
    )
    if metadata["plots_enabled"]:
        metadata.update(
            tuned_threshold_dashboard_file=str(
                metrics_dir / "tuned_threshold_dashboard.pdf"
            ),
            tuned_threshold_error_examples_file=str(
                metrics_dir / "tuned_threshold_errors.pdf"
            ),
        )
    write_metadata(tuning_dir / "tuned_config.json", metadata)
    timing.mark("wrote fine-tune config")

    print(f"Selected fine-tune validation threshold: {best_threshold:.6f}")
    print(
        "Live fine-tuning took",
        chop_microseconds(timedelta(seconds=time.time() - start_time)),
        "to run.",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    from dkfz.live.training_config import add_prediction_training_arguments

    add_prediction_training_arguments(parser)
    parser.add_argument(
        "inputs",
        nargs="*",
        help="Optional parquet files, directories, or globs. Defaults to config inputs.",
    )
    parser.add_argument(
        "--threshold-inputs",
        nargs="+",
        default=None,
        help=(
            "Disjoint parquet files, directories, or globs used for best "
            "fine-tuned checkpoint and threshold selection."
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Training/fine-tune config JSON. Defaults to "
            "models/<modelname>/train_config.json."
        ),
    )
    parser.add_argument("--modelname", type=str, default=None)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Source checkpoint. Defaults to models/<modelname>/best.pth.",
    )
    parser.add_argument(
        "--output-checkpoint",
        type=str,
        default=None,
        help="Fine-tuned checkpoint. Defaults to models/<modelname>/best_tuned.pth.",
    )
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for Python, NumPy, PyTorch, and CUDA. Defaults to config or 42.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=None,
        help=(
            "Legacy within-recording validation fraction, used only when "
            "threshold_inputs is empty."
        ),
    )
    parser.add_argument(
        "--long-sequence-split-seconds",
        type=float,
        default=None,
        help="Split raw sequences longer than this many seconds before framing.",
    )
    parser.add_argument(
        "--long-sequence-split-parts",
        type=int,
        default=None,
        help="Number of independent streams made from each long raw sequence.",
    )
    parser.add_argument("--min-distance", type=int, default=None)
    parser.add_argument("--error-association-frames", type=int, default=None)
    parser.add_argument(
        "--threshold-objective",
        choices=("min_error", "min_fp"),
        default=None,
        help=("Threshold selection policy using exact-frame event metrics."),
    )
    parser.add_argument(
        "--threshold-sensitivity-efficiency",
        type=float,
        default=None,
        help=(
            "Fraction of the best attainable validation exact sensitivity to "
            "retain as the selection floor. Defaults to 0.95."
        ),
    )
    parser.add_argument("--amp", action="store_true", dest="use_amp")
    parser.add_argument("--no-amp", action="store_false", dest="use_amp")
    parser.set_defaults(use_amp=None)
    parser.add_argument(
        "--save-last",
        action="store_true",
        dest="save_last",
        default=None,
        help="Save final fine-tune weights to models/<modelname>/last_tuned.pth.",
    )
    parser.add_argument("--no-save-last", action="store_false", dest="save_last")
    parser.add_argument(
        "--save-prepped", action="store_true", dest="save_prepped", default=None
    )
    parser.add_argument("--no-save-prepped", action="store_false", dest="save_prepped")
    parser.add_argument(
        "--use-prepped", action="store_true", dest="use_prepped", default=None
    )
    parser.add_argument("--no-use-prepped", action="store_false", dest="use_prepped")
    parser.add_argument(
        "--prepped-file",
        type=str,
        default=None,
        help=(
            "Path for --save-prepped/--use-prepped. Fine-tune generated "
            "prepped data defaults to "
            "models/<modelname>/tuning/tuned_prepped.parquet."
        ),
    )
    parser.add_argument(
        "--debug-timing", action="store_true", dest="debug_timing", default=None
    )
    parser.add_argument("--no-debug-timing", action="store_false", dest="debug_timing")

    from dkfz.live.training_config import add_recovery_training_arguments

    add_recovery_training_arguments(parser, fine_tune=True)
    args = parser.parse_args()
    live_fine_tuning(**vars(args))
