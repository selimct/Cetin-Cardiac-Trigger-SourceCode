# Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Developed at the German Cancer Research Center (DKFZ).

"""Train an ECG_RNN using the same feature path used during live prediction.

This script reads annotated parquet files at raw sample rate, converts them into
causal live-style model-rate feature frames, trains an LSTM classifier, evaluates a
threshold, and writes:

- `models/<modelname>/best.pth`
- `models/<modelname>/train_config.json`
- `models/<modelname>/train_trigger_events.parquet`
- `models/<modelname>/train_all_events.parquet`
- `models/<modelname>/metrics/train_metrics.json`
- `models/<modelname>/metrics/train_threshold_search.csv`
- `models/<modelname>/metrics/train_threshold_dashboard.pdf`
- `models/<modelname>/metrics/train_threshold_errors.pdf`

The config JSON stores the preprocessing choices needed by `run/offline_predict.py`,
including ECG column names, pulse usage, frame size, normalization settings, and
ECG rotation angle. A config can also be passed back into this script with
`--config`; explicit CLI arguments override config values.

By default, training reads all rows from `data/live/p*/*`, while checkpoint and
threshold selection use all rows from `data/live/sp*/*`. Use
`--save-prepped` to cache the slow 400 Hz to 50 Hz preprocessing step, and
`--use-prepped` to train from that cache when preprocessing settings have not
changed.
"""

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

from dkfz.live import (
    default_torch_device,
)
from dkfz.live.training_config import (
    TRAINING_DEFAULTS,
    load_training_config_values,
    training_kwargs_from_config_and_values,
)
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
from dkfz.live.training_preprocessing import (
    default_prepped_file,
)
from dkfz.live.training_seed import set_training_seed
from dkfz.core.network import train_torch
from dkfz.core.utils import chop_microseconds, ensure_dir

sys.stdout.reconfigure(line_buffering=True, write_through=True)

_load_training_config_values = load_training_config_values
_training_kwargs_from_config_and_values = training_kwargs_from_config_and_values


def live_training(
    inputs: list[str] | None = None,
    config: str | None = None,
    modelname: str | None = None,
    lr: float | None = None,
    hidden_dim: int | None = None,
    num_layers: int | None = None,
    window_size: int | None = None,
    num_epochs: int | None = None,
    batch_size: int | None = None,
    seed: int | None = None,
    test_size: float | None = None,
    long_sequence_split_seconds: float | None = None,
    long_sequence_split_parts: int | None = None,
    target_col: str | None = None,
    ecg1_col: str | None = None,
    ecg2_col: str | None = None,
    ecg3_col: str | None = None,
    puls_col: str | None = None,
    not_use_puls: bool | None = None,
    time_col: str | None = None,
    time_unit: str | None = None,
    raw_hz: float | None = None,
    frame_size_raw: int | None = None,
    target_reduce: str | None = None,
    ecg_quantile: float | None = None,
    ecg_norm_window_samples: int | None = None,
    ecg1_offset: float | None = None,
    ecg2_offset: float | None = None,
    ecg3_offset: float | None = None,
    ecg_slope_mode: str | None = None,
    ecg_slope_slow_window_samples: int | None = None,
    signal_quality_mode: str | None = None,
    ecg_rotation_angle: float | None = None,
    puls_offset: float | None = None,
    puls_quantile_low: float | None = None,
    puls_quantile_high: float | None = None,
    puls_norm_window_frames: int | None = None,
    puls_norm_warmup_value: float | None = None,
    puls_clip: bool | None = None,
    align_to_puls_blocks: bool | None = None,
    verify_puls_constant_within_block: bool | None = None,
    min_distance: int | None = None,
    error_association_frames: int | None = None,
    threshold_objective: str | None = None,
    threshold_sensitivity_efficiency: float | None = None,
    use_amp: bool | None = None,
    save_last: bool | None = None,
    save_prepped: bool | None = None,
    use_prepped: bool | None = None,
    prepped_file: str | None = None,
    prep_workers: int | None = None,
    debug_timing: bool | None = None,
    plots: bool | None = None,
    threshold_inputs: list[str] | None = None,
    recovery_head: bool | None = None,
    recovery_max_fp_increase: float | None = None,
    recovery_min_extra_fp_per_minute: float | None = None,
    cache_frozen_recovery: bool | None = None,
    prediction_mode: str | None = None,
    prediction_delay_frames: int | None = None,
    signal_quality_puls_flat_window_frames: int | None = None,
    signal_quality_puls_median_window_frames: int | None = None,
):
    """Train and evaluate a live-style ECG trigger model.

    Args:
        inputs: Parquet files, directories, or globs containing annotated data.
        threshold_inputs: Disjoint parquet inputs used for validation, best
            checkpoint selection, and operating-threshold selection. When
            empty, retain the legacy within-recording split from `inputs`.
        config: Optional training config JSON path.
        modelname: Name used for saved model/config/log files.
        lr: Learning rate.
        hidden_dim: LSTM hidden dimension.
        num_layers: Number of LSTM layers.
        window_size: Number of model frames per model input window.
        num_epochs: Maximum training epochs.
        batch_size: DataLoader batch size.
        seed: Base random seed for Python, NumPy, PyTorch, and CUDA.
        test_size: Legacy fraction of each raw sequence segment used for
            validation when `threshold_inputs` is empty.
        long_sequence_split_seconds: Raw sequences longer than this are split
            into `long_sequence_split_parts` independent streams before framing.
        long_sequence_split_parts: Number of independent streams produced from
            raw sequences longer than `long_sequence_split_seconds`.
        target_col: Ground-truth column, usually `target`.
        ecg1_col: Input ECG1 column, usually `ecg1_raw`.
        ecg2_col: Input ECG2 column, usually `ecg2_raw`.
        ecg3_col: Optional measured ECG3 column, usually `ecg3_raw`.
        puls_col: Input pulse column. Ignored when `not_use_puls=True`.
        not_use_puls: Train without pulse as a model feature.
        time_col: Optional time column used for output metadata.
        time_unit: Unit of `time_col`, either `ms` or `s`.
        raw_hz: Raw sample rate.
        frame_size_raw: Raw samples per feature frame.
        target_reduce: How raw labels are reduced into one label per frame.
        ecg_quantile: ECG norm quantile used for causal scaling.
        ecg_norm_window_samples: ECG norm history length.
        ecg1_offset: Offset subtracted from ECG1 before rotation/normalization.
        ecg2_offset: Offset subtracted from ECG2 before rotation/normalization.
        ecg3_offset: Offset subtracted from measured ECG3 when rotation is off.
        ecg_slope_mode: Frame-level ECG slope features: none, fast, slow, or
            fast_slow.
        ecg_slope_slow_window_samples: Trailing raw samples in the slow slope fit.
        signal_quality_mode: Optional signal-quality feature: none, ecg, or ecg_puls.
        ecg_rotation_angle: ECG1/ECG2 rotation angle in radians.
        puls_offset: Offset subtracted from pulse before pulse normalization.
        puls_quantile_low: Lower causal rolling pulse quantile.
        puls_quantile_high: Upper causal rolling pulse quantile.
        puls_norm_window_frames: Pulse normalization history length at 50 Hz.
        puls_norm_warmup_value: Pulse output before a usable quantile range exists.
        puls_clip: Whether pulse normalization clips to [-0.1, 1.1].
        align_to_puls_blocks: Require pulse to be constant across each raw block.
        verify_puls_constant_within_block: Raise if a completed block is not
            pulse-constant.
        min_distance: Evaluation minimum distance between predicted events.
        error_association_frames: Maximum distance used only to associate a
            non-exact prediction with its target as one position error.
        threshold_objective: Threshold policy: min_error or conservative min_fp.
        threshold_sensitivity_efficiency: Fraction of the best attainable
            exact-frame sensitivity retained as the selection floor.
        use_amp: Use mixed precision where supported.
        save_last: Save the final epoch weights to `models/<modelname>/last.pth`.
        save_prepped: Save the preprocessed model-rate feature rows to parquet.
        use_prepped: Load preprocessed model-rate feature rows instead of rebuilding
            them from raw-rate parquet files.
        prepped_file: Path used by `save_prepped` or `use_prepped`. Defaults to
            `models/<modelname>/prepped.parquet`.
        prep_workers: Number of worker processes for raw parquet preprocessing.
            Use 0 to use all available CPU cores. Ignored with `use_prepped`.
        debug_timing: Print elapsed timings between major training phases.
        plots: Write report plots. None defaults off on Jetson and on elsewhere.
    """
    explicit_values = locals().copy()
    config_values = _load_training_config_values(config)
    kwargs = _training_kwargs_from_config_and_values(
        explicit_values,
        config_values=config_values,
    )
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
    ecg3_col = kwargs["ecg3_col"]
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
    ecg3_offset = float(kwargs["ecg3_offset"])
    ecg_slope_mode = str(kwargs["ecg_slope_mode"])
    ecg_slope_slow_window_samples = int(kwargs["ecg_slope_slow_window_samples"])
    signal_quality_mode = str(kwargs["signal_quality_mode"])
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
    from dkfz.live.prediction_contract import check_training_destination

    check_training_destination(Path("models") / modelname, kwargs)
    model_dir = ensure_dir(Path("models") / modelname)
    ensure_dir(model_dir / "metrics")

    if prepped_file is None:
        prepped_file = default_prepped_file(modelname)
    timing.mark("initialized training paths")
    derived_config_values = {
        "model_dir": str(model_dir),
        "include_puls": not not_use_puls,
    }
    if prepped_file != kwargs.get("prepped_file"):
        derived_config_values["prepped_file"] = str(prepped_file)
    print_resolved_training_configuration(
        kwargs=kwargs,
        defaults=TRAINING_DEFAULTS,
        explicit_values=explicit_values,
        config_values=config_values,
        derived_values=derived_config_values,
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
    print(f"{feature_cols = }")
    print(f"{input_dim = }, {window_size = }, {hidden_dim = }, {num_layers = }")
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
        recovery_head=bool(kwargs.get("recovery_head", False)),
        prediction_delay_frames=int(kwargs.get("prediction_delay_frames", 0)),
    )

    criterion = nn.BCEWithLogitsLoss()
    if kwargs.get("recovery_only"):
        raise ValueError(
            "Use run/train_finetune.py --recovery-only with an existing checkpoint"
        )
    timing.mark("initialized model")
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
    )
    timing.mark("trained model")

    saved_model = build_ecg_rnn(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        device=device,
        checkpoint_path=model_dir / "best.pth",
        recovery_head=bool(kwargs.get("recovery_head", False)),
        prediction_delay_frames=int(kwargs.get("prediction_delay_frames", 0)),
    )
    timing.mark("loaded best checkpoint")

    threshold_selection = evaluate_and_write_live_artifacts(
        model=saved_model,
        recovery_max_fp_increase=float(kwargs.get("recovery_max_fp_increase", 0.10)),
        recovery_min_extra_fp_per_minute=float(
            kwargs.get("recovery_min_extra_fp_per_minute", 0.5)
        ),
        loader=val_loader,
        dataset=val_dataset,
        model_dir=model_dir,
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
        plots=kwargs["plots"],
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
            "train_all_events_file": str(model_dir / "train_all_events.parquet"),
            "train_trigger_events_file": str(
                model_dir / "train_trigger_events.parquet"
            ),
            "train_metrics_file": str(model_dir / "metrics/train_metrics.json"),
            "train_threshold_search_file": str(
                model_dir / "metrics/train_threshold_search.csv"
            ),
        }
    )
    if metadata["plots_enabled"]:
        metadata.update(
            train_threshold_dashboard_file=str(
                model_dir / "metrics/train_threshold_dashboard.pdf"
            ),
            train_threshold_error_examples_file=str(
                model_dir / "metrics/train_threshold_errors.pdf"
            ),
        )
    write_metadata(model_dir / "train_config.json", metadata)
    timing.mark("wrote training config")

    print(f"Selected validation threshold: {best_threshold:.6f}")
    print(
        "Live training took",
        chop_microseconds(timedelta(seconds=time.time() - start_time)),
        "to run.",
    )


if __name__ == "__main__":
    # CLI wrapper around `live_training`. Defaults are chosen for 400 Hz raw
    # data framed to 50 Hz with 8 samples per frame.
    parser = argparse.ArgumentParser()
    from dkfz.live.training_config import add_prediction_training_arguments

    add_prediction_training_arguments(parser)
    parser.add_argument(
        "inputs", nargs="*", help="Parquet files, directories, or globs."
    )
    parser.add_argument(
        "--threshold-inputs",
        nargs="+",
        default=None,
        help=(
            "Disjoint parquet files, directories, or globs used for best "
            "checkpoint and threshold selection. Defaults to data/live/sp*/*."
        ),
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Training config JSON."
    )
    parser.add_argument("--modelname", type=str, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for Python, NumPy, PyTorch, and CUDA. Defaults to 42.",
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
    parser.add_argument("--target-col", type=str, default=None)
    parser.add_argument("--ecg1-col", type=str, default=None)
    parser.add_argument("--ecg2-col", type=str, default=None)
    parser.add_argument("--ecg3-col", type=str, default=None)
    parser.add_argument("--puls-col", type=str, default=None)
    parser.add_argument("--not-use-puls", action="store_true", default=None)
    parser.add_argument("--use-puls", action="store_false", dest="not_use_puls")
    parser.add_argument("--time-col", type=str, default=None)
    parser.add_argument("--time-unit", type=str, choices=["ms", "s"], default=None)
    parser.add_argument("--raw-hz", type=float, default=None)
    parser.add_argument("--frame-size-raw", type=int, default=None)
    parser.add_argument(
        "--target-reduce",
        type=str,
        choices=["max", "last", "center", "mean"],
        default=None,
    )
    parser.add_argument("--ecg-quantile", type=float, default=None)
    parser.add_argument("--ecg-norm-window-samples", type=int, default=None)
    parser.add_argument("--ecg1-offset", type=float, default=None)
    parser.add_argument("--ecg2-offset", type=float, default=None)
    parser.add_argument("--ecg3-offset", type=float, default=None)
    parser.add_argument(
        "--ecg-slope-mode",
        choices=("none", "fast", "slow", "fast_slow"),
        default=None,
    )
    parser.add_argument(
        "--ecg-slope-slow-window-samples",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--signal-quality-mode",
        choices=("none", "ecg", "ecg_puls"),
        default=None,
        help=(
            "Add one causal signal-quality feature. 'ecg_puls' uses pulse when "
            "a raw pulse column is available, otherwise it falls back to ECG-only."
        ),
    )
    parser.add_argument(
        "--ecg-rotation-angle",
        type=float,
        default=None,
        help="Rotate ECG1/ECG2 by this angle in radians before normalization.",
    )
    parser.add_argument("--puls-offset", type=float, default=None)
    parser.add_argument("--puls-quantile-low", type=float, default=None)
    parser.add_argument("--puls-quantile-high", type=float, default=None)
    parser.add_argument("--puls-norm-window-frames", type=int, default=None)
    parser.add_argument("--puls-norm-warmup-value", type=float, default=None)
    parser.add_argument("--puls-clip", action="store_true", dest="puls_clip")
    parser.add_argument("--puls-no-clip", action="store_false", dest="puls_clip")
    parser.set_defaults(puls_clip=None)
    parser.add_argument(
        "--puls-block-alignment",
        action="store_true",
        dest="align_to_puls_blocks",
    )
    parser.add_argument(
        "--no-puls-block-alignment",
        action="store_false",
        dest="align_to_puls_blocks",
    )
    parser.set_defaults(align_to_puls_blocks=None)
    parser.add_argument(
        "--verify-puls-constant-within-block",
        action="store_true",
        default=None,
    )
    parser.add_argument(
        "--no-verify-puls-constant-within-block",
        action="store_false",
        dest="verify_puls_constant_within_block",
    )
    parser.add_argument("--min-distance", type=int, default=None)
    parser.add_argument(
        "--error-association-frames",
        type=int,
        default=None,
        help=(
            "Maximum distance used only to attribute a non-exact prediction as "
            "one position error. Exact-match tolerance is always zero."
        ),
    )
    parser.add_argument(
        "--threshold-objective",
        choices=("min_error", "min_fp"),
        default=None,
        help=(
            "Threshold selection policy. min_error prioritizes unmatched false "
            "positives, then total unique errors; min_fp prioritizes unmatched "
            "false positives and missed beats."
        ),
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
        help="Save final epoch weights to models/<modelname>/last.pth.",
    )
    parser.add_argument(
        "--no-save-last",
        action="store_false",
        dest="save_last",
    )
    parser.add_argument(
        "--save-prepped",
        action="store_true",
        dest="save_prepped",
        default=None,
        help="Save preprocessed model-rate feature frames to parquet.",
    )
    parser.add_argument(
        "--no-save-prepped",
        action="store_false",
        dest="save_prepped",
    )
    parser.add_argument(
        "--use-prepped",
        action="store_true",
        dest="use_prepped",
        default=None,
        help="Load preprocessed model-rate feature frames instead of rebuilding them.",
    )
    parser.add_argument(
        "--no-use-prepped",
        action="store_false",
        dest="use_prepped",
    )
    parser.add_argument(
        "--prepped-file",
        type=str,
        default=None,
        help="Path for --save-prepped/--use-prepped. Defaults to models/<modelname>/prepped.parquet.",
    )
    parser.add_argument(
        "-j",
        "--prep-workers",
        type=int,
        default=None,
        help=(
            "Worker processes for raw parquet preprocessing. "
            "Use 0 for all available CPU cores. Default keeps serial behavior."
        ),
    )
    parser.add_argument(
        "--debug-timing",
        action="store_true",
        dest="debug_timing",
        default=None,
        help="Print elapsed timings between major training phases.",
    )
    parser.add_argument(
        "--no-debug-timing",
        action="store_false",
        dest="debug_timing",
    )

    from dkfz.live.training_config import add_recovery_training_arguments

    add_recovery_training_arguments(parser, fine_tune=False)
    args = parser.parse_args()
    live_training(**vars(args))
