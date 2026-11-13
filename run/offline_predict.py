# Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Developed at the German Cancer Research Center (DKFZ).

"""Run live-style ECG trigger prediction on a parquet file.

This script loads the model and live preprocessing config produced by
`run/train.py`, streams a parquet file through the same feature framing path,
and writes two outputs:

- event parquet: one row per emitted trigger
- prediction parquet: one row per evaluated model-rate window, including
  `prob`, peak-event `pred`, optional `target`/`siemens_tr`, and input features

Use ``--skip-metrics`` to save predictions without computing or writing
evaluation statistics, even when the input contains annotations.

The trained model architecture and preprocessing contract are read from the
selected model variant config: `models/<modelname>/train_config.json` for
`best`, or `models/<modelname>/tuning/tuned_config.json` for `tuned`. Use
``--model-dir`` to load the same artifact layout from another model root.
Runtime CLI arguments can override paths, threshold/holdoff, and input
column/time mapping.

``--synthetic-gradient`` adds independent notebook noise to ECG1/ECG2 and
stores them in ecg1_syn/ecg2_syn, then derives ecg3_syn = ecg2_syn - ecg1_syn
before live preprocessing. Predictions use these columns; raw ECGs are kept. The
seed 42 is fixed; the sample rate comes from the model config.
``--synthetic-main-frequency-hz`` shifts peaks relative to the main frequency
and aliases out-of-band components. ``--save-synthetic`` additionally writes the modified
input to ``data/synthetic/[main_<frequency>hz/]pNN/<stem>_synthetic.parquet``.
The input file is preserved. Default prediction output names gain
``_synthetic``; explicit output paths are used as supplied.
"""

import argparse
import json
from pathlib import Path
import sys
import time
from datetime import timedelta

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dkfz.live import (
    LiveLSTMPredictor,
    default_torch_device,
)
from dkfz.live.eval_metrics import (
    build_prediction_metrics_report,
    format_prediction_metrics_report,
    select_target_column,
    write_prediction_metrics_report,
)
from dkfz.live.io import repo_relative_path, require_raw_signal_column
from dkfz.live.model_variants import MODEL_VARIANT_CHOICES, get_model_variant
from dkfz.live.prediction_runtime import (
    default_prediction_metrics_path,
    events_to_dataframe,
    input_output_prefix,
    load_config_contract,
    make_framer_from_settings,
    print_resolved_prediction_configuration,
    resolve_prediction_inputs,
    resolve_config_and_runtime_values,
    resolve_prediction_columns,
    write_prediction_parquets,
)
from dkfz.live.prediction_logs import build_prediction_log
from dkfz.live.runtime_config import effective_recovery_threshold
from dkfz.core.model import ECG_RNN
from dkfz.core.gradient_noise import (
    GRADIENT_MAIN_FREQUENCIES_HZ,
    SYNTHETIC_ECG_COLUMNS,
    apply_synthetic_gradient,
    gradient_config_id,
    save_synthetic_parquet,
)
from dkfz.core.utils import chop_microseconds

sys.stdout.reconfigure(line_buffering=True, write_through=True)

PREDICTION_RUNTIME_DEFAULTS = {
    "out_events": None,
    "out_log": None,
    "include_input_name_in_outputs": True,
    "threshold": None,
    "modelname": "target_r",
    "model_variant": "best",
    "model_dir": None,
    "holdoff_frames": None,
    "holdoff_ms": 200.0,
    "error_association_frames": 1,
    "ecg1_col": "ecg1_raw",
    "ecg2_col": "ecg2_raw",
    "ecg3_col": None,
    "ecg_slope_mode": "none",
    "ecg_slope_slow_window_samples": 40,
    "signal_quality_mode": "none",
    "signal_quality_stats": None,
    "puls_col": "puls_raw",
    "target_col": None,
    "time_col": "time_ms",
    "time_unit": "ms",
    "out_metrics": None,
    "skip_metrics": False,
    "synthetic_gradient": False,
    "synthetic_main_frequency_hz": None,
    "save_synthetic": False,
}

PREDICTION_CONFIG_REQUIRED_KEYS = {
    "input_dim",
    "hidden_dim",
    "num_layers",
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
}

PREDICTION_CONFIG_OPTIONAL_KEYS = {
    "feature_cols",
    "preprocessing_contract_version",
    "prediction_mode",
    "prediction_delay_frames",
    "signal_quality_puls_flat_window_frames",
    "signal_quality_puls_median_window_frames",
    "recovery_head",
    "recovery_enabled",
    "recovery_threshold",
    "threshold",
    "holdoff_frames",
    "error_association_frames",
    "target_col",
    "verify_puls_constant_within_block",
    "ecg3_offset",
    "ecg_slope_mode",
    "ecg_slope_slow_window_samples",
    "signal_quality_mode",
    "signal_quality_stats",
}


def _prediction_config_resolution_from_values(values: dict):
    """Resolve trained config contract values and runtime overrides.

    Model architecture and preprocessing values come from the training config
    only. Explicit arguments override runtime-safe values such as output paths,
    threshold/holdoff, and input column/time mapping.

    Training configs store `include_puls`, while prediction internally uses
    `not_use_puls`, so that value is translated here.
    """
    modelname = values.get("modelname") or PREDICTION_RUNTIME_DEFAULTS["modelname"]
    model_variant = (
        values.get("model_variant") or PREDICTION_RUNTIME_DEFAULTS["model_variant"]
    )
    variant = get_model_variant(model_variant)
    model_variant = variant.name
    requested_model_dir = values.get("model_dir")
    default_cfg_path = variant.config_path(modelname, requested_model_dir)
    cfg_path = Path(values.get("config") or default_cfg_path)
    config_keys = PREDICTION_CONFIG_REQUIRED_KEYS | PREDICTION_CONFIG_OPTIONAL_KEYS
    cfg = load_config_contract(
        cfg_path,
        required_keys=PREDICTION_CONFIG_REQUIRED_KEYS,
        error_label="prediction",
    )
    resolution = resolve_config_and_runtime_values(
        cfg=cfg,
        values=values,
        defaults=PREDICTION_RUNTIME_DEFAULTS,
        config_keys=config_keys,
        column_keys=(
            "ecg1_col",
            "ecg2_col",
            "ecg3_col",
            "puls_col",
            "time_col",
            "time_unit",
        ),
    )
    kwargs = resolution.values
    from dkfz.live.prediction_contract import prediction_delay

    prediction_delay(kwargs)
    from dkfz.live.event_matching import event_tolerances

    metric_policy = event_tolerances(
        cfg, error_association_frames=values.get("error_association_frames")
    )
    kwargs.update(metric_policy)
    resolution.derived_values.update(metric_policy)
    kwargs["recovery_threshold"] = effective_recovery_threshold(
        kwargs, disable_recovery=bool(values.get("disable_recovery", False))
    )
    resolution.derived_values["recovery_threshold"] = kwargs["recovery_threshold"]

    model_dir = variant.model_dir(str(kwargs["modelname"]), requested_model_dir)
    kwargs["model_dir"] = str(model_dir)
    kwargs["model_variant"] = model_variant
    kwargs["model_path"] = str(
        variant.checkpoint_path(str(kwargs["modelname"]), model_dir)
    )
    kwargs["config"] = str(cfg_path)
    kwargs["in_file"] = values["in_file"]
    kwargs["verify_puls_constant_within_block"] = bool(
        kwargs.get("verify_puls_constant_within_block", False)
    )
    output_prefix = input_output_prefix(values, kwargs)
    if kwargs["synthetic_gradient"]:
        output_prefix += "_synthetic"
        if kwargs["synthetic_main_frequency_hz"] is not None:
            output_prefix += "_" + gradient_config_id(
                kwargs["synthetic_main_frequency_hz"]
            )
    output_suffix = variant.output_suffix
    if kwargs["out_events"] is None:
        kwargs["out_events"] = str(
            model_dir
            / f"prediction/prediction_trigger_events{output_suffix}{output_prefix}.parquet"
        )
    if kwargs["out_log"] is None:
        kwargs["out_log"] = str(
            model_dir
            / f"prediction/prediction_all_events{output_suffix}{output_prefix}.parquet"
        )
    if kwargs["out_metrics"] is None:
        kwargs["out_metrics"] = default_prediction_metrics_path(
            model_dir,
            kwargs["out_log"],
        )
    derived_values = {
        "model_path": kwargs["model_path"],
        "include_puls": not bool(kwargs["not_use_puls"]),
    }
    if values.get("config") is None:
        derived_values["config"] = kwargs["config"]
    if values.get("out_events") is None:
        derived_values["out_events"] = kwargs["out_events"]
    if values.get("out_log") is None:
        derived_values["out_log"] = kwargs["out_log"]
    if values.get("out_metrics") is None:
        derived_values["out_metrics"] = kwargs["out_metrics"]
    resolution.derived_values.update(derived_values)
    return resolution


def _live_predict_one(
    in_file: str,
    out_events: str | None = None,
    out_log: str | None = None,
    include_input_name_in_outputs: bool | None = None,
    threshold: float | None = None,
    modelname: str | None = None,
    model_variant: str | None = None,
    model_dir: str | None = None,
    config: str | None = None,
    holdoff_frames: int | None = None,
    holdoff_ms: float | None = None,
    error_association_frames: int | None = None,
    ecg1_col: str | None = None,
    ecg2_col: str | None = None,
    ecg3_col: str | None = None,
    puls_col: str | None = None,
    target_col: str | None = None,
    time_col: str | None = None,
    time_unit: str | None = None,
    out_metrics: str | None = None,
    synthetic_gradient: bool = False,
    synthetic_main_frequency_hz: float | None = None,
    save_synthetic: bool = False,
    skip_metrics: bool = False,
    disable_recovery: bool = False,
):
    """Run a trained live ECG model over one parquet file.

    Args:
        in_file: Input parquet containing raw-rate ECG and optional pulse columns.
        out_events: Output parquet path for emitted trigger events.
        out_log: Output parquet path for the prediction log.
        include_input_name_in_outputs: Prefix default output filenames with the
            input file stem. Explicit `out_events` and `out_log` paths are used
            as-is.
        threshold: Trigger threshold. If omitted, it is read from config.
        modelname: Name used to locate model/config files.
        model_variant: Model variant to load, either `best` or `tuned`.
            Defaults to `best`. The variant also selects the default config path
            and therefore the matching threshold.
        model_dir: Optional directory containing this model's checkpoint/config
            pair. Defaults to ``models/<modelname>``.
        config: Optional explicit config JSON path.
        holdoff_frames: Holdoff duration in frames. Overrides `holdoff_ms`.
        holdoff_ms: Holdoff duration in milliseconds if `holdoff_frames` is not
            supplied.
        error_association_frames: Maximum distance used only to associate a
            non-exact prediction as one position error.
        ecg1_col: ECG1 input column.
        ecg2_col: ECG2 input column.
        ecg3_col: Optional measured ECG3 input column.
        puls_col: Pulse input column. Ignored when pulse is disabled.
        target_col: Optional target column used for evaluation statistics.
        time_col: Optional time column used for event/log times.
        time_unit: Unit of `time_col`, either `ms` or `s`.
        out_metrics: Optional text path for evaluation metrics. When omitted
            and a target column is available, the report is written under
            the selected model directory's `metrics` subdirectory.
        skip_metrics: Skip evaluation statistics even when annotations exist.
        disable_recovery: Emit events from the current output only, retaining
            the checkpoint's two-output shape and saved recovery calibration.
        synthetic_gradient: Apply the notebook's gradient noise to ECG1/ECG2
            before preprocessing, with seed 42, then derive
            ecg3_syn = ecg2_syn - ecg1_syn. Predictions consume ecgN_syn columns
            while original ECG columns and the model's input count stay unchanged.
        synthetic_main_frequency_hz: Optional main peak frequency; TR_ms is
            4000 / frequency. Moves dependent peaks and simulates aliasing.
        save_synthetic: Also save the modified recording, including derived ECG3,
            as data/synthetic/pNN/<stem>_synthetic.parquet, preserving the source.
            Requires synthetic_gradient.
    """
    if save_synthetic and not synthetic_gradient:
        raise ValueError("--save-synthetic requires --synthetic-gradient")
    if synthetic_main_frequency_hz is not None and not synthetic_gradient:
        raise ValueError("--synthetic-main-frequency-hz requires --synthetic-gradient")
    resolution = _prediction_config_resolution_from_values(locals())
    kwargs = resolution.values
    out_events = kwargs["out_events"]
    out_log = kwargs["out_log"]
    threshold = kwargs["threshold"]
    modelname = kwargs["modelname"]
    model_variant = kwargs["model_variant"]
    model_path = Path(kwargs["model_path"])
    input_dim = kwargs["input_dim"]
    hidden_dim = kwargs["hidden_dim"]
    num_layers = kwargs["num_layers"]
    window_size = kwargs["window_size"]
    holdoff_frames = kwargs["holdoff_frames"]
    holdoff_ms = kwargs["holdoff_ms"]
    error_association_frames = int(kwargs["error_association_frames"])
    if error_association_frames < 0:
        raise ValueError("error_association_frames must be zero or positive.")
    ecg1_col = kwargs["ecg1_col"]
    ecg2_col = kwargs["ecg2_col"]
    ecg3_col = kwargs["ecg3_col"]
    puls_col = kwargs["puls_col"]
    target_col = kwargs["target_col"]
    not_use_puls = kwargs["not_use_puls"]
    time_col = kwargs["time_col"]
    time_unit = kwargs["time_unit"]
    raw_hz = float(kwargs["raw_hz"])
    frame_size_raw = int(kwargs["frame_size_raw"])
    ecg_quantile = float(kwargs["ecg_quantile"])
    ecg_norm_window_samples = int(kwargs["ecg_norm_window_samples"])
    ecg1_offset = float(kwargs["ecg1_offset"])
    ecg2_offset = float(kwargs["ecg2_offset"])
    ecg_rotation_angle = float(kwargs["ecg_rotation_angle"])
    puls_offset = float(kwargs["puls_offset"])
    puls_clip = bool(kwargs["puls_clip"])
    align_to_puls_blocks = bool(kwargs["align_to_puls_blocks"])
    verify_puls_constant_within_block = bool(
        kwargs["verify_puls_constant_within_block"]
    )
    out_metrics = kwargs["out_metrics"]

    start_time = time.time()
    include_puls = not not_use_puls
    frame_hz = raw_hz / frame_size_raw

    if threshold is None:
        raise ValueError("Provide --threshold or a config containing a threshold.")

    hidden_dim = int(hidden_dim)
    num_layers = int(num_layers)
    window_size = int(window_size)
    input_dim = int(input_dim)
    threshold = float(threshold)

    if holdoff_frames is None:
        holdoff_frames = LiveLSTMPredictor.holdoff_frames_from_ms(
            holdoff_ms, frame_hz=frame_hz
        )
        resolution.derived_values["holdoff_frames"] = holdoff_frames
    kwargs["holdoff_frames"] = holdoff_frames
    print_resolved_prediction_configuration(resolution)

    df = pd.read_parquet(in_file)
    if synthetic_gradient:
        require_raw_signal_column(df, ecg1_col, "ECG1")
        require_raw_signal_column(df, ecg2_col, "ECG2")
        df = apply_synthetic_gradient(
            df,
            columns=(ecg1_col, ecg2_col),
            fs=raw_hz,
            main_frequency_hz=synthetic_main_frequency_hz,
        )
        ecg1_col, ecg2_col = SYNTHETIC_ECG_COLUMNS[:2]
        ecg3_col = SYNTHETIC_ECG_COLUMNS[2] if ecg3_col is not None else None
    (
        actual_ecg1_col,
        actual_ecg2_col,
        actual_ecg3_col,
        actual_puls_col,
    ) = resolve_prediction_columns(
        df,
        ecg1_col=ecg1_col,
        ecg2_col=ecg2_col,
        ecg3_col=ecg3_col,
        puls_col=puls_col,
        include_puls=include_puls,
        optional_puls=str(kwargs.get("signal_quality_mode", "none")) == "ecg_puls",
    )

    if synthetic_gradient:
        print(
            "Applied notebook synthetic gradient noise "
            f"to {actual_ecg1_col}, {actual_ecg2_col} (seed=42, fs={raw_hz:g} Hz); "
            "derived ecg3_syn = ecg2_syn - ecg1_syn; original ECG columns retained"
        )
        if synthetic_main_frequency_hz is not None:
            print(
                "Synthetic gradient configuration: "
                + json.dumps(df.attrs["synthetic_gradient"])
            )
        if save_synthetic:
            synthetic_path = save_synthetic_parquet(
                df, in_file, ROOT / "data" / "synthetic"
            )
            print(f"Wrote synthetic input to {repo_relative_path(synthetic_path)}")

    device = default_torch_device()
    model = ECG_RNN(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        recovery_head=bool(kwargs.get("recovery_head", False)),
        prediction_delay_frames=int(kwargs.get("prediction_delay_frames", 0)),
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(
        f"Loaded {model_variant} checkpoint from {repo_relative_path(model_path)} "
        f"with config {repo_relative_path(kwargs['config'])}"
    )

    framer = make_framer_from_settings(kwargs)

    predictor = LiveLSTMPredictor(
        model=model,
        window_size=window_size,
        threshold=float(threshold),
        holdoff_frames=int(holdoff_frames),
        device=device,
        model_output_is_probability=False,
        framer=framer,
        recovery_threshold=kwargs.get("recovery_threshold"),
    )

    events, log_df = predictor.run_dataframe(
        df=df,
        ecg1_col=actual_ecg1_col,
        ecg2_col=actual_ecg2_col,
        ecg3_col=actual_ecg3_col,
        puls_col=actual_puls_col,
        time_col=time_col,
        time_unit=time_unit,
    )

    log_df = build_prediction_log(
        log_df=log_df,
        events=events,
        raw_df=df,
        feature_cols=predictor.framer.feature_cols,
        reduced_raw_cols=[target_col] if target_col else None,
    )
    resolved_target_col = select_target_column(log_df, preferred=target_col)
    events_df = events_to_dataframe(events)

    write_prediction_parquets(
        events_df=events_df,
        log_df=log_df,
        out_events=out_events,
        out_log=out_log,
    )

    print(
        f"Wrote {len(events_df):_} trigger events to {repo_relative_path(out_events)}"
    )
    print(f"Wrote {len(log_df):_} prediction rows to {repo_relative_path(out_log)}")

    if skip_metrics:
        print("Skipped prediction metrics (--skip-metrics).")
    elif resolved_target_col is not None:
        metrics = build_prediction_metrics_report(
            prediction_df=log_df,
            target_col=resolved_target_col,
            error_association_frames=error_association_frames,
            exact_tolerance_frames=int(kwargs["exact_tolerance_frames"]),
            outer_error_association_frames=int(
                kwargs["outer_error_association_frames"]
            ),
            threshold=float(threshold),
            events_df=events_df,
        )
        if predictor.prediction_delay_frames:
            from dkfz.live.raw_event_evaluation import evaluate_raw_events

            metrics["raw_events"] = evaluate_raw_events(
                df,
                events_df,
                log_df,
                frame_size_raw=int(kwargs["frame_size_raw"]),
                target_col=resolved_target_col,
                raw_hz=float(kwargs["raw_hz"]),
                time_col=time_col,
                time_unit=time_unit,
            )
        metrics.update(
            recovery_head=bool(kwargs.get("recovery_head", False)),
            recovery_threshold=kwargs.get("recovery_threshold"),
        )
        report_text = format_prediction_metrics_report(metrics)
        text_path = write_prediction_metrics_report(
            report_text,
            text_path=out_metrics,
            metrics=metrics,
        )
        print(f"Wrote prediction metrics report to {repo_relative_path(text_path)}")
        print(report_text)
    else:
        print("No target column found in the input; skipped prediction metrics.")

    print(
        "Live prediction took",
        chop_microseconds(timedelta(seconds=time.time() - start_time)),
        "to run.",
    )


def live_predict(
    inputs: list[str],
    out_events: str | None = None,
    out_log: str | None = None,
    include_input_name_in_outputs: bool | None = None,
    threshold: float | None = None,
    modelname: str | None = None,
    model_variant: str | None = None,
    model_dir: str | None = None,
    config: str | None = None,
    holdoff_frames: int | None = None,
    holdoff_ms: float | None = None,
    error_association_frames: int | None = None,
    ecg1_col: str | None = None,
    ecg2_col: str | None = None,
    ecg3_col: str | None = None,
    puls_col: str | None = None,
    target_col: str | None = None,
    time_col: str | None = None,
    time_unit: str | None = None,
    out_metrics: str | None = None,
    synthetic_gradient: bool = False,
    synthetic_main_frequency_hz: float | None = None,
    save_synthetic: bool = False,
    skip_metrics: bool = False,
    disable_recovery: bool = False,
):
    """Run a trained live ECG model over parquet files, directories, or globs."""
    input_paths, multiple_inputs = resolve_prediction_inputs(
        inputs,
        out_events=out_events,
        out_log=out_log,
        out_metrics=out_metrics,
    )
    if multiple_inputs:
        include_input_name_in_outputs = True

    for input_path in input_paths:
        _live_predict_one(
            in_file=str(input_path),
            out_events=out_events,
            out_log=out_log,
            include_input_name_in_outputs=include_input_name_in_outputs,
            threshold=threshold,
            modelname=modelname,
            model_variant=model_variant,
            model_dir=model_dir,
            config=config,
            holdoff_frames=holdoff_frames,
            holdoff_ms=holdoff_ms,
            error_association_frames=error_association_frames,
            ecg1_col=ecg1_col,
            ecg2_col=ecg2_col,
            ecg3_col=ecg3_col,
            puls_col=puls_col,
            target_col=target_col,
            time_col=time_col,
            time_unit=time_unit,
            out_metrics=out_metrics,
            synthetic_gradient=synthetic_gradient,
            synthetic_main_frequency_hz=synthetic_main_frequency_hz,
            save_synthetic=save_synthetic,
            skip_metrics=skip_metrics,
            disable_recovery=disable_recovery,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Parquet files, directories, or globs.",
    )
    parser.add_argument("--out-events", default=None)
    parser.add_argument("--out-log", default=None)
    parser.add_argument(
        "--include-input-name-in-outputs",
        action="store_true",
        dest="include_input_name_in_outputs",
        help="Prefix default output parquet filenames with the input file stem.",
    )
    parser.add_argument(
        "--no-include-input-name-in-outputs",
        action="store_false",
        dest="include_input_name_in_outputs",
    )
    parser.set_defaults(include_input_name_in_outputs=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument(
        "--no-recovery",
        dest="disable_recovery",
        action="store_true",
        help="Use only the current output of a two-output model; keep saved calibration.",
    )
    parser.add_argument("--modelname", type=str, default=None)
    parser.add_argument(
        "--model-variant",
        choices=MODEL_VARIANT_CHOICES,
        default=None,
        help=(
            "Checkpoint/config pair to use. 'best' loads best.pth with "
            "train_config.json; 'tuned' loads best_tuned.pth with "
            "tuning/tuned_config.json."
        ),
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help=(
            "Directory containing the selected checkpoint/config pair. "
            "Defaults to models/<modelname>."
        ),
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--holdoff-frames", type=int, default=None)
    parser.add_argument("--holdoff-ms", type=float, default=None)
    parser.add_argument("--error-association-frames", type=int, default=None)
    parser.add_argument("--ecg1-col", type=str, default=None)
    parser.add_argument("--ecg2-col", type=str, default=None)
    parser.add_argument("--ecg3-col", type=str, default=None)
    parser.add_argument("--puls-col", type=str, default=None)
    parser.add_argument("--target-col", type=str, default=None)
    parser.add_argument("--time-col", type=str, default=None)
    parser.add_argument("--time-unit", type=str, choices=["ms", "s"], default=None)
    parser.add_argument("--out-metrics", type=str, default=None)
    parser.add_argument(
        "--skip-metrics",
        action="store_true",
        help="Save prediction parquets without computing or writing evaluation statistics.",
    )
    parser.add_argument(
        "--synthetic-gradient",
        action="store_true",
        help=(
            "Add independent notebook gradient noise to ECG1/ECG2 (seed 42), "
            "predict from ecg1_syn/ecg2_syn and derived ecg3_syn, keeping raw ECGs."
        ),
    )
    parser.add_argument(
        "--synthetic-main-frequency-hz",
        type=float,
        choices=GRADIENT_MAIN_FREQUENCIES_HZ,
        default=None,
        help="Main gradient frequency; TR_ms = 4000 / frequency. Requires --synthetic-gradient.",
    )
    parser.add_argument(
        "--save-synthetic",
        action="store_true",
        help=(
            "Save raw ECGs plus ecg1_syn/ecg2_syn/ecg3_syn to "
            "data/synthetic/[main_<frequency>hz/]pNN/<stem>_synthetic.parquet; "
            "requires --synthetic-gradient."
        ),
    )

    live_predict(**vars(parser.parse_args()))
