# Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Developed at the German Cancer Research Center (DKFZ).

"""Run live-style ECG trigger prediction using a TensorRT engine."""

from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path
import sys
import time

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dkfz.live.eval_metrics import (
    build_prediction_metrics_report,
    format_prediction_metrics_report,
    select_target_column,
    write_prediction_metrics_report,
)
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
from dkfz.live.io import repo_relative_path
from dkfz.live.model_variants import MODEL_VARIANT_CHOICES, get_model_variant
from dkfz.live.prediction_logs import build_prediction_log
from dkfz.live.runtime_config import (
    effective_recovery_threshold,
    holdoff_frames_from_settings,
)
from dkfz.core.utils import ensure_dir

TRT_RUNTIME_DEFAULTS = {
    "engine": None,
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
    "trt_batch_size": 1,
}

TRT_CONFIG_REQUIRED_KEYS = {
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
}

TRT_CONFIG_OPTIONAL_KEYS = {
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


def _chop_microseconds(delta: timedelta) -> timedelta:
    return delta - timedelta(microseconds=delta.microseconds)


def _trt_prediction_config_resolution_from_values(values: dict):
    """Resolve TensorRT config contract values and runtime overrides.

    Model shape and preprocessing values come from the training config only.
    Explicit arguments only override runtime-safe options such as paths,
    threshold/holdoff, and input column/time mapping.
    """
    modelname = values.get("modelname") or TRT_RUNTIME_DEFAULTS["modelname"]
    variant = get_model_variant(
        values.get("model_variant") or TRT_RUNTIME_DEFAULTS["model_variant"]
    )
    model_variant = variant.name
    requested_model_dir = values.get("model_dir")
    default_cfg_path = variant.config_path(modelname, requested_model_dir)
    config_value = values.get("config")
    cfg_path = Path(config_value or default_cfg_path)
    config_keys = TRT_CONFIG_REQUIRED_KEYS | TRT_CONFIG_OPTIONAL_KEYS
    cfg = load_config_contract(
        cfg_path,
        required_keys=TRT_CONFIG_REQUIRED_KEYS,
        error_label="TensorRT prediction",
    )
    resolution = resolve_config_and_runtime_values(
        cfg=cfg,
        values=values,
        defaults=TRT_RUNTIME_DEFAULTS,
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
    ensure_dir(model_dir / "prediction")
    ensure_dir(model_dir / "metrics")
    kwargs["in_file"] = values["in_file"]
    kwargs["model_variant"] = model_variant
    kwargs["config"] = str(cfg_path)
    kwargs["verify_puls_constant_within_block"] = bool(
        kwargs.get("verify_puls_constant_within_block", False)
    )

    if kwargs["engine"] is None:
        kwargs["engine"] = str(
            variant.trt_engine_path(str(kwargs["modelname"]), requested_model_dir)
        )

    output_prefix = input_output_prefix(values, kwargs)
    output_suffix = variant.output_suffix
    if kwargs["out_events"] is None:
        kwargs["out_events"] = str(
            model_dir
            / f"prediction/trt_trigger_events{output_suffix}{output_prefix}.parquet"
        )
    if kwargs["out_log"] is None:
        kwargs["out_log"] = str(
            model_dir
            / f"prediction/trt_all_events{output_suffix}{output_prefix}.parquet"
        )
    if kwargs["out_metrics"] is None:
        kwargs["out_metrics"] = default_prediction_metrics_path(
            model_dir,
            kwargs["out_log"],
        )
    derived_values = {
        "include_puls": not bool(kwargs["not_use_puls"]),
    }
    if config_value is None:
        derived_values["config"] = kwargs["config"]
    if values.get("engine") is None:
        derived_values["engine"] = kwargs["engine"]
    if values.get("out_events") is None:
        derived_values["out_events"] = kwargs["out_events"]
    if values.get("out_log") is None:
        derived_values["out_log"] = kwargs["out_log"]
    if values.get("out_metrics") is None:
        derived_values["out_metrics"] = kwargs["out_metrics"]
    resolution.derived_values.update(derived_values)
    resolution.title = "Resolved TensorRT prediction configuration"
    return resolution


def _live_predict_trt_one(
    in_file: str,
    engine: str | None = None,
    config: str | None = None,
    modelname: str | None = None,
    model_variant: str | None = None,
    model_dir: str | None = None,
    out_events: str | None = None,
    out_log: str | None = None,
    include_input_name_in_outputs: bool | None = None,
    threshold: float | None = None,
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
    trt_batch_size: int | None = None,
    disable_recovery: bool = False,
):
    """Run a trained live ECG TensorRT engine over one parquet file."""
    resolution = _trt_prediction_config_resolution_from_values(locals())
    kwargs = resolution.values
    engine = kwargs["engine"]
    out_events = kwargs["out_events"]
    out_log = kwargs["out_log"]
    threshold = kwargs["threshold"]
    input_dim = kwargs["input_dim"]
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
    out_metrics = kwargs["out_metrics"]
    trt_batch_size = int(kwargs["trt_batch_size"])
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

    include_puls = not bool(not_use_puls)
    frame_hz = raw_hz / frame_size_raw

    if threshold is None:
        raise ValueError("Provide --threshold or a config containing a threshold.")

    window_size = int(window_size)
    input_dim = int(input_dim)
    threshold = float(threshold)
    if holdoff_frames is None:
        holdoff_frames = holdoff_frames_from_settings(kwargs)
        resolution.derived_values["holdoff_frames"] = holdoff_frames
    kwargs["holdoff_frames"] = holdoff_frames
    print_resolved_prediction_configuration(resolution)

    start_time = time.time()
    from dkfz.live.tensorrt_engine import TensorRTLogitEngine
    from dkfz.live.tensorrt_streaming_predictor import LiveTensorRTPredictor

    framer = make_framer_from_settings(kwargs)

    engine_runner = TensorRTLogitEngine(
        engine_path=Path(engine),
        input_shape=(1, window_size, input_dim),
    )
    predictor = LiveTensorRTPredictor(
        engine=engine_runner,
        window_size=window_size,
        threshold=threshold,
        holdoff_frames=int(holdoff_frames),
        model_output_is_probability=False,
        framer=framer,
        recovery_head=bool(kwargs.get("recovery_head", False)),
        prediction_delay_frames=int(kwargs.get("prediction_delay_frames", 0)),
        recovery_threshold=kwargs.get("recovery_threshold"),
    )

    df = pd.read_parquet(in_file)
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

    if trt_batch_size > 1:
        events, log_df = predictor.run_dataframe_batched(
            df=df,
            batch_size=trt_batch_size,
            ecg1_col=actual_ecg1_col,
            ecg2_col=actual_ecg2_col,
            ecg3_col=actual_ecg3_col,
            puls_col=actual_puls_col,
            time_col=time_col,
            time_unit=time_unit,
        )
    else:
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
        f"Wrote {len(events_df):_} TensorRT trigger events to "
        f"{repo_relative_path(out_events)}"
    )
    print(
        f"Wrote {len(log_df):_} TensorRT prediction rows to "
        f"{repo_relative_path(out_log)}"
    )
    if resolved_target_col is not None:
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
        print(
            f"Wrote TensorRT prediction metrics report to "
            f"{repo_relative_path(text_path)}"
        )
        print(report_text)
    else:
        print(
            "No target column found in the input; "
            "skipped TensorRT prediction metrics."
        )
    print(
        "TensorRT live prediction took",
        _chop_microseconds(timedelta(seconds=time.time() - start_time)),
        "to run.",
    )


def live_predict_trt(
    inputs: list[str],
    engine: str | None = None,
    config: str | None = None,
    modelname: str | None = None,
    model_variant: str | None = None,
    model_dir: str | None = None,
    out_events: str | None = None,
    out_log: str | None = None,
    include_input_name_in_outputs: bool | None = None,
    threshold: float | None = None,
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
    trt_batch_size: int | None = None,
    disable_recovery: bool = False,
):
    """Run a trained live ECG TensorRT engine over parquet files or globs."""
    input_paths, multiple_inputs = resolve_prediction_inputs(
        inputs,
        out_events=out_events,
        out_log=out_log,
        out_metrics=out_metrics,
    )
    if multiple_inputs:
        include_input_name_in_outputs = True

    for input_path in input_paths:
        _live_predict_trt_one(
            in_file=str(input_path),
            engine=engine,
            config=config,
            modelname=modelname,
            model_variant=model_variant,
            model_dir=model_dir,
            out_events=out_events,
            out_log=out_log,
            include_input_name_in_outputs=include_input_name_in_outputs,
            threshold=threshold,
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
            trt_batch_size=trt_batch_size,
            disable_recovery=disable_recovery,
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-recovery",
        dest="disable_recovery",
        action="store_true",
        help="Use only the current output of a two-output model; keep saved calibration.",
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Parquet files, directories, or globs.",
    )
    parser.add_argument("--engine", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--modelname", default=None)
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Exact source artifact directory; config and engine defaults are resolved here.",
    )
    parser.add_argument(
        "--model-variant",
        choices=MODEL_VARIANT_CHOICES,
        default=None,
        help=(
            "Engine/config defaults to use. 'best' loads train_config.json and "
            "<modelname>_fp16.engine; 'tuned' loads "
            "tuning/tuned_config.json and <modelname>_best_tuned_fp16.engine."
        ),
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
        "--trt-batch-size",
        type=int,
        default=None,
        help=(
            "Batch size for offline TensorRT window inference. Values >1 require "
            "an engine built with a dynamic batch profile."
        ),
    )
    args = parser.parse_args()
    live_predict_trt(**vars(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
