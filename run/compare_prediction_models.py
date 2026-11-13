# Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Developed at the German Cancer Research Center (DKFZ).
"""Evaluate saved 50/100 Hz models on the same recordings and raw-time intervals.

Thresholds are read unchanged from each model's validation-selected config.
This command never trains, selects thresholds, or builds/deploys engines.
"""

from __future__ import annotations
import argparse
import math
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import pandas as pd
from dkfz.live.prediction_fingerprint import file_digest
from dkfz.live.prediction_runtime import events_to_dataframe
from dkfz.live.prediction_logs import add_prediction_result_columns
from dkfz.live.event_matching import event_tolerances
from dkfz.live.eval_metrics import build_prediction_metrics_report
from dkfz.live.raw_event_evaluation import (
    evaluate_raw_events,
    observable_intervals,
    common_intervals,
    false_positive_changes,
    explain_misses,
    raw_timeline,
)
from dkfz.live.runtime_config import (
    make_checked_runtime_framer,
    holdoff_frames_from_settings,
    effective_recovery_threshold,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+")
    parser.add_argument(
        "--model",
        nargs=3,
        action="append",
        required=True,
        metavar=("LABEL", "CONFIG", "ARTIFACT"),
    )
    parser.add_argument("--backend", choices=("torch", "tensorrt"), default="torch")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tolerance-ms", type=float, default=20)
    parser.add_argument("--limit-seconds", type=float)
    parser.add_argument("--holdoff-ms", type=float, help="Common comparison-only physical holdoff; saved configs stay unchanged")
    parser.add_argument("--early-emission", action="store_true", help="Use the native service's higher-reference-threshold emission policy")
    args = parser.parse_args()
    if args.holdoff_ms is not None and (not math.isfinite(args.holdoff_ms) or args.holdoff_ms < 0):
        parser.error("--holdoff-ms must be finite and nonnegative")
    if len(args.model) < 2:
        parser.error("Provide at least two saved models; the first is the baseline")
    labels = [m[0] for m in args.model]
    if len(set(labels)) != len(labels) or any(
        not label
        or any(
            c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
            for c in label
        )
        for label in labels
    ):
        parser.error(
            "Model labels must be unique and contain only letters, digits, underscores or hyphens"
        )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    models = []
    for label, config, artifact in args.model:
        cfg = json.loads(Path(config).read_text())
        cfg.setdefault("not_use_puls", not cfg.get("include_puls", True))
        cfg.setdefault("verify_puls_constant_within_block", False)
        common = dict(
            window_size=int(cfg["window_size"]),
            threshold=float(cfg.get("manual_threshold") if cfg.get("manual_threshold") is not None else cfg["threshold"]),
            holdoff_frames=(holdoff_frames_from_settings(cfg) if args.holdoff_ms is None else
                math.ceil(args.holdoff_ms * float(cfg["raw_hz"]) / int(cfg["frame_size_raw"]) / 1000)),
            framer=make_checked_runtime_framer(cfg),
            recovery_threshold=effective_recovery_threshold(cfg),
        )
        if args.backend == "torch":
            import torch
            from dkfz.live.onnx_export import build_model_from_config
            from dkfz.live.torch_streaming_predictor import LiveLSTMPredictor

            torch.set_num_threads(1)
            predictor = LiveLSTMPredictor(
                build_model_from_config(cfg, artifact), device="cpu", **common
            )
        else:
            from dkfz.live.tensorrt_engine import TensorRTLogitEngine
            from dkfz.live.tensorrt_streaming_predictor import LiveTensorRTPredictor

            engine = TensorRTLogitEngine(
                engine_path=Path(artifact),
                input_shape=(1, int(cfg["window_size"]), int(cfg["input_dim"])),
            )
            predictor = LiveTensorRTPredictor(
                engine,
                recovery_head=bool(cfg.get("recovery_head", False)),
                prediction_delay_frames=int(cfg.get("prediction_delay_frames", 0)),
                **common,
            )
        if args.early_emission:
            reference = cfg.get("fp_threshold")
            predictor.trigger_logic.set_thresholds(common["threshold"], max(common["threshold"], float(cfg["threshold"] if reference is None else reference)))
        # Comparison-only settings are kept separately from the immutable config.
        metadata_policy = dict(effective_holdoff_frames=common["holdoff_frames"],
            emission_threshold=(max(common["threshold"],float(cfg["threshold"] if cfg.get("fp_threshold") is None else cfg["fp_threshold"])) if args.early_emission else common["threshold"]),
            effective_threshold=common["threshold"], early_emission=args.early_emission,
            saved_holdoff_ms=holdoff_frames_from_settings(cfg)*int(cfg["frame_size_raw"])*1000/float(cfg["raw_hz"]))
        models.append(
            (
                label,
                cfg,
                predictor,
                {
                    **metadata_policy,
                    "config": config,
                    "config_sha256": file_digest(config),
                    "artifact": artifact,
                    "artifact_sha256": file_digest(artifact),
                },
            )
        )
    raw_hz = float(models[0][1]["raw_hz"])
    if any(float(cfg["raw_hz"]) != raw_hz for _, cfg, _, _ in models):
        parser.error("Models must consume the same raw sample rate")
    reports = []
    for source in args.inputs:
        raw = pd.read_parquet(source)
        if args.limit_seconds is not None:
            raw = raw.iloc[: int(args.limit_seconds * raw_hz)].copy()
        arms = []
        for label, cfg, predictor, metadata in models:
            events, log = predictor.run_dataframe(
                raw,
                ecg1_col=cfg.get("ecg1_col", "ecg1_raw"),
                ecg2_col=cfg.get("ecg2_col", "ecg2_raw"),
                ecg3_col=cfg.get("ecg3_col"),
                puls_col=(
                    None if cfg["not_use_puls"] else cfg.get("puls_col", "puls_raw")
                ),
                time_col=cfg.get("time_col", "time_ms"),
                time_unit=cfg.get("time_unit", "ms"),
            )
            log = add_prediction_result_columns(
                log_df=log,
                events=events,
                raw_df=raw,
                reduced_raw_cols=[cfg.get("target_col", "target")],
            )
            events = events_to_dataframe(events)
            stem = Path(source).stem + "_" + label
            events.to_parquet(output / (stem + "_events.parquet"), index=False)
            log.to_parquet(output / (stem + "_probabilities.parquet"), index=False)
            arms.append((label, cfg, events, log, metadata))
        intervals = common_intervals(
            *(
                observable_intervals(log, int(cfg["frame_size_raw"]))
                for _, cfg, _, log, _ in arms
            )
        )
        result = {
            "input": source,
            "input_sha256": file_digest(source),
            "raw_samples": len(raw),
            "backend": args.backend,
            "raw_event_matching_note": "Independent raw-time diagnostic with fixed tolerance; model-frame TP/PE/oPE tiers are recorded separately.",
            "models": {},
        }
        baseline = None
        baseline_fps = None
        for label, cfg, events, log, metadata in arms:
            metrics = evaluate_raw_events(
                raw,
                events,
                log,
                frame_size_raw=int(cfg["frame_size_raw"]),
                target_col=cfg.get("target_col", "target"),
                raw_hz=raw_hz,
                intervals=intervals,
                tolerance_ms=args.tolerance_ms,
                time_col=cfg.get("time_col", "time_ms"),
                time_unit=cfg.get("time_unit", "ms"),
            )
            axis=raw_timeline(raw,raw_hz=raw_hz,time_col=cfg.get("time_col","time_ms"),time_unit=cfg.get("time_unit","ms"))
            margin=math.ceil(args.tolerance_ms*raw_hz/1000)
            duration_ms=0.0
            for sid,group in axis.groupby("sid",sort=False):
                for lo,hi in intervals.get(sid,[]):
                    included=group[group.raw_idx.between(lo+margin,hi-margin)]
                    if len(included): duration_ms+=float(included.time_ms.max()-included.time_ms.min())+1000/raw_hz
            metrics["evaluation_duration_ms"]=duration_ms
            matched = {(m["sid"], m["annotation_raw_idx"]) for m in metrics["matches"]}
            if baseline is None:
                baseline = matched
            if baseline_fps is None:
                baseline_fps = metrics["false_positive_events"]
            changes = false_positive_changes(
                baseline_fps,
                metrics["false_positive_events"],
                tolerance_samples=args.tolerance_ms * raw_hz / 1000,
            )
            metrics["miss_details"] = explain_misses(
                raw,
                metrics,
                events,
                log,
                threshold=metadata["effective_threshold"],
                holdoff_frames=metadata["effective_holdoff_frames"],
                frame_size_raw=int(cfg["frame_size_raw"]),
                raw_hz=raw_hz,
                time_col=cfg.get("time_col", "time_ms"),
                time_unit=cfg.get("time_unit", "ms"),
            )
            # Frame categories are separate from the raw-time diagnostic. Only
            # complete frames inside the common guarded interval are included.
            margin = math.ceil(args.tolerance_ms * raw_hz / 1000)
            masks = []
            for _, row in log.iterrows():
                ranges = intervals.get(row.sid, [])
                end = int(row.get("raw_frame_end_idx", row.raw_idx + int(cfg["frame_size_raw"]) - 1))
                masks.append(any(int(row.raw_idx) >= lo + margin and end <= hi - margin for lo,hi in ranges))
            frame_log = log.loc[masks].copy()
            frame_metrics = build_prediction_metrics_report(frame_log,
                target_col=cfg.get("target_col", "target"), threshold=metadata["effective_threshold"],
                **event_tolerances(cfg))
            result["models"][label] = {
                **metadata,
                "frame_event_tolerances": event_tolerances(cfg),
                "prediction_mode": cfg.get("prediction_mode", "current"),
                "frame_metrics": frame_metrics,
                "threshold": metadata["effective_threshold"],
                "holdoff_ms": metadata["effective_holdoff_frames"]
                * int(cfg["frame_size_raw"])
                / float(cfg["raw_hz"])
                * 1000,
                "recovered_baseline_misses": sorted(matched - baseline),
                "new_misses": sorted(baseline - matched),
                **changes,
                **metrics,
            }
            print(
                f"{source} {label}: matched={metrics['matched']} misses={metrics['misses']} false_positives={metrics['false_positives']}",
                flush=True,
            )
        reports.append(result)
    for _, _, _, metadata in models:
        if (
            file_digest(metadata["artifact"]) != metadata["artifact_sha256"]
            or file_digest(metadata["config"]) != metadata["config_sha256"]
        ):
            raise RuntimeError("Saved model artifacts changed during evaluation")
    (output / "comparison.json").write_text(json.dumps(reports, indent=2) + "\n")


if __name__ == "__main__":
    main()
