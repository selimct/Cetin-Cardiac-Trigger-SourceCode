#!/usr/bin/env python
# Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Developed at the German Cancer Research Center (DKFZ).

"""Compare PyTorch and TensorRT logits for a built live ECG engine."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dkfz.live.onnx_export import load_train_config  # noqa: E402
from dkfz.live.parity import (
    make_parity_inputs,
    print_parity_report,
    print_performance_report,
    torch_reference_logits,
)  # noqa: E402
from dkfz.live.tensorrt_engine import TensorRTLogitEngine  # noqa: E402
from dkfz.live.model_variants import (  # noqa: E402
    MODEL_VARIANT_CHOICES,
    default_checkpoint_path,
    default_config_path,
    default_trt_engine_path,
    require_model_artifact_path,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelname", default="target_r")
    parser.add_argument(
        "--model-variant", choices=MODEL_VARIANT_CHOICES, default="best"
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--engine", default=None)
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Artifact root. Defaults to models/<modelname>.",
    )
    parser.add_argument(
        "--allow-engine-outside-model-dir",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--prepped-file", default=None)
    parser.add_argument("--samples", type=int, default=8192)
    parser.add_argument("--atol", type=float, default=5e-3)
    parser.add_argument("--rtol", type=float, default=5e-3)
    parser.add_argument(
        "--warmup-samples",
        type=int,
        default=100,
        help="Untimed TensorRT inferences before latency measurement.",
    )
    parser.add_argument(
        "--min-throughput-hz",
        type=float,
        default=None,
        help=(
            "Required sustained inference rate. Defaults to raw_hz/frame_size_raw "
            "from the model config."
        ),
    )
    parser.add_argument(
        "--max-p99-latency-ms",
        type=float,
        default=None,
        help=(
            "Maximum p99 inference latency. Defaults to one configured model "
            "frame period."
        ),
    )
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be at least 1")
    if args.warmup_samples < 0:
        parser.error("--warmup-samples cannot be negative")
    if args.min_throughput_hz is not None and args.min_throughput_hz <= 0:
        parser.error("--min-throughput-hz must be positive")
    if args.max_p99_latency_ms is not None and args.max_p99_latency_ms <= 0:
        parser.error("--max-p99-latency-ms must be positive")

    config_path = (
        Path(args.config)
        if args.config
        else default_config_path(args.modelname, args.model_variant, args.model_dir)
    )
    checkpoint = (
        Path(args.checkpoint)
        if args.checkpoint
        else default_checkpoint_path(args.modelname, args.model_variant, args.model_dir)
    )
    engine_path = (
        Path(args.engine)
        if args.engine
        else default_trt_engine_path(args.modelname, args.model_variant, args.model_dir)
    )
    try:
        for label, path in (
            ("training config", config_path),
            ("checkpoint", checkpoint),
        ):
            require_model_artifact_path(
                path,
                modelname=args.modelname,
                model_dir=args.model_dir,
                label=label,
            )
        if not args.allow_engine_outside_model_dir:
            require_model_artifact_path(
                engine_path,
                modelname=args.modelname,
                model_dir=args.model_dir,
                label="TensorRT engine",
            )
    except ValueError as exc:
        parser.error(str(exc))

    cfg = load_train_config(
        args.modelname, config_path, args.model_variant, args.model_dir
    )
    raw_hz = float(cfg["raw_hz"])
    frame_size_raw = int(cfg["frame_size_raw"])
    if raw_hz <= 0 or frame_size_raw <= 0:
        parser.error("raw_hz and frame_size_raw must be positive")
    configured_inference_rate_hz = raw_hz / frame_size_raw
    min_throughput_hz = (
        float(args.min_throughput_hz)
        if args.min_throughput_hz is not None
        else configured_inference_rate_hz
    )
    max_p99_latency_ms = (
        float(args.max_p99_latency_ms)
        if args.max_p99_latency_ms is not None
        else 1000.0 / configured_inference_rate_hz
    )

    inputs = make_parity_inputs(
        cfg, samples=args.samples, prepped_file=args.prepped_file
    )
    engine = TensorRTLogitEngine(
        engine_path=engine_path,
        input_shape=(1, int(cfg["window_size"]), int(cfg["input_dim"])),
    )

    torch_logits = torch_reference_logits(cfg, checkpoint, inputs)

    for index in range(args.warmup_samples):
        engine.infer_logit(inputs[index % len(inputs)])

    trt_logits = np.empty(len(inputs), dtype=np.float32)
    durations_ns = np.empty(len(inputs), dtype=np.int64)
    measurement_started_ns = time.perf_counter_ns()
    for index, window in enumerate(inputs):
        inference_started_ns = time.perf_counter_ns()
        trt_logits[index] = engine.infer_logit(window)
        durations_ns[index] = time.perf_counter_ns() - inference_started_ns
    measurement_elapsed_ns = time.perf_counter_ns() - measurement_started_ns

    parity_passed = print_parity_report(
        reference=torch_logits,
        candidate=trt_logits,
        samples=len(inputs),
        rtol=args.rtol,
        atol=args.atol,
    )
    performance_passed = print_performance_report(
        durations_ns=durations_ns,
        elapsed_ns=measurement_elapsed_ns,
        min_throughput_hz=min_throughput_hz,
        max_p99_latency_ms=max_p99_latency_ms,
    )
    if not parity_passed:
        return 1
    if not performance_passed:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
