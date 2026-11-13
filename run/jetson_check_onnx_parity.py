#!/usr/bin/env python
# Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Developed at the German Cancer Research Center (DKFZ).

"""Compare PyTorch and ONNX Runtime logits for a live ECG LSTM export."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dkfz.live.onnx_export import load_train_config  # noqa: E402
from dkfz.live.parity import (
    make_parity_inputs,
    print_parity_report,
    torch_reference_logits,
)  # noqa: E402
from dkfz.live.model_variants import (  # noqa: E402
    MODEL_VARIANT_CHOICES,
    default_checkpoint_path,
    default_onnx_path,
    default_config_path,
    require_model_artifact_path,
)


def _run_onnx(session, inputs: np.ndarray) -> np.ndarray:
    """Run a fixed-batch ONNX model over several windows."""
    try:
        return session.run(["logit"], {"x": inputs})[0]
    except Exception as exc:
        message = str(exc)
        if "Got invalid dimensions" not in message or "Expected: 1" not in message:
            raise

    outputs = [
        session.run(["logit"], {"x": inputs[i : i + 1]})[0] for i in range(len(inputs))
    ]
    return np.concatenate(outputs, axis=0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelname", default="target_r")
    parser.add_argument(
        "--model-variant", choices=MODEL_VARIANT_CHOICES, default="best"
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--onnx", default=None)
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Artifact root. Defaults to models/<modelname>.",
    )
    parser.add_argument("--prepped-file", default=None)
    parser.add_argument("--samples", type=int, default=8192)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    args = parser.parse_args()

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
    onnx_path = (
        Path(args.onnx)
        if args.onnx
        else default_onnx_path(args.modelname, args.model_variant, args.model_dir)
    )
    try:
        for label, path in (
            ("training config", config_path),
            ("checkpoint", checkpoint),
            ("ONNX model", onnx_path),
        ):
            require_model_artifact_path(
                path,
                modelname=args.modelname,
                model_dir=args.model_dir,
                model_variant=args.model_variant,
                label=label,
            )
    except ValueError as exc:
        parser.error(str(exc))

    cfg = load_train_config(
        args.modelname, config_path, args.model_variant, args.model_dir
    )

    from dkfz.live.onnx_export import validate_onnx_prediction_contract

    validate_onnx_prediction_contract(cfg, onnx_path)
    inputs = make_parity_inputs(
        cfg, samples=args.samples, prepped_file=args.prepped_file
    )
    torch_logits = torch_reference_logits(cfg, checkpoint, inputs)

    try:
        import onnxruntime as ort
    except ImportError:
        import onnx
        from onnx.reference import ReferenceEvaluator

        session = ReferenceEvaluator(onnx.load(str(onnx_path)))
        onnx_logits = np.concatenate([
            session.run(["logit"], {"x": window[None]})[0] for window in inputs
        ], axis=0)
        print("ONNX backend: CPU ReferenceEvaluator (onnxruntime unavailable)")
    else:
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        onnx_logits = _run_onnx(session, inputs)
        print("ONNX backend: CPUExecutionProvider")

    passed = print_parity_report(
        reference=torch_logits,
        candidate=onnx_logits,
        samples=len(inputs),
        rtol=float(args.rtol),
        atol=float(args.atol),
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
