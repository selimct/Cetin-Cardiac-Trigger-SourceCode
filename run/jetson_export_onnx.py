#!/usr/bin/env python
# Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Developed at the German Cancer Research Center (DKFZ).

"""Export a trained live ECG LSTM checkpoint to ONNX."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import onnx

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dkfz.live.onnx_export import (  # noqa: E402
    DEFAULT_ONNX_OPSET,
    build_model_from_config,
    export_model_onnx,
    load_train_config,
)
from dkfz.live.io import repo_relative_path  # noqa: E402
from dkfz.live.model_variants import (  # noqa: E402
    MODEL_VARIANT_CHOICES,
    default_checkpoint_path,
    default_config_path,
    default_onnx_path,
    require_model_artifact_path,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelname", default="target_r")
    parser.add_argument(
        "--model-variant",
        choices=MODEL_VARIANT_CHOICES,
        default="best",
        help=(
            "Checkpoint/config/export defaults to use. 'best' uses best.pth, "
            "train_config.json, and <modelname>.onnx; 'tuned' uses "
            "best_tuned.pth, tuning/tuned_config.json, and "
            "<modelname>_best_tuned.onnx."
        ),
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Artifact root. Defaults to models/<modelname>.",
    )
    parser.add_argument("--opset", type=int, default=DEFAULT_ONNX_OPSET)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--dynamic-batch",
        action="store_true",
        help="Export ONNX with a dynamic batch axis for batched TensorRT inference.",
    )
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else None
    if config_path is None:
        config_path = default_config_path(
            args.modelname, args.model_variant, args.model_dir
        )
    checkpoint = (
        Path(args.checkpoint)
        if args.checkpoint
        else default_checkpoint_path(
            args.modelname, args.model_variant, args.model_dir
        )
    )
    output = (
        Path(args.output)
        if args.output
        else default_onnx_path(args.modelname, args.model_variant, args.model_dir)
    )
    try:
        for label, path in (
            ("training config", config_path),
            ("checkpoint", checkpoint),
            ("ONNX output", output),
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

    model = build_model_from_config(cfg, checkpoint)
    output = export_model_onnx(
        model=model,
        output=output,
        window_size=int(cfg["window_size"]),
        input_dim=int(cfg["input_dim"]),
        opset=args.opset,
        batch_size=args.batch_size,
        dynamic_batch=args.dynamic_batch,
    )

    onnx_model = onnx.load(output)
    onnx.checker.check_model(onnx_model)
    opsets = ", ".join(
        f"{opset.domain or 'ai.onnx'}={opset.version}"
        for opset in onnx_model.opset_import
    )
    print(f"Wrote ONNX model to {repo_relative_path(output)}")
    print(f"ONNX opset: {opsets}")
    print(
        "Input contract: "
        f"x float32[{'batch' if args.dynamic_batch else args.batch_size}, "
        f"{int(cfg['window_size'])}, {int(cfg['input_dim'])}]"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
