#!/usr/bin/env python
# Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Developed at the German Cancer Research Center (DKFZ).

"""Export a training configuration for the C++ TensorRT live runtime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dkfz.live.deployment_manifest import (  # noqa: E402
    default_native_manifest_path,
    write_native_deployment_manifest,
)
from dkfz.live.io import repo_relative_path  # noqa: E402
from dkfz.live.model_variants import (  # noqa: E402
    MODEL_VARIANT_CHOICES,
    default_config_path,
    default_trt_engine_path,
    get_model_variant,
    require_config_modelname,
    require_model_artifact_path,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelname", default="target_r")
    parser.add_argument(
        "--model-variant", choices=MODEL_VARIANT_CHOICES, default="best"
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--engine", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--no-recovery",
        action="store_true",
        help="Disable recovery events in the manifest, preserving the two-output engine.",
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Source artifact root. Defaults to models/<modelname>.",
    )
    parser.add_argument(
        "--allow-engine-outside-model-dir",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    config_path = (
        Path(args.config)
        if args.config
        else default_config_path(args.modelname, args.model_variant, args.model_dir)
    )
    engine_path = (
        Path(args.engine)
        if args.engine
        else default_trt_engine_path(args.modelname, args.model_variant, args.model_dir)
    )
    output_path = (
        Path(args.output)
        if args.output
        else default_native_manifest_path(
            args.modelname, args.model_variant, args.model_dir
        )
    )
    try:
        require_model_artifact_path(
            config_path,
            modelname=args.modelname,
            model_dir=args.model_dir,
            model_variant=args.model_variant,
            label="training config",
        )
        if not args.allow_engine_outside_model_dir:
            require_model_artifact_path(
                engine_path,
                modelname=args.modelname,
                model_dir=args.model_dir,
                model_variant=args.model_variant,
                label="TensorRT engine",
            )
    except ValueError as exc:
        parser.error(str(exc))
    with config_path.open("r", encoding="utf-8") as source:
        cfg = json.load(source)
    if args.no_recovery:
        cfg["recovery_enabled"] = False
    try:
        require_config_modelname(
            cfg,
            modelname=args.modelname,
            config_path=config_path,
        )
    except ValueError as exc:
        parser.error(str(exc))
    output_path = write_native_deployment_manifest(
        cfg,
        output_path,
        engine_path=engine_path,
        model_variant=args.model_variant,
        source_model_dir=repo_relative_path(
            get_model_variant(args.model_variant).model_dir(
                args.modelname, args.model_dir
            ).resolve()
        ),
    )
    print(f"Wrote native deployment manifest to {repo_relative_path(output_path)}")
    print(f"Training config: {repo_relative_path(config_path)}")
    print(f"TensorRT engine: {repo_relative_path(engine_path)}")
    if engine_path.is_file():
        print("Engine fingerprint: included")
    else:
        print(
            "WARNING: engine file does not exist yet; manifest has no engine "
            "fingerprint. Regenerate it after building the engine.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
