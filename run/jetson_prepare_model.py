#!/usr/bin/env python
# Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Developed at the German Cancer Research Center (DKFZ).

"""Prepare an isolated, fingerprinted deployment bundle on this Jetson.

This command never activates a deployment, starts/stops a service, changes
weights/thresholds, installs packages, or opens physical UART/GPIO.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dkfz.live.deployment_validation import require_idle, sha256, validate_streams
from dkfz.live.deployment_manifest import write_native_deployment_manifest
from dkfz.live.model_variants import require_config_modelname
from dkfz.live.prediction_contract import prediction_delay, supported_preprocessing_contract
from dkfz.live.runtime_config import holdoff_frames_from_settings


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modelname", required=True)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--variant", choices=["best", "tuned"], default="best")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--precision", choices=["auto", "fp16", "fp32"], default="auto")
    parser.add_argument("--validation-input", action="append", default=[], type=Path)
    parser.add_argument("--prepped-file", type=Path)
    parser.add_argument("--max-stream-samples", type=int, default=4000)
    parser.add_argument("--parity-samples", type=int, default=128)
    parser.add_argument("--ecg3-source", choices=["configured", "ecg2_minus_ecg1"], default="configured")
    parser.add_argument("--no-recovery", action="store_true")
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build_native")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def source_artifacts(args):
    source = args.model_dir.resolve(strict=True)
    config = args.config or source / ("train_config.json" if args.variant == "best" else "tuning/tuned_config.json")
    checkpoint = args.checkpoint or source / ("best.pth" if args.variant == "best" else "best_tuned.pth")
    config = config.resolve(strict=True)
    checkpoint = checkpoint.resolve(strict=True)
    for path in [config, checkpoint]:
        if not path.is_relative_to(source):
            raise ValueError("Config and checkpoint must be inside the selected model directory")
    cfg = json.loads(config.read_text())
    require_config_modelname(cfg, modelname=args.modelname, config_path=config)
    supported_preprocessing_contract(cfg)
    delay = prediction_delay(cfg)
    if delay and not args.validation_input:
        raise ValueError("Delayed 100 Hz preparation requires --validation-input for event-coordinate parity")
    if delay and cfg.get("ecg3_col") and args.ecg3_source != "ecg2_minus_ecg1":
        raise ValueError("UART ECG3 is derived: confirm the trained source with --ecg3-source ecg2_minus_ecg1")
    for path in args.validation_input:
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.parity_samples < 1 or args.max_stream_samples < 0:
        raise ValueError("Parity sample count must be positive; stream limit cannot be negative")
    if args.max_stream_samples and args.max_stream_samples < int(cfg["window_size"]) * int(cfg["frame_size_raw"]):
        raise ValueError("Stream sample limit is shorter than the model warmup window")
    return cfg, config, checkpoint


def target_hardware():
    import importlib.metadata
    marker = Path("/etc/nv_tegra_release")
    if not marker.is_file():
        raise RuntimeError("Build deployment engines on the target Jetson; use --dry-run elsewhere")
    versions = {}
    for name in ("torch", "tensorrt", "onnx", "onnxruntime"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            # Some Jetson wheels expose an importable module without the
            # matching distribution metadata directory.
            try:
                versions[name] = getattr(importlib.import_module(name), "__version__", "unknown")
            except ImportError:
                versions[name] = None
    return dict(machine=os.uname().machine, kernel=os.uname().release,
                l4t=marker.read_text().strip(), python=sys.version, packages=versions)


def prepare(args):
    cfg, config, checkpoint = source_artifacts(args)
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError("Use a new --output-dir; preparation never overwrites an earlier bundle")
    if output == (ROOT / "deploy/model").resolve() or output.is_relative_to((ROOT / "deploy").resolve()):
        raise ValueError("Preparation output must be outside the active deployment tree")
    if args.dry_run:
        print(json.dumps(dict(source=str(config), checkpoint=str(checkpoint), output=str(output),
            prediction_mode=cfg.get("prediction_mode", "current"), frame_hz=float(cfg["raw_hz"])/int(cfg["frame_size_raw"]),
            precision=args.precision, max_stream_samples=args.max_stream_samples,
            validation_inputs=list(map(str,args.validation_input)), activation=False), indent=2))
        return output
    hardware = target_hardware()
    require_idle()
    output.mkdir(parents=True)
    logs = output / "logs"
    logs.mkdir()
    model = output / "model"
    model.mkdir()
    config_copy = model / "train_config.json"
    checkpoint_copy = model / "best.pth"
    original = {str(config): sha256(config), str(checkpoint): sha256(checkpoint)}
    shutil.copy2(config, config_copy)
    shutil.copy2(checkpoint, checkpoint_copy)
    if sha256(config_copy) != original[str(config)] or sha256(checkpoint_copy) != original[str(checkpoint)]:
        raise RuntimeError("Source model changed during staging; retry after the transfer has finished")
    if json.loads(config_copy.read_text()) != cfg:
        raise RuntimeError("Source configuration changed during staging")
    report = dict(schema="dkfz-prepared-model-v1", modelname=args.modelname,
        model_variant=args.variant,
        validation_passed=False, activated=False, source_sha256=original,
        target=hardware,
        prediction_mode=cfg.get("prediction_mode", "current"),
        frame_hz=float(cfg["raw_hz"])/int(cfg["frame_size_raw"]),
        physical_holdoff_ms=holdoff_frames_from_settings(cfg)*1000*int(cfg["frame_size_raw"])/float(cfg["raw_hz"]),
        disable_recovery=args.no_recovery, ecg3_source=args.ecg3_source,
        source_preprocessing_contract_version=supported_preprocessing_contract(cfg),
        sustained_latency_verified=False, physical_gpio_latency_verified=False)
    report_path = output / "validation_report.json"

    def run(label, command, env=None):
        print(label, flush=True)
        with (logs / (label + ".log")).open("w") as stream:
            result = subprocess.run(list(map(str, command)), cwd=ROOT, stdout=stream,
                stderr=subprocess.STDOUT, env=env, timeout=1800)
        return result.returncode

    try:
        base = [sys.executable]
        onnx = model / (args.modelname + ".onnx")
        common = ["--modelname", args.modelname, "--model-dir", model]
        if run("onnx-export", base + [ROOT / "run/jetson_export_onnx.py"] + common):
            raise RuntimeError("ONNX export failed; see logs/onnx-export.log")
        parity_args = ["--samples", str(args.parity_samples)]
        if args.prepped_file:
            parity_args += ["--prepped-file", str(args.prepped_file.resolve())]
        if run("onnx-parity", base + [ROOT / "run/jetson_check_onnx_parity.py"] + common + parity_args):
            raise RuntimeError("ONNX numerical parity failed")
        selected_engine = None
        for precision in (["fp16", "fp32"] if args.precision == "auto" else [args.precision]):
            engine = model / ("model_" + precision + ".engine")
            env = dict(os.environ)
            env.setdefault("EXTRA_TRTEXEC_ARGS", "--skipInference --memPoolSize=workspace:512")
            if run("build-" + precision, ["bash", ROOT / "run/jetson_build_trt_engine.sh", args.modelname,
                "--model-dir", model, "--engine", engine, "--precision", precision], env):
                continue
            status = run("numerical-" + precision, base + [ROOT / "run/jetson_check_trt_parity.py"] + common +
                ["--engine", engine] + parity_args)
            if status == 3:
                raise RuntimeError("Inference-rate/p99 check failed; changing precision is not a latency remedy")
            if status:
                continue
            manifest = model / ("model_" + precision + ".conf")
            manifest_cfg = dict(cfg)
            if args.no_recovery:
                manifest_cfg["recovery_enabled"] = False
            write_native_deployment_manifest(manifest_cfg, manifest, engine_path=engine,
                model_variant=args.variant, source_model_dir=args.model_dir)
            if args.validation_input:
                streams = validate_streams(cfg, checkpoint_copy, engine, manifest,
                    args.validation_input, output / ("streams-" + precision),
                    native_runtime=args.build_dir / "inference/native_live_runtime",
                    max_samples=args.max_stream_samples, ecg3_source=args.ecg3_source,
                    disable_recovery=args.no_recovery)
                if not streams["passed"]:
                    print(f"{precision} event-coordinate parity failed; retaining the failed evidence", flush=True)
                    continue
                report["stream_validation"] = streams
            selected_engine = engine
            report["precision"] = precision
            break
        if selected_engine is None:
            raise RuntimeError("No engine passed numerical and required stream/event validation")
        final_engine = model / "model.engine"
        shutil.copy2(selected_engine, final_engine)
        final_manifest = model / "model.conf"
        manifest_cfg = dict(cfg)
        if args.no_recovery:
            manifest_cfg["recovery_enabled"] = False
        write_native_deployment_manifest(manifest_cfg, final_manifest, engine_path=final_engine,
            model_variant=args.variant, source_model_dir=args.model_dir)
        status = run("native-engine", [args.build_dir / "inference/native_uart_live_runtime",
            "--manifest", final_manifest, "--trigger-output", "jsonl", "--threshold-override",
            output / "unused-threshold-override.conf", "--validate-only", "--check-engine"])
        if status:
            raise RuntimeError("Native configuration/engine validation failed")
        report["sha256"] = {key:sha256(path) for key,path in dict(config=config_copy,
            checkpoint=checkpoint_copy, engine=final_engine, onnx=onnx, manifest=final_manifest).items()}
        report["validation_passed"] = True
        print(f"Prepared bundle: {output}; no deployment was activated", flush=True)
    except Exception as error:
        report["error"] = str(error)
        raise
    finally:
        report["source_unchanged"] = all(Path(path).is_file() and sha256(path) == digest for path,digest in original.items())
        if not report["source_unchanged"]:
            report["validation_passed"] = False
            report["error"] = "Source model changed during preparation; do not activate this bundle"
        report_path.write_text(json.dumps(report, indent=2))
    if not report["source_unchanged"]:
        raise RuntimeError(report["error"])
    return output


def main():
    args = build_parser().parse_args()
    # Lock is acquired by the caller (GUI or shell wrapper) - we inherit it.
    # No need to acquire/release here; just run the preparation.
    prepare(args)


if __name__ == "__main__":
    main()
