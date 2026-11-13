# Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Developed at the German Cancer Research Center (DKFZ).

"""Isolated validation helpers for target-built 50/100 Hz deployments."""
from __future__ import annotations

import contextlib
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess

from dkfz.live.io import REPO_ROOT


def sha256(path):
    with Path(path).open("rb") as source:
        digest = hashlib.sha256()
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_idle():
    """Do not stop services or compete with an active detector/training run."""
    result = subprocess.run(
        ["systemctl", "show", "dkfz-native-uart-live.service", "-p", "LoadState", "-p", "ActiveState"],
        capture_output=True, text=True, timeout=15,
    )
    properties = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    if result.returncode or not (
        properties.get("LoadState") == "not-found"
        or properties.get("ActiveState") in {"inactive", "failed"}
    ):
        raise RuntimeError("Detector service activity is active or unverified; an idle period is required")
    processes = subprocess.check_output(["ps", "-eo", "comm=,args="], text=True, timeout=15)
    scripts = {
        "train.py", "train_lopo.py", "train_lopo_hpo.py", "train_finetune.py",
        "train_recovery_lopo.py", "offline_predict_trt.py", "jetson_check_trt_parity.py",
    }
    modules = {f"run.{Path(script).stem}" for script in scripts}
    for line in processes.splitlines():
        fields = line.split(maxsplit=1)
        if not fields:
            continue
        name = fields[0]
        arguments = fields[1] if len(fields) > 1 else ""
        if name in {"native_uart_liv", "native_live_run", "trtexec"} or (
            name.startswith("python") and any(
                Path(argument).name in scripts or argument in modules
                for argument in arguments.split()
            )
        ):
            raise RuntimeError(f"An inference/training process is active: {name}")


def validate_streams(cfg, checkpoint, engine, manifest, recordings, output_dir,
                     *, native_runtime=None, max_samples=4000, ecg3_source="configured",
                     disable_recovery=False):
    """Compare real preprocessing, probabilities, peak and emission coordinates.

    No annotation, checkpoint, saved threshold or source config is modified.
    This proves engine equivalence; recordings are not assumed held out.
    """
    import pandas as pd
    from dkfz.live.dev_jsonl_runtime import run_jsonl_predictor
    from dkfz.live.onnx_export import build_model_from_config
    from dkfz.live.torch_streaming_predictor import LiveLSTMPredictor
    from dkfz.live.runtime_config import make_checked_runtime_framer, holdoff_frames_from_settings, effective_recovery_threshold
    from dkfz.live.native_runtime_parity import compare_native_runtime_outputs

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=False)
    cfg = dict(cfg)
    cfg.setdefault("not_use_puls", not cfg.get("include_puls", True))
    cfg.setdefault("verify_puls_constant_within_block", False)
    native_runtime = Path(native_runtime or REPO_ROOT / "build_native/inference/native_live_runtime")
    records = []
    failures = []
    for index, source in enumerate(recordings):
        source = Path(source).resolve()
        raw = pd.read_parquet(source)
        if max_samples:
            raw = raw.iloc[:max_samples].copy()
        if cfg.get("ecg3_col") and ecg3_source == "ecg2_minus_ecg1":
            raw[cfg["ecg3_col"]] = raw[cfg.get("ecg2_col", "ecg2_raw")].astype("float64") - raw[cfg.get("ecg1_col", "ecg1_raw")].astype("float64")
        columns = [cfg.get("ecg1_col", "ecg1_raw"), cfg.get("ecg2_col", "ecg2_raw")]
        if cfg.get("ecg3_col"):
            columns.append(cfg["ecg3_col"])
        if not cfg["not_use_puls"]:
            columns.append(cfg.get("puls_col", "puls_raw"))
        # Normalize only the harness coordinates, preserving supplied times,
        # explicit indices, gaps and sequence changes where present.
        if "raw_idx" not in raw:
            raw["raw_idx"] = range(len(raw))
        if "sid" not in raw:
            raw["sid"] = 0
        time_col = cfg.get("time_col", "time_ms")
        if time_col and time_col in raw:
            if cfg.get("time_unit", "ms") == "s":
                raw[time_col] = raw[time_col] * 1000
            columns.append(time_col)
        columns.extend(["raw_idx", "sid"])
        input_path = directory / f"{index}-input.jsonl"
        with input_path.open("w") as target:
            for row in raw[list(dict.fromkeys(columns))].to_dict("records"):
                target.write(json.dumps(row, allow_nan=False) + "\n")
        predictor = LiveLSTMPredictor(
            build_model_from_config(cfg, checkpoint), device="cpu",
            window_size=int(cfg["window_size"]), threshold=float(cfg.get("manual_threshold") if cfg.get("manual_threshold") is not None else cfg["threshold"]),
            holdoff_frames=holdoff_frames_from_settings(cfg),
            recovery_threshold=effective_recovery_threshold(cfg, disable_recovery=disable_recovery),
            framer=make_checked_runtime_framer(cfg),
        )
        reference_threshold = cfg.get("fp_threshold")
        if reference_threshold is None:
            reference_threshold = cfg["threshold"]
        predictor.trigger_logic.set_thresholds(
            predictor.threshold, max(predictor.threshold, float(reference_threshold))
        )
        python_path = directory / f"{index}-torch.jsonl"
        with python_path.open("w") as target, input_path.open() as stream:
            with contextlib.redirect_stdout(target):
                run_jsonl_predictor(
                    predictor=predictor, raw_hz=float(cfg["raw_hz"]),
                    include_puls=not cfg["not_use_puls"],
                    ecg1_col=cfg.get("ecg1_col", "ecg1_raw"), ecg2_col=cfg.get("ecg2_col", "ecg2_raw"),
                    ecg3_col=cfg.get("ecg3_col"), puls_col=cfg.get("puls_col", "puls_raw"),
                    time_col=time_col, sid_col="sid", raw_idx_col="raw_idx",
                    finished_label="Torch validation", input_stream=stream, emit_probabilities=True,
                )
        native_path = directory / f"{index}-native.jsonl"
        with native_path.open("w") as target, (directory / f"{index}-native.log").open("w") as log:
            subprocess.run([str(native_runtime), "--manifest", str(manifest),
                "--input", str(input_path), "--emit-probabilities"],
                stdout=target, stderr=log, check=True, timeout=600)
        record = dict(input=str(source), input_sha256=sha256(source), raw_samples=len(raw))
        try:
            parity = compare_native_runtime_outputs(python_path, native_path, atol=5e-4, rtol=5e-3)
            if parity.probability_frames == 0:
                raise ValueError("Recording contains no usable inference windows after warmup")
            record.update(passed=True, **asdict(parity))
        except ValueError as error:
            record.update(passed=False, error=str(error))
            failures.append(str(error))
        records.append(record)
    report = dict(passed=bool(records) and not failures, recordings=records,
                  ecg3_source=ecg3_source, source_engine_sha256=sha256(engine),
                  note="Fixed saved thresholds. Recordings are parity fixtures, not assumed held out; no accuracy or sustained-latency certification.")
    (directory / "report.json").write_text(json.dumps(report, indent=2))
    return report


def verify_prepared_report(report_path, cfg_path, engine_path, checkpoint=None, *, disable_recovery=False, model_variant=None):
    """Reject stale or substituted artifacts before activation of delayed models."""
    report = json.loads(Path(report_path).read_text())
    if report.get("schema") != "dkfz-prepared-model-v1" or not report.get("validation_passed") or not report.get("source_unchanged"):
        raise ValueError("Prepared report does not record successful model validation")
    cfg = json.loads(Path(cfg_path).read_text())
    if cfg.get("modelname") != report.get("modelname"):
        raise ValueError("Prepared report uses a different model identity")
    if cfg.get("prediction_mode") == "delayed_100hz":
        streams = report.get("stream_validation", {})
        if not streams.get("passed") or not streams.get("recordings"):
            raise ValueError("Delayed preparation requires successful real-stream event parity")
        if cfg.get("ecg3_col") and report.get("ecg3_source") != "ecg2_minus_ecg1":
            raise ValueError("Prepared ECG3 source does not match the Jetson UART")
    expected = {"config": cfg_path, "engine": engine_path}
    if checkpoint is not None:
        expected["checkpoint"] = checkpoint
    for key, path in expected.items():
        if report.get("sha256", {}).get(key) != sha256(path):
            raise ValueError(f"Prepared validation report {key} fingerprint differs")
    if model_variant is not None and report.get("model_variant") != model_variant:
        raise ValueError("Prepared report uses a different model variant")
    if bool(report.get("disable_recovery", False)) != bool(disable_recovery):
        raise ValueError("Prepared report uses a different recovery policy")
    return report
