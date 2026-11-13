#!/usr/bin/env python
# Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Developed at the German Cancer Research Center (DKFZ).
"""Qualify a delayed ECG3 model without training or production activation.

Run as the ordinary project user. Sudo is used only for separate transient
service tests and deletion of their uniquely named /run directories.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import shlex
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dkfz.live.deployment_validation import require_idle, sha256, verify_prepared_report
from dkfz.live.prediction_contract import prediction_delay, supported_preprocessing_contract
from dkfz.live.runtime_config import holdoff_frames_from_settings
from dkfz.live.qualification import SERVICE_KEYS, UNIT_PREFIX, settings, profile_failures, performance_failures, uart_integrity_failures, valid_count


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", required=True, type=Path)
    p.add_argument("--variant", choices=["best", "tuned"], default="best")
    p.add_argument("--baseline-dir", type=Path, default=ROOT / "finished_models/ecg3")
    p.add_argument("--baseline-variant", choices=["best", "tuned"], default="best")
    p.add_argument("--parity-input", required=True, action="append", type=Path)
    p.add_argument("--test-input", required=True, action="append", type=Path,
                   help="Caller-declared held-out annotated raw parquet; no threshold tuning")
    p.add_argument("--uart-recording", required=True, type=Path)
    p.add_argument("--ecg3-source", required=True, choices=["ecg2_minus_ecg1"])
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--build-dir", type=Path, default=ROOT / "build_native")
    p.add_argument("--precision", choices=["auto", "fp16", "fp32"], default="auto")
    p.add_argument("--parity-samples", type=int, default=512)
    p.add_argument("--seconds", type=int, default=612)
    p.add_argument("--timing-runs", type=int, default=2)
    p.add_argument("--live-uart-seconds", type=int, default=60, help="Actual UART check with GPIO off; 0 skips it and requires review")
    p.add_argument("--accuracy-limit-seconds", type=float)
    p.add_argument("--min-sensitivity", type=float)
    p.add_argument("--max-fp-per-minute", type=float)
    p.add_argument("--max-emission-latency-ms", type=float)
    p.add_argument("--max-missed-deadlines", type=int, default=0)
    p.add_argument("--max-backlog-samples", type=int, default=8)
    p.add_argument("--dry-run", action="store_true")
    return p


def model(directory, variant):
    directory = directory.resolve(strict=True)
    cfg_path = directory / ("train_config.json" if variant == "best" else "tuning/tuned_config.json")
    checkpoint = directory / ("best.pth" if variant == "best" else "best_tuned.pth")
    cfg = json.loads(cfg_path.read_text())
    name=cfg.get("modelname")
    if not checkpoint.is_file() or not isinstance(name,str) or not name or Path(name).name!=name or name in {".",".."}:
        raise ValueError(f"Missing selected checkpoint or invalid model identity in {directory}")
    supported_preprocessing_contract(cfg)
    prediction_delay(cfg)
    return dict(directory=directory, config=cfg_path, checkpoint=checkpoint, cfg=cfg)


def arguments(args):
    candidate = model(args.model_dir, args.variant)
    baseline = model(args.baseline_dir, args.baseline_variant)
    if prediction_delay(candidate["cfg"]) != 1 or int(baseline["cfg"]["frame_size_raw"]) != 8 or prediction_delay(baseline["cfg"]):
        raise ValueError("Select a separately trained delayed 100 Hz candidate and current-frame 50 Hz baseline")
    if candidate["cfg"]["modelname"] == baseline["cfg"]["modelname"]:
        raise ValueError("Candidate and baseline must have distinct model identities")
    if any(float(m["cfg"]["raw_hz"]) != 400 for m in (candidate, baseline)):
        raise ValueError("Both models must consume 400 Hz raw input")
    if any(m["cfg"].get("recovery_head") for m in (candidate, baseline)):
        raise ValueError("This qualification compares single-output primary models")
    if args.live_uart_seconds != 0 and args.live_uart_seconds < 30:
        raise ValueError("Live UART check must be at least 30 seconds, or 0 to skip explicitly")
    if args.seconds < 600 or args.timing_runs < 1 or args.parity_samples < 1:
        raise ValueError("Use at least 600 seconds, one timing run and one numerical window")
    if args.max_missed_deadlines < 0 or args.max_backlog_samples < 0:
        raise ValueError("Timing budgets must be nonnegative")
    for name in ("min_sensitivity", "max_fp_per_minute", "max_emission_latency_ms", "accuracy_limit_seconds"):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value < 0):
            raise ValueError(f"{name} must be finite and nonnegative")
    if args.accuracy_limit_seconds is not None and args.accuracy_limit_seconds <= 0:
        raise ValueError("--accuracy-limit-seconds must be positive")
    if args.max_emission_latency_ms is not None and args.min_sensitivity is None:
        raise ValueError("Emission-latency acceptance needs the sensitivity and FP/min bounds too")
    if args.min_sensitivity is not None and args.min_sensitivity > 1:
        raise ValueError("--min-sensitivity must be within [0,1]")
    if (args.min_sensitivity is None) != (args.max_fp_per_minute is None):
        raise ValueError("Supply both accuracy sensitivity and FP/min bounds, or omit both for review-only accuracy")
    for path in [*args.parity_input, *args.test_input, args.uart_recording]:
        if not path.resolve(strict=True).is_relative_to(ROOT):
            raise ValueError("Recording paths must be inside this project for the read-only service bind")
    args.build_dir=args.build_dir.resolve()
    if not args.build_dir.is_relative_to(ROOT) or not (args.build_dir / "CMakeCache.txt").is_file():
        raise ValueError("Choose the existing project-local native build (--build-dir); configure it with run/rebuild_native.sh first if missing")
    output = (args.output_dir or ROOT / "artifact" / ("qualification-" + time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8])).resolve()
    if output.exists() or not output.is_relative_to((ROOT / "artifact").resolve()):
        raise ValueError("Choose a new project-local output directory under artifact/")
    profile = settings(ROOT / "deploy/dkfz-native-uart-live.service", "Service")
    known=set(SERVICE_KEYS) | {"Type","WorkingDirectory","BindReadOnlyPaths","ExecStart","StateDirectory","StateDirectoryMode",
        "ReadWritePaths","Restart","RestartSec","RestartPreventExitStatus","StandardOutput","StandardError","SyslogIdentifier",
        "LogNamespace","LogRateLimitIntervalSec","LogRateLimitBurst"}
    if set(profile)-known:
        raise ValueError(f"New service directives need qualification-adapter review: {set(profile)-known}")
    for key in (*SERVICE_KEYS, "WorkingDirectory", "BindReadOnlyPaths", "Type"):
        if key not in profile:
            raise ValueError(f"Service profile missing {key}; review the qualification adapter")
    if profile["Type"] != "notify" or profile["WorkingDirectory"] != "/opt/dkfz-live":
        raise ValueError("Unsupported service layout; review the qualification adapter")
    command=shlex.split(profile.get("ExecStart",""))
    if "--realtime-priority" not in command or command[command.index("--realtime-priority")+1] != "20" or "--require-realtime" not in command:
        raise ValueError("Review the adapter for the service worker-scheduling command")
    if int(profile["LimitRTPRIO"]) != 20 or profile["CPUSchedulingPolicy"] != "other":
        raise ValueError("Expected the bounded worker-priority service profile")
    return candidate, baseline, output, profile


def execute(command, log, timeout=1800):
    """No shell interpolation; preserve logs and show a periodic heartbeat."""
    command = list(map(str, command))
    print(f"Running {log.name}", flush=True)
    with log.open("w") as output:
        process = subprocess.Popen(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT,
                                   env={k:v for k,v in os.environ.items() if k not in {"NOTIFY_SOCKET", "WATCHDOG_PID", "WATCHDOG_USEC", "INVOCATION_ID"}})
        deadline = time.monotonic() + timeout
        try:
            while True:
                try:
                    code = process.wait(timeout=min(30, max(.1, deadline-time.monotonic())))
                    break
                except subprocess.TimeoutExpired:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"Timed out: {log}")
                    print(f"  {log.name}: still running", flush=True)
        except BaseException:
            process.terminate()
            try: process.wait(timeout=10)
            except subprocess.TimeoutExpired: process.kill(); process.wait()
            raise
    if code:
        raise subprocess.CalledProcessError(code, command)


@contextlib.contextmanager
def deployment_lock():
    lock_path = ROOT / ".jetson-control-deployment.lock"
    lock = lock_path.open("a")
    acquired = False
    for attempt in range(60):
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
            break
        except BlockingIOError:
            if attempt == 0:
                print(f"Waiting for lock {lock_path} (another process is running)...")
            time.sleep(1)

    if not acquired:
        lock.close()
        raise RuntimeError(f"Could not acquire lock after 60 seconds: {lock_path}")

    try:
        yield
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


def recording_copies(paths, models, directory, *, limit_seconds=None):
    """Only derive the channel in new copies; annotation values stay unchanged."""
    import pandas as pd
    directory.mkdir()
    outputs, evidence = [], []
    for index, path in enumerate(paths):
        source = path.resolve(strict=True)
        raw = pd.read_parquet(source)
        if limit_seconds is not None:
            raw = raw.iloc[:int(limit_seconds*400)].copy()
        targets = {m["cfg"].get("target_col", "target") for m in models}
        if any(col not in raw for col in targets):
            raise ValueError(f"Missing raw annotation column in {source}: {targets}")
        annotations = raw[sorted(targets)].copy(deep=True)
        if "sid" not in raw: raw["sid"]=0
        if "raw_idx" not in raw:
            raw["raw_idx"]=raw["idx"] if "idx" in raw else raw.groupby("sid",sort=False).cumcount()
        for m in models:
            cfg = m["cfg"]
            if cfg.get("ecg3_col"):
                raw[cfg["ecg3_col"]] = raw[cfg.get("ecg2_col", "ecg2_raw")].astype("float64") - raw[cfg.get("ecg1_col", "ecg1_raw")].astype("float64")
        assert annotations.equals(raw[sorted(targets)])
        dest = directory / f"{index:03d}-{source.stem}.parquet"
        raw.to_parquet(dest, index=False)
        outputs.append(dest)
        evidence.append(dict(source=str(source), source_sha256=sha256(source), copy=str(dest),
                             copy_sha256=sha256(dest), samples=len(raw), annotations_unchanged=True))
    return outputs, evidence


def parity_features(source,cfg,output):
    """Causal real recording windows for numerical Torch/ONNX/TRT checks."""
    import pandas as pd
    from dkfz.live.runtime_config import make_checked_runtime_framer
    from dkfz.live.feature_framing import make_live_feature_dataframe
    cfg=dict(cfg)
    cfg.setdefault("not_use_puls",not cfg.get("include_puls",True))
    cfg.setdefault("verify_puls_constant_within_block",False)
    frames=make_live_feature_dataframe(pd.read_parquet(source),
        ecg1_col=cfg.get("ecg1_col","ecg1_raw"),ecg2_col=cfg.get("ecg2_col","ecg2_raw"),
        ecg3_col=cfg.get("ecg3_col"),puls_col=None if cfg["not_use_puls"] else cfg.get("puls_col","puls_raw"),
        target_col=None,time_col=cfg.get("time_col","time_ms"),time_unit=cfg.get("time_unit","ms"),
        framer=make_checked_runtime_framer(cfg))
    # Never join a numerical window across a reset, even when SID is unchanged.
    if frames.empty: raise ValueError("Parity recording produced no feature frames")
    breaks=frames.sid.ne(frames.sid.shift()) | frames.frame_idx.diff().ne(1) | frames.raw_idx.diff().ne(int(cfg["frame_size_raw"]))
    frames["sid"]=breaks.cumsum()
    if frames.groupby("sid").size().max()<int(cfg["window_size"]):
        raise ValueError("Parity recording has no complete contiguous model window")
    frames.to_parquet(output,index=False)
    return output


def discontinuity_fixture(source, cfg, output):
    import pandas as pd
    raw = pd.read_parquet(source)
    segment = int(cfg["window_size"])*int(cfg["frame_size_raw"]) + 8
    count = 3*segment + 2
    if len(raw) < count:
        raise ValueError(f"Parity input needs at least {count} raw samples to exercise fresh warmups/gaps/sequence/partial EOF")
    raw = raw.iloc[:count].copy()
    raw["raw_idx"] = [i + (4 if i >= segment else 0) for i in range(count)]
    raw["sid"] = [0 if i < 2*segment else 1 for i in range(count)]
    raw.to_parquet(output, index=False)
    return output


def namespace_path(path):
    return Path("/opt/dkfz-live") / Path(path).resolve().relative_to(ROOT)


def service_command(args, profile, bundle, runtime_cfg, unit, *, live=False):
    seconds=args.live_uart_seconds if live else args.seconds
    properties = {key:profile[key] for key in SERVICE_KEYS}
    properties.update(WorkingDirectory="/opt/dkfz-live", NotifyAccess="all", Restart="no",
        RuntimeDirectory=unit, RuntimeDirectoryMode="0755", RuntimeDirectoryPreserve="yes",
        ReadWritePaths="/run/"+unit,
        BindReadOnlyPaths=f"{ROOT}:/opt/dkfz-live {Path(sys.prefix)}:/opt/dkfz-qualification-python",
        RuntimeMaxSec=f"{seconds+120}s", UMask="0022")
    command = ["sudo", "systemd-run", "--unit", unit, "--wait", "--pipe", "--collect", "--service-type=notify",
               "--setenv", f"DKFZ_QUALIFICATION_UNIT={unit}"]
    for key,value in properties.items(): command += ["--property", f"{key}={value}"]
    command += ["/opt/dkfz-qualification-python/bin/python", "/opt/dkfz-live/run/jetson_paced_uart_check.py",
        "--manifest", namespace_path(bundle / "model/model.conf"),
        "--native-runtime", namespace_path(args.build_dir / "inference/native_uart_live_runtime"),
        "--runtime-config", namespace_path(runtime_cfg), "--managed-service",
        "--seconds", str(seconds), "--realtime-priority", "20", "--require-realtime",
        "--output-dir", f"/run/{unit}/results"]
    if live: command += ["--live-device",settings(runtime_cfg)["device"]]
    else: command += ["--recording",namespace_path(args.uart_recording)]
    return list(map(str,command)), properties


def service_test(args, profile, bundle, runtime_cfg, output, index, run=execute):
    unit = UNIT_PREFIX + uuid.uuid4().hex
    ephemeral = Path("/run") / unit
    if ephemeral.exists(): raise FileExistsError(ephemeral)
    live=index=="live-uart"
    seconds=args.live_uart_seconds if live else args.seconds
    command, properties = service_command(args, profile, bundle, runtime_cfg, unit,live=live)
    (output / f"service-{index}-command.json").write_text(json.dumps(dict(unit=unit, argv=command, properties=properties),indent=2))
    copied = output / f"service-{index}"
    try:
        run(command, output / f"service-{index}.log", seconds+240)
    except BaseException:
        # Only our uniquely named test unit; never stop the production detector.
        subprocess.run(["sudo", "-n", "systemctl", "stop", unit+".service"], check=False, timeout=15)
        if (ephemeral / "results").is_dir(): shutil.copytree(ephemeral / "results", copied)
        raise
    shutil.copytree(ephemeral / "results", copied)
    # Validated, generated /run path; private service already exited (--wait).
    run(["sudo", "rm", "-rf", "--", str(ephemeral)], output / f"service-{index}-cleanup.log", 60)
    summary = json.loads((copied / "latency-summary.json").read_text())
    pacer = json.loads((copied / "pacer.json").read_text())
    context = json.loads((copied / "service-context.json").read_text())
    diagnostics = [json.loads(line) for line in (copied / "runtime.log").read_text().splitlines() if line.startswith("{")]
    finished = next(r for r in reversed(diagnostics) if r.get("code") == "runtime_finished")
    failures = profile_failures(profile, context) + performance_failures(summary,pacer,finished,
        json.loads((bundle / "model/train_config.json").read_text()),seconds,
        max_missed=args.max_missed_deadlines,max_backlog=args.max_backlog_samples)
    failures += uart_integrity_failures(diagnostics,finished,live=live)
    import re
    thermal={}
    for line in (copied / "tegrastats.log").read_text().splitlines():
        for name,temp in re.findall(r"(\w+)@([0-9.]+)C",line): thermal.setdefault(name,[]).append(float(temp))
    thermal={key:dict(minimum_c=min(values),maximum_c=max(values),readings=len(values)) for key,values in thermal.items()}
    return dict(passed=not failures, failures=failures, output=str(copied), summary=summary,
                thermal=thermal,power_mode=(copied / "power-mode.txt").read_text(),power_watts_available=False,
                pacer=pacer, context=context, runtime_finished=finished,
                input_pacer_delays_excluded_from_native_deadlines=pacer["pacer_delays_over_10ms"])


def accuracy_review(comparisons, args, *, label="candidate"):
    matched=missed=false_positives=0
    minutes=0
    max_latency=None
    for recording in comparisons:
        metric=recording["models"][label]
        for key in ("matched", "misses", "false_positives"):
            if not valid_count(metric[key]):
                raise ValueError(f"Accuracy {key} must be a finite nonnegative integer")
        matched+=metric["matched"]; missed+=metric["misses"]; false_positives+=metric["false_positives"]
        ranges=metric["observable_raw_intervals"]
        # The raw diagnostic applies a 20 ms edge guard at each interval end.
        duration=(metric["evaluation_duration_ms"] if "evaluation_duration_ms" in metric else
                  sum(max(0,hi-lo+1-16)/400*1000 for intervals in ranges.values() for lo,hi in intervals))
        if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration < 0:
            raise ValueError("Accuracy evaluation_duration_ms must be finite and nonnegative")
        minutes+=duration/60000
        value=metric["stream_trigger_latency_ms"]["max"]
        if value is not None:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("Accuracy stream trigger latency must be finite")
            max_latency=max(value,max_latency if max_latency is not None else value)
    if not matched+missed or not math.isfinite(minutes) or minutes <= 0:
        raise ValueError("No annotated beats/common valid interval in held-out comparison")
    sensitivity=matched/(matched+missed)
    fp_min=false_positives/minutes
    failures=[]
    configured=args.min_sensitivity is not None and args.max_fp_per_minute is not None
    if configured:
        if sensitivity < args.min_sensitivity: failures.append("Sensitivity below specified bound")
        if fp_min > args.max_fp_per_minute: failures.append("False positives/min exceed specified bound")
        if args.max_emission_latency_ms is not None and (max_latency is None or max_latency > args.max_emission_latency_ms):
            failures.append("Sample-axis emission latency exceeds specified bound")
    return dict(status=("review_required" if not configured else "failed" if failures else "passed"),
        failures=failures,matched=matched,misses=missed,false_positives=false_positives,
        sensitivity=sensitivity,false_positives_per_minute=fp_min,evaluated_minutes=minutes,
        maximum_sample_axis_emission_latency_ms=max_latency, thresholds_tuned=False,
        held_out_status="declared by caller; split provenance requires review")


def qualify(args, run=execute, idle=require_idle, service_runner=service_test):
    candidate, baseline, output, profile=arguments(args)
    holdoff_ms=holdoff_frames_from_settings(candidate["cfg"])*int(candidate["cfg"]["frame_size_raw"])*1000/400
    common_holdoff_ms=math.ceil(holdoff_ms/20)*20
    plan=dict(candidate=str(candidate["config"]), baseline=str(baseline["config"]), output=str(output),
        candidate_saved_holdoff_ms=holdoff_ms, comparison_holdoff_ms=common_holdoff_ms, seconds_per_run=args.seconds, timing_runs=args.timing_runs,
        service_profile=profile, live_uart_seconds=args.live_uart_seconds,
        parity_inputs=list(map(str,args.parity_input)),held_out_inputs=list(map(str,args.test_input)),
        uart_recording=str(args.uart_recording),
        accuracy_acceptance=dict(min_sensitivity=args.min_sensitivity,max_fp_per_minute=args.max_fp_per_minute,
                                 max_emission_latency_ms=args.max_emission_latency_ms), production_activation=False, physical_gpio=False,
        physical_measurement="requires separate hardware measurement; not verified here")
    if args.dry_run:
        print(json.dumps(plan,indent=2)); return plan
    if os.geteuid()==0: raise ValueError("Run as the ordinary project user; the script prompts sudo only for its private service tests")
    idle()
    output.mkdir(parents=True)
    (output / "plan.json").write_text(json.dumps(plan,indent=2))
    sources=[m[k] for m in (candidate,baseline) for k in ("config","checkpoint")]
    sources += [p.resolve() for p in [*args.parity_input,*args.test_input,args.uart_recording]]
    sources += [ROOT / "deploy/dkfz-native-uart-live.service", ROOT / "config/native_uart_live.conf"]
    sources += [p for p in [ROOT / "deploy/model/model.conf",ROOT / "deploy/model/model.engine"] if p.exists()]
    for tree in (ROOT / "src/dkfz",ROOT / "native",ROOT / "run"):
        sources += [p for p in tree.rglob("*") if p.is_file() and (p.suffix in {".py",".cpp",".hpp",".sh"} or p.name=="CMakeLists.txt")]
    hashes={str(p):sha256(p) for p in sources}
    version=dict(python=sys.version,kernel=os.uname().release)
    if (ROOT / ".git").exists():
        version["commit"]=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
        version["git_status"]=subprocess.check_output(["git","status","--short"],cwd=ROOT,text=True)
    report=dict(schema="dkfz-model-qualification-v1",plan=plan,stages={},status="incomplete",source_sha256=hashes,environment=version)
    bundle=output / "prepared"
    try:
        print("Sudo is needed later for private service tests; production is never started/stopped.",flush=True)
        subprocess.run(["sudo","-v"],check=True)
        with deployment_lock():
            idle()
            run(["cmake","--build",args.build_dir,"-j2"],output / "build.log")
            report["stages"]["development_regressions"] = dict(
                status="not_bundled",
                reason="Development test suites are maintained in the upstream repository")
        cache=args.build_dir / "CMakeCache.txt"
        if cache.is_file():
            report["environment"]["cmake_cache_sha256"]=sha256(cache)
            hashes[str(cache)]=sha256(cache)
        for binary in ("native_uart_live_runtime","native_live_runtime"):
            path=args.build_dir / "inference" / binary
            if path.is_file(): hashes[str(path)]=sha256(path)
        parity_inputs,parity_evidence=recording_copies(args.parity_input,[candidate],output / "parity-inputs")
        fixture=discontinuity_fixture(parity_inputs[0],candidate["cfg"],output / "reset-gap-partial-eof.parquet")
        features=parity_features(parity_inputs[0],candidate["cfg"],output / "parity-features.parquet")
        prep=[sys.executable,ROOT / "run/jetson_prepare_model.py","--modelname",candidate["cfg"]["modelname"],
            "--model-dir",candidate["directory"],"--variant",args.variant,"--output-dir",bundle,
            "--precision",args.precision,"--parity-samples",str(args.parity_samples),"--max-stream-samples","0",
            "--ecg3-source",args.ecg3_source,"--build-dir",args.build_dir,"--prepped-file",features]
        for path in [*parity_inputs,fixture]: prep += ["--validation-input",path]
        idle(); run(prep,output / "prepare.log",7200)
        prepared=verify_prepared_report(bundle / "validation_report.json",bundle / "model/train_config.json",bundle / "model/model.engine",bundle / "model/best.pth",model_variant=args.variant)
        if sum(r.get("trigger_events",0) for r in prepared["stream_validation"]["recordings"])==0:
            raise ValueError("No positive events in parity fixtures; choose representative validation recordings")
        report["environment"]["target"]=prepared.get("target",{})
        report["stages"]["parity"]=dict(passed=True,report=str(bundle / "validation_report.json"),inputs=parity_evidence,
            real_numerical_features=dict(path=str(features),sha256=sha256(features)))
        tests, evidence=recording_copies(args.test_input,[candidate,baseline],output / "held-out-inputs",limit_seconds=args.accuracy_limit_seconds)
        comparison=[sys.executable,ROOT / "run/compare_prediction_models.py",*tests,
            "--model","baseline",baseline["config"],baseline["checkpoint"],
            "--model","candidate",candidate["config"],candidate["checkpoint"],
            "--output-dir",output / "accuracy","--holdoff-ms",str(common_holdoff_ms)]
        production_cfg=settings(ROOT / "config/native_uart_live.conf")
        if production_cfg.get("early_emission","true") in {"true","1"}: comparison += ["--early-emission"]
        with deployment_lock():
            idle(); run(comparison,output / "accuracy.log",14400)
        comparisons=json.loads((output / "accuracy/comparison.json").read_text())
        acceptance_comparisons=comparisons
        if common_holdoff_ms != holdoff_ms:
            # A 50 Hz grid cannot represent an odd 10 ms holdoff. Keep the fair
            # common-policy comparison and separately evaluate the saved policy.
            saved_command=list(comparison)
            offset=saved_command.index("--holdoff-ms"); del saved_command[offset:offset+2]
            offset=saved_command.index("--output-dir"); saved_command[offset+1]=output / "accuracy-saved-policies"
            with deployment_lock():
                idle(); run(saved_command,output / "accuracy-saved-policies.log",14400)
            acceptance_comparisons=json.loads((output / "accuracy-saved-policies/comparison.json").read_text())
        report["stages"]["accuracy"]=dict(accuracy_review(acceptance_comparisons,args),inputs=evidence,
            comparison_holdoff_ms=common_holdoff_ms,candidate_saved_holdoff_ms=holdoff_ms,
            baseline_saved_holdoff_ms=holdoff_frames_from_settings(baseline["cfg"])*20,
            note="Controlled comparison uses a common representable physical holdoff. Acceptance uses the candidate saved policy; a separate saved-policy evaluation is made if rounding was needed. Frame metrics and raw-time diagnostics are separate.")
        runtime_cfg=output / "private-runtime.conf"
        live_device=production_cfg["device"]
        if args.live_uart_seconds and not live_device.startswith("/dev/tty"):
            raise ValueError("Live check requires the configured /dev/tty UART device")
        production_cfg.update(manifest_path=str(namespace_path(bundle / "model/model.conf")),device="/dev/null",
            capture_path="",replay_path="",threshold_override_path="",trigger_output="jsonl",
            allow_gpio_during_replay="false",emit_probabilities="true",realtime_priority="20",require_realtime="true")
        runtime_cfg.write_text("".join(f"{k}={v}\n" for k,v in production_cfg.items()))
        report["stages"]["service"]=dict(runs=[],passed=True,
            differences=["private PTY replaces hardware UART", "JSONL replaces GPIO", "private /run replaces production state",
                "Python pacer plus read-only venv bind", "NotifyAccess=all for native child readiness/watchdog", "Restart=no; private unit collected after exit"])
        with deployment_lock():
            for index in range(args.timing_runs):
                idle()
                result=service_runner(args,profile,bundle,runtime_cfg,output,index,run=run)
                report["stages"]["service"]["runs"].append(result)
                report["stages"]["service"]["passed"] &= result["passed"]
        if args.live_uart_seconds:
            live_cfg=output / "private-live-runtime.conf"
            production_cfg["device"]=live_device
            live_cfg.write_text("".join(f"{k}={v}\n" for k,v in production_cfg.items()))
            with deployment_lock():
                idle()
                live_result=service_runner(args,profile,bundle,live_cfg,output,"live-uart",run=run)
            report["stages"]["service"]["live_uart"]=live_result
            report["stages"]["service"]["passed"] &= live_result["passed"]
        else: report["stages"]["service"]["live_uart"]=dict(status="not_requested")
        passed=report["stages"]["service"]["passed"] and report["stages"]["accuracy"]["status"] != "failed"
        report["status"]=("failed" if not passed else "review_required" if report["stages"]["accuracy"]["status"]=="review_required" or not args.live_uart_seconds else "passed")
    except BaseException as error:
        report["status"]="failed"; report["error"]=f"{type(error).__name__}: {error}"
        raise
    finally:
        report["source_unchanged"]=all(p.is_file() and sha256(p)==h for name,h in hashes.items() for p in [Path(name)])
        if not report["source_unchanged"]: report["status"]="failed"; report["error"]="Source/deployment artifacts changed during qualification"
        (output / "qualification_report.json").write_text(json.dumps(report,indent=2))
        write_summary(report,output / "qualification_report.md")
        print(f"Qualification {report['status']}: {output / 'qualification_report.json'}",flush=True)
    return report


def write_summary(report,path):
    stages=report["stages"]
    accuracy=stages.get("accuracy",{})
    lines=["# Jetson model qualification", "", f"Status: **{report['status']}**", "",
        f"Source/deployment files unchanged: {report.get('source_unchanged')}", "",
        f"Parity: {stages.get('parity',{}).get('passed','not completed')}",
        f"Accuracy: {accuracy.get('status','not completed')}",
        f"Sensitivity: {accuracy.get('sensitivity','unavailable')}",
        f"False positives/min: {accuracy.get('false_positives_per_minute','unavailable')}", "",
        "| Service check | Passed | p50 ms | p95 ms | p99 ms | Max ms | Missed deadlines |",
        "| --- | --- | --- | --- | --- | --- | --- |"]
    service=stages.get("service",{})
    checks=[(f"paced {i}",r) for i,r in enumerate(service.get("runs",[]))]
    if "live_uart" in service: checks.append(("live UART",service["live_uart"]))
    for label,result in checks:
        summary=result.get("summary",{})
        timing=summary.get("processing_ms_including_all_preprocessing_transfers_inference_trigger",{})
        lines.append("| "+" | ".join(map(str,[label,result.get("passed",result.get("status")),
            *[timing.get(k,"—") for k in ["p50","p95","p99","max"]],summary.get("missed_deadlines","—")]))+" |")
        for failure in result.get("failures",[]): lines.append(f"\n{label}: {failure}\n")
    if "error" in report: lines += ["",f"Error: {report['error']}"]
    lines += ["", "The JSON report and per-stage logs contain complete evidence, model/input hashes, recovered/new misses, false-positive changes and boundary/holdoff diagnostics.", "",
        "No training, threshold tuning, production activation or physical GPIO. Physical output measurements must be performed separately and are not verified here. Native deadlines start at host decode; upstream pacing/acquisition delay is reported separately. These tests do not establish a universal worst-case bound."]
    path.write_text("\n".join(lines)+"\n")


def main():
    args=parser().parse_args()
    report=qualify(args)
    if report.get("status")=="failed": return 1
    if report.get("status")=="review_required": return 3
    return 0


if __name__=="__main__":
    raise SystemExit(main())
