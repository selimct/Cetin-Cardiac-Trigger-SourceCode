# Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Developed at the German Cancer Research Center (DKFZ).
"""Paced private-PTY load test, or managed live UART check; GPIO stays off."""
import argparse
from contextlib import ExitStack
import fcntl
import grp
import json
import os
from pathlib import Path
import pty
import pwd
import re
import subprocess
import sys
import time

import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from dkfz.live.deployment_validation import require_idle
from dkfz.live.qualification import PROPERTY_KEYS
from run.summarize_inference_latency import summarize
from dkfz.live.uart_byte_stream import iter_uart_byte_groups, UartByteConversionStats


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seconds", type=int, default=612)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--manifest", required=True, type=Path)
    inputs=p.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--recording", type=Path)
    inputs.add_argument("--live-device", help="Managed qualification only; no physical GPIO")
    p.add_argument("--native-runtime", type=Path, default=ROOT / "build_native/inference/native_uart_live_runtime")
    p.add_argument("--no-early-emission", action="store_true")
    p.add_argument("--realtime-priority", type=int, choices=range(21), default=0)
    p.add_argument("--require-realtime", action="store_true")
    p.add_argument("--runtime-config", type=Path)
    p.add_argument("--managed-service", action="store_true", help=argparse.SUPPRESS)
    return p


def child_environment(env, managed):
    env=dict(env)
    if managed:
        # The native child owns readiness/watchdog (NotifyAccess=all).
        env.pop("WATCHDOG_PID",None)
    return env


def acquire_lock(managed):
    if managed:
        if not re.fullmatch(r"dkfz-qualification-[a-f0-9]{32}",os.environ.get("DKFZ_QUALIFICATION_UNIT","")) or not os.environ.get("NOTIFY_SOCKET"):
            raise ValueError("Managed mode requires the private qualification service")
        lock=(ROOT / ".jetson-control-deployment.lock").open("r")
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            return lock
        lock.close()
        raise ValueError("Managed qualification requires the coordinator deployment lock")

    # Retry logic: wait up to 60 seconds for the lock to become available
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
    return lock


def service_context():
    unit=os.environ["DKFZ_QUALIFICATION_UNIT"]
    query=["systemctl","show",unit+".service"]
    for key in PROPERTY_KEYS: query += ["-p",key]
    properties=subprocess.check_output(query,text=True,timeout=15)
    return dict(unit=unit,uid=os.getuid(),gid=os.getgid(),
        user=pwd.getpwuid(os.getuid()).pw_name,group=grp.getgrgid(os.getgid()).gr_name,
        nice=os.getpriority(os.PRIO_PROCESS,0),
        properties=dict(line.split("=",1) for line in properties.splitlines() if "=" in line))


def stop_process(process):
    """Reap a child even when it ignores the initial termination request."""
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


def main():
    p=parser(); args=p.parse_args()
    if args.seconds<1:p.error("--seconds must be positive; sustained validation requires at least 600 seconds")
    if args.live_device and (not args.managed_service or not args.live_device.startswith("/dev/tty")):
        p.error("Live UART input requires managed qualification and an explicit /dev/tty device")
    run=args.output_dir.resolve()
    if run.is_relative_to((ROOT / "deploy").resolve()):p.error("Output must be outside the deployment tree")
    with ExitStack() as resources:
        resources.enter_context(acquire_lock(args.managed_service))
        run_check(args, run, resources)


def run_check(args, run, resources):
    master=slave=None
    require_idle()
    run.mkdir(parents=True,exist_ok=False)
    stats=UartByteConversionStats(); groups=[]
    if args.recording:
        source=args.recording.resolve(strict=True)
        with source.open("rb") as input_file:groups=[g for _,g in iter_uart_byte_groups(input_file,stats=stats)]
        if not groups:raise ValueError("No complete recorded UART groups")
        master,slave=pty.openpty()
        resources.callback(os.close, master)
        resources.callback(os.close, slave)
        device=os.ttyname(slave)
    else:source=None;device=args.live_device
    command=[str(args.native_runtime.resolve()),"--manifest",str(args.manifest.resolve(strict=True)),
        "--device",device,"--sample-rate","400","--baud-rate","38600",
        "--max-samples",str(args.seconds*400),"--trigger-output","jsonl",
        "--threshold-override",str(run / "unused-threshold-override.conf"),
        "--emit-probabilities","--status-interval-ms","10000",
        "--realtime-priority",str(args.realtime_priority)]
    if args.runtime_config: command += ["--config",str(args.runtime_config.resolve(strict=True))]
    if args.require_realtime:command.append("--require-realtime")
    if args.no_early_emission:command.append("--no-early-emission")
    (run / "command.json").write_text(json.dumps(command,indent=2))
    with (run / "power-mode.txt").open("w") as power:
        subprocess.run(["/usr/sbin/nvpmodel","-q"],stdout=power,stderr=subprocess.STDOUT,check=True)
    with (run / "probabilities.jsonl").open("w") as output,(run / "runtime.log").open("w") as log:
        with ExitStack() as processes:
            process=subprocess.Popen(command,stdout=output,stderr=log,env=child_environment(os.environ,args.managed_service))
            processes.callback(stop_process, process)
            deadline=time.monotonic()+15
            while '"code":"runtime_starting"' not in (run / "runtime.log").read_text():
                if process.poll() is not None or time.monotonic()>deadline:raise RuntimeError("isolated runtime did not initialize")
                time.sleep(.05)
            time.sleep(.2)
            if args.managed_service:(run / "service-context.json").write_text(json.dumps(service_context(),indent=2))
            with (run / "tegrastats.log").open("w") as temperature:
                thermal=subprocess.Popen(["tegrastats","--interval","1000"],stdout=temperature,stderr=subprocess.STDOUT)
                processes.callback(stop_process, thermal)
            lateness=[];started=time.monotonic_ns()
            if groups:
                for index in range(args.seconds*100):
                    target=started+index*10_000_000
                    remaining=(target-time.monotonic_ns())/1e9
                    if remaining>0:time.sleep(remaining)
                    lateness.append(max(0,time.monotonic_ns()-target)/1e6)
                    if process.poll() is not None:raise RuntimeError("native runtime exited during paced input")
                    data=groups[index%len(groups)]
                    if os.write(master,data)!=len(data):raise RuntimeError("partial pseudo-UART write")
                    if index and index%6000==0:print(json.dumps(dict(elapsed_seconds=index/100,groups_sent=index+1)),flush=True)
                elapsed=(time.monotonic_ns()-started)/1e9
                code=process.wait(timeout=15)
            else:
                code=process.wait(timeout=args.seconds+60)
                elapsed=(time.monotonic_ns()-started)/1e9
            metadata=dict(input_kind="paced_pty" if groups else "live_uart",source=str(source) if source else device,
                source_groups=len(groups),parsed_source_stats=stats.__dict__,groups_sent=len(lateness),
                raw_samples_sent=len(lateness)*4 if groups else None,send_elapsed_seconds=elapsed,
                send_rate_raw_samples_per_second=(len(lateness)-1)*4/elapsed if groups else None,
                pacer_lateness_ms={str(q):float(np.percentile(lateness,q)) for q in [50,95,99,100]} if groups else None,
                pacer_delays_over_10ms=sum(v>10 for v in lateness),runtime_exit_code=code,physical_gpio_enabled=False,
                note="Host-decode timing only; no physical GPIO. Recorded groups repeat in paced mode; live mode receives actual UART input without a software pacer.")
            (run / "pacer.json").write_text(json.dumps(metadata,indent=2));print(json.dumps(metadata,indent=2),flush=True)
            records=[json.loads(line) for line in (run / "probabilities.jsonl").read_text().splitlines() if line.strip()]
            report=summarize(records)
            report.update(paced_raw_samples_per_second=metadata["send_rate_raw_samples_per_second"],duration_at_least_10_minutes=args.seconds>=600,
                physical_gpio_enabled=False,input_kind=metadata["input_kind"])
            scheduling=[json.loads(line) for line in (run / "runtime.log").read_text().splitlines()
                        if line.startswith("{") and '"code":"worker_scheduling"' in line]
            report.update(worker_scheduling=scheduling,requested_realtime_priority=args.realtime_priority,
                all_workers_realtime=len(scheduling)==3 and all(r["realtime_applied"] for r in scheduling))
            (run / "latency-summary.json").write_text(json.dumps(report,indent=2))
            if code:raise RuntimeError("isolated native runtime failed")


if __name__=="__main__":main()
