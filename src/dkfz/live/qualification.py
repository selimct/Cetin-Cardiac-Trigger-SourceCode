# Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Developed at the German Cancer Research Center (DKFZ).
"""Evidence checks shared by the isolated Jetson qualification tools."""
from __future__ import annotations

import math
import re

UNIT_PREFIX = "dkfz-qualification-"
SERVICE_KEYS = (
    "User", "Group", "SupplementaryGroups", "Nice", "CPUWeight", "IOWeight",
    "CPUSchedulingPolicy", "LimitRTPRIO", "LimitRTTIME", "TimerSlackNSec",
    "NoNewPrivileges", "PrivateTmp", "ProtectHome", "ProtectSystem", "WatchdogSec",
    "TimeoutStopSec", "KillSignal",
)
PROPERTY_KEYS = (
    "User", "Group", "SupplementaryGroups", "Nice", "CPUWeight", "IOWeight",
    "CPUSchedulingPolicy", "LimitRTPRIO", "LimitRTTIME", "TimerSlackNSec",
    "NoNewPrivileges", "PrivateTmp", "ProtectHome", "ProtectSystem", "WatchdogUSec",
    "Type", "NotifyAccess", "ReadWritePaths", "BindReadOnlyPaths", "RuntimeDirectory",
)


def settings(path, section=None):
    values = {}
    current = None
    for line in path.read_text().splitlines():
        line = line.partition("#")[0].strip()
        if line.startswith("["):
            current = line[1:-1]
        elif line and (section is None or current == section):
            key, separator, value = line.partition("=")
            if not separator or key in values:
                raise ValueError(f"Malformed or repeated setting in {path}: {line}")
            values[key.strip()] = value.strip()
    return values


def duration_us(value):
    match = re.fullmatch(r"([0-9.]+)(ns|us|ms|s|min)?", str(value))
    if not match:
        raise ValueError(f"Invalid duration: {value}")
    return float(match[1]) * {None: 1, "ns": .001, "us": 1, "ms": 1000, "s": 1e6, "min": 60e6}[match[2]]


def profile_failures(profile, context):
    """Check actual manager properties, not just requested command flags."""
    actual = context.get("properties", {})
    failures = []
    if context.get("user") != profile["User"] or context.get("group") != profile["Group"]:
        failures.append("Process account/group differs from the service")
    keys = set(SERVICE_KEYS) - {"WatchdogSec", "TimeoutStopSec", "KillSignal"}
    for key in keys:
        expected = profile[key]
        observed = actual.get(key, "<missing>")
        if key == "SupplementaryGroups":
            same = set(expected.split()) == set(observed.split())
        elif key == "CPUSchedulingPolicy":
            same = observed == {"other": "0", "fifo": "1", "rr": "2"}.get(expected, expected)
        elif key == "LimitRTTIME":
            try: same = duration_us(expected) == duration_us(observed)
            except ValueError: same = False
        elif key in {"NoNewPrivileges", "PrivateTmp", "ProtectHome"}:
            same = observed.lower() == {"true": "yes", "false": "no"}.get(expected.lower(), expected.lower())
        else:
            same = observed == expected
        if not same:
            failures.append(f"{key}: requested {expected}, observed {observed}")
    if duration_us(actual.get("WatchdogUSec", "0")) != duration_us(profile["WatchdogSec"]):
        failures.append("Service watchdog interval differs")
    if actual.get("Type") != "notify" or actual.get("NotifyAccess") != "all":
        failures.append("Private service notification configuration differs")
    return failures


def valid_count(value):
    """Accept finite, nonnegative JSON counters, including float summaries."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
        and value == int(value)
    )


def performance_failures(summary, pacer, finished, cfg, seconds, *, max_missed=0, max_backlog=8):
    failures = []
    expected_samples = seconds * 400
    expected_frames = expected_samples // int(cfg["frame_size_raw"]) - int(cfg["window_size"]) + 1
    for key, expected in {"samples_decoded": expected_samples, "samples_consumed": expected_samples,
                          "frames_inferred": expected_frames, "exit_code": 0,
                          "invalid_samples": 0, "input_state_resets": 0, "gpio_pulses": 0}.items():
        if finished.get(key) != expected:
            failures.append(f"Runtime {key}: expected {expected}, observed {finished.get(key)}")
    if summary.get("evaluations") != expected_frames or summary.get("all_workers_realtime") is not True:
        failures.append("Evaluation count or verified RT worker coverage differs")
    rate = summary.get("observed_evaluations_per_second", 0) or 0
    hz = 400 / int(cfg["frame_size_raw"])
    if not math.isclose(rate, hz, rel_tol=.002):
        failures.append(f"Evaluation rate {rate} does not match {hz} Hz")
    missed = summary.get("missed_deadlines")
    if not valid_count(missed) or missed > max_missed:
        failures.append("Native deadline budget exceeded")
    backlog = summary.get("backlog_samples", {}).get("max")
    if not valid_count(backlog) or backlog > max_backlog:
        failures.append("Completion backlog budget exceeded")
    if summary.get("frame_end_clock") != "host_decode":
        failures.append("Frame-end clock is unverified")
    triggers = finished.get("trigger_events")
    if not valid_count(triggers) or triggers == 0:
        failures.append("No positive triggers exercised the output path")
    if pacer.get("runtime_exit_code") != 0:
        failures.append("Input runtime did not complete successfully")
    if pacer.get("input_kind") != "live_uart":
        if pacer.get("raw_samples_sent") != expected_samples:
            failures.append("Paced source did not complete the expected samples")
        if not math.isclose(pacer.get("send_rate_raw_samples_per_second", 0), 400, rel_tol=.002):
            failures.append("Input pacing rate differs from 400 Hz")
    return failures


def uart_integrity_failures(diagnostics, finished, *, live=False):
    if any(k not in finished for k in ("malformed_groups", "bytes_discarded")):
        return ["Final UART counters missing; rebuild the native runtime"]
    malformed=max([finished["malformed_groups"],*[r.get("malformed_groups",0) for r in diagnostics]])
    discarded=max([finished["bytes_discarded"],*[r.get("bytes_discarded",0) for r in diagnostics]])
    startup=0
    if live:
        startup=max([0,*[r.get("bytes_discarded",0) for r in diagnostics
            if r.get("code")=="uart_resynchronized" and r.get("decoded_samples_before_chunk")==0 and not r.get("malformed_groups",0)]])
    errors=[]
    if malformed: errors.append("Malformed runtime UART groups")
    if (not live and discarded) or (live and (discarded>startup or startup>51)):
        errors.append("Runtime UART discard outside bounded initial live synchronization")
    return errors
