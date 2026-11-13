# Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Developed at the German Cancer Research Center (DKFZ).

"""Summarize native UART processing telemetry from a sustained Jetson run.

Use live UART or paced replay. Unpaced replay cannot establish real-time
performance. Latencies use host monotonic time; physical input/GPIO edge
validation requires independently captured hardware edges.
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dkfz.live.raw_event_evaluation import distribution


def summarize(records):
    probabilities = [
        r for r in records if r.get("schema") == "dkfz-live-probability-v1"
    ]
    required = (
        "processing_duration_ns",
        "frame_end_to_completion_ns",
        "completed_monotonic_ns",
        "backlog_samples",
        "missed_deadline",
    )
    if not probabilities or any(
        any(k not in r for k in required) for r in probabilities
    ):
        raise ValueError(
            "Missing native UART probability telemetry; enable --emit-probabilities and JSONL output"
        )
    elapsed = (
        probabilities[-1]["completed_monotonic_ns"]
        - probabilities[0]["completed_monotonic_ns"]
    ) / 1e9
    decode_clock_verified = all(
        r.get("decoded_monotonic_ns", 0) > 0
        and r["completed_monotonic_ns"] - r["decoded_monotonic_ns"]
        == r["frame_end_to_completion_ns"]
        for r in probabilities
    )
    events = [r for r in records if r.get("schema") == "dkfz-live-trigger-v1"]
    gpio = [
        (r["gpio_asserted_monotonic_ns"] - r["decision_monotonic_ns"]) / 1e6
        for r in events
        if r.get("gpio_asserted_monotonic_ns") is not None
        and r.get("decision_monotonic_ns") is not None
    ]
    return {
        "evaluations": len(probabilities),
        "elapsed_seconds": elapsed,
        "frame_end_clock": (
            "host_decode" if decode_clock_verified else "unverified_legacy_clock"
        ),
        "observed_evaluations_per_second": (
            (len(probabilities) - 1) / elapsed if elapsed > 0 else None
        ),
        "processing_ms_including_all_preprocessing_transfers_inference_trigger": distribution(
            [r["processing_duration_ns"] / 1e6 for r in probabilities]
        ),
        "decoder_frame_end_to_completion_ms": distribution(
            [r["frame_end_to_completion_ns"] / 1e6 for r in probabilities]
        ),
        "stages_ms": {
            field.removesuffix("_ns"): distribution([r[field] / 1e6 for r in probabilities])
            for field in ("preprocessing_ns", "window_assembly_ns", "inference_ns", "trigger_ns",
                          "cpu_thread_ns", "sample_queue_wait_ns", "gpu_upload_ns", "gpu_compute_ns", "gpu_download_ns")
            if all(field in r for r in probabilities)
        },
        "cuda_graph_evaluations": sum(bool(r.get("cuda_graph")) for r in probabilities),
        "processing_over_10ms": sum(r["processing_duration_ns"] > 10_000_000 for r in probabilities),
        "backlog_samples": distribution([r["backlog_samples"] for r in probabilities]),
        "missed_deadlines": sum(bool(r["missed_deadline"]) for r in probabilities),
        "decision_to_gpio_assertion_ms": distribution(gpio),
        "note": "GPIO timestamps follow the output ioctl; verify physical edges separately. This summary alone does not certify sustained real-time operation or input validity.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl")
    args = parser.parse_args()
    records = []
    with Path(args.jsonl).open() as source:
        for line in source:
            if line.strip():
                records.append(json.loads(line))
    print(json.dumps(summarize(records), indent=2))


if __name__ == "__main__":
    main()
