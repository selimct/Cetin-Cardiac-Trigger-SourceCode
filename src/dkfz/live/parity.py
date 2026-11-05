# Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Developed at the German Cancer Research Center (DKFZ).

"""Shared helpers for export/runtime parity checks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from dkfz.live.onnx_export import LogitOnly, build_model_from_config


def load_inputs_from_prepped(
    prepped_file: str | Path,
    feature_cols: list[str],
    window_size: int,
    samples: int,
) -> np.ndarray:
    import pandas as pd

    df = pd.read_parquet(prepped_file)
    starts: list[int] = []
    if "sid" in df.columns:
        sid = df["sid"].to_numpy()
        for i in range(0, max(0, len(df) - window_size + 1)):
            if sid[i] == sid[i + window_size - 1]:
                starts.append(i)
                if len(starts) >= samples:
                    break
    else:
        starts = list(range(min(samples, max(0, len(df) - window_size + 1))))

    if not starts:
        raise ValueError(
            f"No valid {window_size}-frame windows found in {prepped_file}."
        )

    values = df[feature_cols].to_numpy(dtype=np.float32)
    return np.stack([values[i : i + window_size] for i in starts], axis=0)


def make_parity_inputs(cfg: dict, samples: int, prepped_file: str | None) -> np.ndarray:
    window_size = int(cfg["window_size"])
    input_dim = int(cfg["input_dim"])
    if prepped_file:
        return load_inputs_from_prepped(
            prepped_file=prepped_file,
            feature_cols=list(cfg["feature_cols"]),
            window_size=window_size,
            samples=samples,
        )

    rng = np.random.default_rng(42)
    return rng.normal(size=(samples, window_size, input_dim)).astype(np.float32)


def torch_reference_logits(
    cfg: dict, checkpoint: str | Path, inputs: np.ndarray
) -> np.ndarray:
    model = LogitOnly(build_model_from_config(cfg, checkpoint)).eval()
    with torch.inference_mode():
        return model(torch.from_numpy(inputs)).detach().cpu().numpy()


def print_parity_report(
    *,
    reference: np.ndarray,
    candidate: np.ndarray,
    samples: int,
    rtol: float,
    atol: float,
) -> bool:
    reference = reference.reshape(-1)
    candidate = candidate.reshape(-1)
    diff = np.abs(reference - candidate)
    max_abs = float(diff.max()) if diff.size else 0.0
    mean_abs = float(diff.mean()) if diff.size else 0.0
    passed = bool(np.allclose(reference, candidate, rtol=rtol, atol=atol))

    print(f"Compared {samples} windows")
    print(f"max_abs_diff={max_abs:.8g}")
    print(f"mean_abs_diff={mean_abs:.8g}")
    print(f"passed={passed}")
    return passed


def print_performance_report(
    *,
    durations_ns: np.ndarray,
    elapsed_ns: int,
    min_throughput_hz: float,
    max_p99_latency_ms: float,
) -> bool:
    """Report and enforce sustained-rate and tail-latency requirements."""
    durations_ns = np.asarray(durations_ns, dtype=np.float64).reshape(-1)
    if durations_ns.size == 0:
        raise ValueError("At least one measured inference duration is required.")
    if not np.all(np.isfinite(durations_ns)) or np.any(durations_ns < 0):
        raise ValueError("Inference durations must be finite and non-negative.")
    if int(elapsed_ns) <= 0:
        raise ValueError("Measured elapsed time must be positive.")
    if float(min_throughput_hz) <= 0:
        raise ValueError("Minimum throughput must be positive.")
    if float(max_p99_latency_ms) <= 0:
        raise ValueError("Maximum p99 latency must be positive.")

    durations_ms = durations_ns / 1_000_000.0
    throughput_hz = durations_ns.size * 1_000_000_000.0 / int(elapsed_ns)
    mean_latency_ms = float(durations_ms.mean())
    p95_latency_ms = float(np.percentile(durations_ms, 95))
    p99_latency_ms = float(np.percentile(durations_ms, 99))
    max_latency_ms = float(durations_ms.max())
    passed = bool(
        throughput_hz >= float(min_throughput_hz)
        and p99_latency_ms <= float(max_p99_latency_ms)
    )

    print(f"mean_inference_latency_ms={mean_latency_ms:.8g}")
    print(f"p95_inference_latency_ms={p95_latency_ms:.8g}")
    print(f"p99_inference_latency_ms={p99_latency_ms:.8g}")
    print(f"max_inference_latency_ms={max_latency_ms:.8g}")
    print(f"sustained_inference_rate_hz={throughput_hz:.8g}")
    print(f"required_min_inference_rate_hz={float(min_throughput_hz):.8g}")
    print(f"required_max_p99_latency_ms={float(max_p99_latency_ms):.8g}")
    print(f"performance_passed={passed}")
    return passed
