#!/usr/bin/env python
# Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Developed at the German Cancer Research Center (DKFZ).

"""Compare JSONL emitted by the Python and native TensorRT runtimes."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dkfz.live.native_runtime_parity import compare_native_runtime_outputs  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python-output", required=True)
    parser.add_argument("--native-output", required=True)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    args = parser.parse_args()

    report = compare_native_runtime_outputs(
        args.python_output,
        args.native_output,
        atol=args.atol,
        rtol=args.rtol,
    )
    print(f"Compared {report.probability_frames} probability frames")
    print(f"Compared {report.trigger_events} trigger events")
    print(f"max_probability_abs_diff={report.max_probability_abs_diff:.9g}")
    print("passed=True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
