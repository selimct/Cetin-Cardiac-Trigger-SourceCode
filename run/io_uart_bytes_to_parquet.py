#!/usr/bin/env python3
# Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Developed at the German Cancer Research Center (DKFZ).

"""Convert a captured pin-10 UART byte stream to sample-level parquet."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dkfz.live.uart_byte_stream import (
    convert_uart_bytes_to_parquet,
    default_uart_parquet_path,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Convert raw bytes captured by uart_receiver_stream into "
            "centered ecg1_raw/ecg2_raw/ecg3_raw/puls_raw plus "
            "resp_raw/ext parquet rows."
        )
    )
    parser.add_argument("input", help="Raw UART .bin capture.")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output parquet path; defaults to INPUT with a .parquet suffix.",
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=400.0,
        help="ECG sample rate used for time_ms (default: 400).",
    )
    parser.add_argument(
        "--ecg1-offset",
        type=float,
        default=2048.0,
        help="UART ECG1 source offset removed before storage (default: 2048).",
    )
    parser.add_argument(
        "--ecg2-offset",
        type=float,
        default=2048.0,
        help="UART ECG2 source offset removed before storage (default: 2048).",
    )
    parser.add_argument(
        "--puls-offset",
        type=float,
        default=2048.0,
        help="UART pulse source offset removed before storage (default: 2048).",
    )
    parser.add_argument(
        "--batch-rows",
        type=int,
        default=100_000,
        help="Rows buffered before each parquet write (default: 100000).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output or stale partial output.",
    )
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    output_path = (
        Path(args.output)
        if args.output is not None
        else default_uart_parquet_path(input_path)
    )
    try:
        destination, stats = convert_uart_bytes_to_parquet(
            input_path,
            output_path,
            sample_rate_hz=args.sample_rate,
            ecg1_offset=args.ecg1_offset,
            ecg2_offset=args.ecg2_offset,
            puls_offset=args.puls_offset,
            batch_rows=args.batch_rows,
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))

    print(
        "UART parquet written: "
        f"output={destination} bytes={stats.bytes_read} "
        f"packets={stats.packets} samples={stats.samples} "
        f"malformed_packets={stats.malformed_packets} "
        f"malformed_samples={stats.malformed_samples} "
        f"startup_samples_discarded={stats.startup_samples_discarded} "
        f"trailing_samples_discarded={stats.trailing_samples_discarded} "
        f"discarded_bytes={stats.bytes_discarded} "
        f"trailing_bytes={stats.trailing_bytes}",
        file=sys.stderr,
    )
    return 0 if stats.malformed_packets == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
