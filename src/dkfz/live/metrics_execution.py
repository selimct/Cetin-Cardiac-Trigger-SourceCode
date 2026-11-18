"""Schedule independent benchmark recordings and describe comparable runs.

This module deliberately avoids importing Torch: workers start fresh interpreters
and create their own CUDA contexts through the normal prediction entrypoint.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess


def prediction_arguments(arguments):
    """Normalize CLI spelling/order; saving source data does not affect scores."""
    result = {}
    arguments = iter(arguments)
    pending = None
    while True:
        option = pending if pending is not None else next(arguments, None)
        pending = None
        if option is None:
            break
        if not option.startswith("--"):
            raise ValueError(f"Expected a prediction option, got {option!r}")
        if "=" in option:
            option, value = option.split("=", 1)
        else:
            value = next(arguments, None)
            if value is None or value.startswith("--"):
                pending, value = value, True
        if option == "--save-synthetic":
            continue
        if isinstance(value, str):
            try:
                value = float(value)
            except ValueError:
                pass
        result[option] = value
    if result.get("--synthetic-gradient"):
        result.setdefault("--synthetic-noise-multiplier", 1.0)
        if result.get("--synthetic-independent-leads"):
            result.pop("--synthetic-lead-phase-deg", None)
        else:
            result.setdefault("--synthetic-lead-phase-deg", -30.0)
    return result


def run_parallel(*, jobs, inputs, script, arguments, work_dir, records):
    """Run one isolated shell worker per recording, then collect in input order."""
    work_dir.mkdir(parents=True, exist_ok=True)
    separator = arguments.index("--") if "--" in arguments else len(arguments)
    environment = os.environ.copy()
    environment.setdefault("OMP_NUM_THREADS", "1")
    environment.setdefault("MKL_NUM_THREADS", "1")

    def run(item):
        index, source = item
        destination = work_dir / f"{index}.jsonl"
        command = [
            "bash",
            str(script),
            *arguments[:separator],
            source,
            "--worker-records",
            str(destination),
            *arguments[separator:],
        ]
        result = subprocess.run(command, env=environment, check=False)
        return result.returncode, destination

    # Threads only wait for subprocesses; no CUDA state is forked from Python.
    with ThreadPoolExecutor(max_workers=min(jobs, len(inputs))) as executor:
        results = list(executor.map(run, enumerate(inputs)))
    with records.open("w", encoding="utf-8") as stream:
        for _, destination in results:
            if destination.exists():
                stream.write(destination.read_text(encoding="utf-8"))
    return int(any(status for status, _ in results))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    settings = subparsers.add_parser("settings")
    settings.add_argument("--output", type=Path, required=True)
    settings.add_argument("--variant", required=True)
    settings.add_argument("--include-outer-position-errors", action="store_true")
    settings.add_argument("--input", action="append", required=True)
    settings.add_argument("--frequency", type=float, action="append")
    settings.add_argument("arguments", nargs=argparse.REMAINDER)
    parallel = subparsers.add_parser("parallel")
    parallel.add_argument("--jobs", type=int, required=True)
    parallel.add_argument("--input", action="append", required=True)
    parallel.add_argument("--script", type=Path, required=True)
    parallel.add_argument("--work-dir", type=Path, required=True)
    parallel.add_argument("--records", type=Path, required=True)
    parallel.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    arguments = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    if args.action == "settings":
        data = dict(
            variant=args.variant,
            inputs=sorted(str(Path(p).resolve()) for p in args.input),
            input_signatures={
                str(Path(p).resolve()): [
                    Path(p).stat().st_size,
                    Path(p).stat().st_mtime_ns,
                ]
                for p in args.input
            },
            frequencies=sorted(set(args.frequency)) if args.frequency else [None],
            include_outer_position_errors=args.include_outer_position_errors,
            prediction_arguments=prediction_arguments(arguments),
        )
        args.output.write_text(json.dumps(data, sort_keys=True), encoding="utf-8")
        return 0
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    return run_parallel(
        jobs=args.jobs,
        inputs=args.input,
        script=args.script,
        arguments=arguments,
        work_dir=args.work_dir,
        records=args.records,
    )


if __name__ == "__main__":
    raise SystemExit(main())
