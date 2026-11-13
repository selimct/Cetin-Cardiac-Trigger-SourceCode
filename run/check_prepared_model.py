#!/usr/bin/env python
# Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Developed at the German Cancer Research Center (DKFZ).

"""Verify prepared model fingerprints before a delayed deployment is activated."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dkfz.live.deployment_validation import verify_prepared_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--model-variant", choices=["best", "tuned"])
    parser.add_argument("--no-recovery", action="store_true")
    args = parser.parse_args()
    verify_prepared_report(args.report, args.config, args.engine, args.checkpoint,
                           disable_recovery=args.no_recovery, model_variant=args.model_variant)
    print("Prepared config, engine and checkpoint fingerprints verified")


if __name__ == "__main__":
    main()
