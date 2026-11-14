#!/usr/bin/env bash
# Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Developed at the German Cancer Research Center (DKFZ).

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$REPO_ROOT"

usage() {
  cat <<'USAGE'
Usage: ./metrics.sh [normal|synthetic] [options]

Evaluate all published normal and recovery models in finished_models/ on the
six preferred recordings: p02 head, p14 heart/head, p16 head/heart/pelvis.
Both modes use --only-missing and generate pooled and per-input reports.
Relative paths in additional options are resolved from the project root.

Modes:
  normal                    Original recordings; default when no mode is given.
                            Reports: metrics/
  synthetic                 Full gradient-frequency sweep.
                            Reports: metrics/synthetic/

Options:
  --exclude-recovery-models Run only normal 50 Hz and delayed 100 Hz models;
                            skip all recovery model versions.
  --save-prepped            Keep full prediction/event parquets under metrics/prepped/
                            with the same normal/synthetic/model subdirectories.
                            Default: keep only small report caches; reuse saved
                            parquets and matching predictions on subsequent runs.
  --dry-run                 Print the command without running inference or plots.
  --help, -h                Show this help.
  Remaining options         Passed to create_prediction_metrics.sh, for example
                            --model ecg3_recovery or --output-dir metrics/custom.
                            Use -- before offline_predict.py overrides.

Examples:
  ./metrics.sh normal
  ./metrics.sh normal --exclude-recovery-models
  ./metrics.sh synthetic
  ./metrics.sh normal --save-prepped
  ./metrics.sh synthetic --save-prepped
  ./metrics.sh normal --model ecg3 --model ecg3_recovery
USAGE
}

MODE="normal"
case "${1:-}" in
  normal|synthetic) MODE="$1"; shift ;;
  ""|-*) ;;
  *) printf 'metrics.sh: unknown mode: %s\n' "$1" >&2; exit 2 ;;
esac

DRY_RUN=0
SAVE_PREPPED=0
FORWARDED_ARGS=()
while (( $# )); do
  case "$1" in
    --help|-h) usage; exit 0 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --save-prepped) SAVE_PREPPED=1; shift ;;
    --no-save-prepped|--no-save-synthetic) SAVE_PREPPED=0; shift ;;
    --) FORWARDED_ARGS+=("$@"); break ;;
    *) FORWARDED_ARGS+=("$1"); shift ;;
  esac
done

COMMAND=("$REPO_ROOT/create_prediction_metrics.sh" finished_models --only-missing)
if [[ "$SAVE_PREPPED" -eq 1 ]]; then
  COMMAND+=(--save-prepped)
fi
if [[ "$MODE" == "synthetic" ]]; then
  COMMAND+=(--synthetic-gradient)
fi
COMMAND+=(
  data/true/p02/p02_head.parquet
  data/true/p14/p14_head.parquet
  data/true/p14/p14_heart.parquet
  data/true/p16/p16_head.parquet
  data/true/p16/p16_heart.parquet
  data/true/p16/p16_pelvis.parquet
  "${FORWARDED_ARGS[@]}"
)

if [[ "$DRY_RUN" -eq 1 ]]; then
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi

exec "${COMMAND[@]}"
