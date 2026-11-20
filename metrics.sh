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

Evaluate published non-recovery 100 Hz models in finished_models/ on the six preferred
recordings: p02 head, p14 heart/head, p16 head/heart/pelvis.
Both modes use --only-missing and generate pooled and per-input reports.
Relative paths in additional options are resolved from the project root.

Modes:
  normal                    Original recordings; default when no mode is given.
                            Reports: metrics/
  synthetic                 Full gradient-frequency sweep at 0.6 noise amplitude.
                            Reports: metrics/synthetic/

Options:
  --jobs N                   Evaluate up to N recordings concurrently (default 1).
                            CUDA workers share the selected GPU.
  --word-tables             Update metrics/Tables_metrics.docx (both modes share
                            the document). Three 100 Hz models; omit 90 Hz in Word.
  --word-tables-only        Create/update both Word tables from saved CSV reports.
                            Never run metrics, predictions, or plots. Both modes
                            are exported regardless of the normal/synthetic mode.
                            Supports --output-dir DIR and --word-output FILE.
  --normal-csv FILE         Normal report for --word-tables-only. Default:
                            <output-dir>/prediction_metrics_overview.csv
  --synthetic-csv FILE      Synthetic report for --word-tables-only. Default:
                            <output-dir>/synthetic/prediction_metrics_overview.csv
  --synthetic-dir DIR       Use this synthetic report directory for tables only.
                            Without an explicit source, an interactive terminal
                            lists available synthetic report directories.
  --word-output FILE        Use a custom Word document path; implies --word-tables.
  --add-model MODEL          Add this model to existing comparison tables without
                            evaluating the other models. Repeatable.
  --recovery-models         Include recovery, frozen recovery, and LOPO recovery
                            model versions. Default: exclude recovery models.
  --include-50hz      Also include 50 Hz models. Naming one with --model
                            explicitly selects it as well.
  --save-prepped            Keep full prediction/event parquets (default) under metrics/prepped/
                            with the same normal/synthetic/model subdirectories.
                            Git ignores these directories. Matching files are reused.
  --no-save-prepped         Keep only compact caches and reports; full prediction
                            and event parquets are temporary for this run.
  --dry-run                 Print the command without running inference or plots.
  --help, -h                Show this help.
  Remaining options         Passed to create_prediction_metrics.sh, for example
                            --model ecg3_recovery or --output-dir metrics/custom.
                            Use -- before offline_predict.py overrides.
                            Full noise: -- --synthetic-noise-multiplier 1

Examples:
  ./metrics.sh normal
  ./metrics.sh normal --recovery-models
  ./metrics.sh synthetic
  ./metrics.sh normal --no-save-prepped
  ./metrics.sh normal --recovery-models --model ecg3 --model ecg3_recovery
  ./metrics.sh normal --jobs 2
  ./metrics.sh normal --add-model slope_ecg3_delayed_100hz --jobs 2
  ./metrics.sh --word-tables-only
  ./metrics.sh --word-tables-only --synthetic-dir metrics/synthetic_0p5
USAGE
}

MODE="normal"
case "${1:-}" in
  normal|synthetic) MODE="$1"; shift ;;
  ""|-*) ;;
  *) printf 'metrics.sh: unknown mode: %s\n' "$1" >&2; exit 2 ;;
esac

DRY_RUN=0
WORD_TABLES_ONLY=0
SAVE_PREPPED=1
FORWARDED_ARGS=()
while (( $# )); do
  case "$1" in
    --help|-h) usage; exit 0 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --word-tables-only) WORD_TABLES_ONLY=1; shift ;;
    --save-prepped) SAVE_PREPPED=1; shift ;;
    --no-save-prepped|--no-save-synthetic) SAVE_PREPPED=0; shift ;;
    --) FORWARDED_ARGS+=("$@"); break ;;
    *) FORWARDED_ARGS+=("$1"); shift ;;
  esac
done

if [[ "$WORD_TABLES_ONLY" -eq 1 ]]; then
  WORD_ROOT="metrics"
  WORD_OUTPUT=""
  WORD_NORMAL_CSV=""
  WORD_SYNTHETIC_CSV=""
  WORD_SYNTHETIC_DIR=""
  WORD_PYTHON=".venv/bin/python"
  WORD_MODELS_ROOT="finished_models"
  WORD_CURRENT_ONLY=0
  set -- "${FORWARDED_ARGS[@]}"
  while (( $# )); do
    case "$1" in
      --output-dir|--word-output|--normal-csv|--synthetic-csv|--synthetic-dir|--python|--models-root)
        if (( $# < 2 )); then
          printf 'metrics.sh: %s requires a value\n' "$1" >&2
          exit 2
        fi
        case "$1" in
          --output-dir) WORD_ROOT="$2" ;;
          --word-output) WORD_OUTPUT="$2" ;;
          --normal-csv) WORD_NORMAL_CSV="$2" ;;
          --synthetic-csv) WORD_SYNTHETIC_CSV="$2" ;;
          --synthetic-dir) WORD_SYNTHETIC_DIR="$2" ;;
          --python) WORD_PYTHON="$2" ;;
          --models-root) WORD_MODELS_ROOT="$2" ;;
        esac
        shift 2
        ;;
      --no-recovery) WORD_CURRENT_ONLY=1; shift ;;
      --word-tables) shift ;;
      *) printf 'metrics.sh: unsupported option with --word-tables-only: %s\n' "$1" >&2; exit 2 ;;
    esac
  done
  if [[ "$WORD_CURRENT_ONLY" -eq 1 ]]; then WORD_ROOT+="/current_only"; fi
  if [[ -n "$WORD_SYNTHETIC_DIR" ]]; then
    if [[ -n "$WORD_SYNTHETIC_CSV" ]]; then
      printf 'metrics.sh: use either --synthetic-dir or --synthetic-csv\n' >&2
      exit 2
    fi
    WORD_SYNTHETIC_CSV="${WORD_SYNTHETIC_DIR%/}/prediction_metrics_overview.csv"
  fi
  if [[ -z "$WORD_SYNTHETIC_CSV" && "$DRY_RUN" -eq 0 ]]; then
    WORD_SYNTHETIC_DIRS=()
    for directory in "$WORD_ROOT"/synthetic*/; do
      [[ -f "${directory}prediction_metrics_overview.csv" ]] || continue
      WORD_SYNTHETIC_DIRS+=("${directory%/}")
    done
    if [[ -t 0 && ${#WORD_SYNTHETIC_DIRS[@]} -gt 0 ]]; then
      printf 'Choose saved synthetic reports (no metrics will run):\n' >&2
      PS3='Synthetic directory number: '
      select directory in "${WORD_SYNTHETIC_DIRS[@]}"; do
        if [[ -n "$directory" ]]; then
          WORD_SYNTHETIC_CSV="$directory/prediction_metrics_overview.csv"
          break
        fi
        printf 'Please enter one of the listed numbers.\n' >&2
      done
      [[ -n "$WORD_SYNTHETIC_CSV" ]] || exit 2
    elif [[ ! -f "$WORD_ROOT/synthetic/prediction_metrics_overview.csv" ]]; then
      printf 'metrics.sh: no default synthetic report. Use --synthetic-dir DIR or --synthetic-csv FILE.\n' >&2
      if (( ${#WORD_SYNTHETIC_DIRS[@]} )); then
        printf 'Available saved synthetic directories:\n' >&2
        printf '  %s\n' "${WORD_SYNTHETIC_DIRS[@]}" >&2
      fi
      exit 2
    fi
  fi
  COMMAND=("$WORD_PYTHON" "$REPO_ROOT/run/create_metrics_word_tables.py"
    --models-root "$WORD_MODELS_ROOT"
    --normal-csv "${WORD_NORMAL_CSV:-$WORD_ROOT/prediction_metrics_overview.csv}"
    --synthetic-csv "${WORD_SYNTHETIC_CSV:-$WORD_ROOT/synthetic/prediction_metrics_overview.csv}"
    --output "${WORD_OUTPUT:-$WORD_ROOT/Tables_metrics.docx}")
  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf '%q ' "${COMMAND[@]}"
    printf '\n'
    exit 0
  fi
  exec "${COMMAND[@]}"
fi

COMMAND=("$REPO_ROOT/create_prediction_metrics.sh" finished_models --only-missing)
if [[ "$SAVE_PREPPED" -eq 1 ]]; then
  COMMAND+=(--save-prepped)
else
  COMMAND+=(--no-save-prepped)
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
