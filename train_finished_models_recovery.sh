#!/usr/bin/env bash
# Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Developed at the German Cancer Research Center (DKFZ).

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$REPO_ROOT"

NUM_EPOCHS=10
MAX_FP_INCREASE=0.10
MIN_EXTRA_FP_PER_MINUTE=0.5
DRY_RUN=0
LOPO=0
ONLY_MODELS=()
CACHE_ARGS=()

usage() {
  cat <<'USAGE'
Usage: ./train_finished_models_recovery.sh [options]

Add a recovery head to each single-output model in finished_models/ while
freezing its LSTM, current head and primary threshold. Train sequentially using
each saved config's data and validation split. Publish separate copies under
finished_models/MODEL/frozen_recovery/{best.pth,train_config.json}.

Completed recovery copies and names ending in _recovery are skipped. A failed
job stops the script; rerun to retry the incomplete model and continue.
Training histories, calibration reports and logs go under models/MODEL_frozen_recovery/.
Joint models publish under MODEL/recovery/ with the name MODEL_recovery.
Use --lopo for nested patient folds and inner-fold epoch/threshold selection.
That mode publishes under MODEL/lopo_recovery/ with name MODEL_lopo_recovery and uses
models/MODEL_lopo_recovery_cv/nested_recovery_lopo/ for CV artifacts.
Only the recovery head is patient-held-out: the frozen base may have seen those
patients during its original training. Original preprocessing stays fixed.
CPU recovery-only training caches the frozen representations in RAM by default.
GPU and unsupported workloads use the original training path.

Options:
  --only MODEL                 Original model name; repeatable. Default: all.
  --num-epochs N               Recovery epochs (LOPO maximum). Default: 10.
  --recovery-max-fp-increase F  Relative validation FP allowance. Default: 0.10.
  --recovery-min-extra-fp-per-minute F
                               Minimum extra-FP allowance per validation minute.
                               Default: 0.5 (one extra per two minutes).
  --dry-run                    Print commands without training or writing files.
  --lopo                       Use nested recovery-head LOPO instead of the saved split.
  --no-cache-frozen-recovery    Use the original CPU training path for comparison.
  --help                       Show this help.

Examples:
  ./train_finished_models_recovery.sh --dry-run
  ./train_finished_models_recovery.sh
  ./train_finished_models_recovery.sh --lopo --dry-run
  ./train_finished_models_recovery.sh --only ecg3 --only slope_ecg3 --num-epochs 5
USAGE
}

die() {
  printf 'train_finished_models_recovery.sh: %s\n' "$*" >&2
  exit 2
}

require_value() {
  [[ -n "${2:-}" ]] || die "$1 requires a value"
}

while (( $# )); do
  case "$1" in
    --only)
      require_value "$1" "${2:-}"
      ONLY_MODELS+=("$2")
      shift 2
      ;;
    --num-epochs)
      require_value "$1" "${2:-}"
      NUM_EPOCHS="$2"
      shift 2
      ;;
    --recovery-max-fp-increase)
      require_value "$1" "${2:-}"
      MAX_FP_INCREASE="$2"
      shift 2
      ;;
    --recovery-min-extra-fp-per-minute)
      require_value "$1" "${2:-}"
      MIN_EXTRA_FP_PER_MINUTE="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --lopo)
      LOPO=1
      shift
      ;;
    --cache-frozen-recovery|--no-cache-frozen-recovery)
      CACHE_ARGS=("$1")
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ -x .venv/bin/python ]] || die "project interpreter missing: .venv/bin/python"
[[ -f run/train_finetune.py ]] || die "missing run/train_finetune.py"
if (( LOPO )); then
  [[ -f run/train_recovery_lopo.py ]] || die "missing run/train_recovery_lopo.py"
fi
[[ "$NUM_EPOCHS" =~ ^[1-9][0-9]*$ ]] || die "--num-epochs must be a positive integer"
.venv/bin/python - "$MAX_FP_INCREASE" "$MIN_EXTRA_FP_PER_MINUTE" <<'PY'
import math
import sys

try:
    value = float(sys.argv[1])
except ValueError:
    raise SystemExit("--recovery-max-fp-increase must be finite and nonnegative")
if not math.isfinite(value) or value < 0:
    raise SystemExit("--recovery-max-fp-increase must be finite and nonnegative")
try:
    rate = float(sys.argv[2])
except ValueError:
    raise SystemExit("--recovery-min-extra-fp-per-minute must be finite and nonnegative")
if not math.isfinite(rate) or rate < 0:
    raise SystemExit("--recovery-min-extra-fp-per-minute must be finite and nonnegative")
PY

# Resolve and check all selected sources before starting any expensive jobs.
configs=()
shopt -s nullglob
for config in finished_models/*/train_config.json; do
  source_dir="${config%/*}"
  name="${source_dir##*/}"
  [[ "$name" == *_recovery ]] && continue
  if (( ${#ONLY_MODELS[@]} )); then
    selected=0
    for requested in "${ONLY_MODELS[@]}"; do
      [[ "$name" == "$requested" ]] && selected=1
    done
    (( selected )) || continue
  fi
  [[ -f "$source_dir/best.pth" ]] || die "missing source checkpoint: $source_dir/best.pth"
  configs+=("$config")
done
shopt -u nullglob
(( ${#configs[@]} )) || die "no matching source models found in finished_models/"
for requested in "${ONLY_MODELS[@]}"; do
  found=0
  for config in "${configs[@]}"; do
    [[ "${config%/train_config.json}" == "finished_models/$requested" ]] && found=1
  done
  (( found )) || die "no source model found for --only $requested"
done

trained=0
skipped=0
for config in "${configs[@]}"; do
  source_dir="${config%/*}"
  name="${source_dir##*/}"
  output_name="${name}_frozen_recovery"
  if (( LOPO )); then
    output_name="${name}_lopo_recovery"
  fi
  destination="$source_dir/frozen_recovery"
  if (( LOPO )); then
    destination="$source_dir/lopo_recovery"
  fi
  [[ ! -L "$destination" ]] || die "recovery destination must not be a symlink: $destination"
  if [[ -f "$destination/train_config.json" && -f "$destination/best.pth" ]]; then
    printf 'Skipping completed recovery model: %s\n' "$output_name"
    skipped=$((skipped + 1))
    continue
  fi

  work_dir="models/$output_name"
  work_checkpoint="$REPO_ROOT/$work_dir/best_tuned.pth"
  source_config="$work_dir/tuning/tuned_config.json"
  if (( LOPO )); then
    work_dir="models/${output_name}_cv/nested_recovery_lopo"
    work_checkpoint="$REPO_ROOT/$work_dir/best.pth"
    source_config="$work_dir/train_config.json"
    command=(.venv/bin/python run/train_recovery_lopo.py --output-dir "$REPO_ROOT/$work_dir" --overwrite)
  else
    command=(.venv/bin/python run/train_finetune.py --recovery-head --recovery-only)
  fi
  command+=(
    --config "$config"
    --checkpoint "$source_dir/best.pth"
    --modelname "$output_name"
    --output-checkpoint "$work_checkpoint"
    --num-epochs "$NUM_EPOCHS"
    --recovery-max-fp-increase "$MAX_FP_INCREASE"
    --recovery-min-extra-fp-per-minute "$MIN_EXTRA_FP_PER_MINUTE"
    "${CACHE_ARGS[@]}"
  )
  printf '\nTraining recovery for %s\nCommand:' "$name"
  printf ' %q' "${command[@]}"
  printf '\nTraining artifacts: %s/\n' "$work_dir"
  printf 'Publish target: %s/{best.pth,train_config.json}\n' "$destination"
  trained=$((trained + 1))
  (( DRY_RUN )) && continue

  log_dir="$work_dir"
  if (( LOPO )); then
    log_dir="${work_dir%/*}"
  fi
  mkdir -p "$log_dir"
  "${command[@]}" 2>&1 | tee "$log_dir/recovery_training.log"
  [[ -s "$work_checkpoint" ]] || die "training did not produce $work_checkpoint"
  # Keep the training artifacts intact; publish the completed deployment pair
  # only after training/calibration and the LOPO frozen-weight audit succeed.
  .venv/bin/python - "$source_config" "$work_checkpoint" "$destination" "$output_name" <<'PY'
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

config_path, checkpoint_path, destination_value, modelname = sys.argv[1:]
config = json.loads(Path(config_path).read_text())
if config.get("modelname") != modelname or not config.get("recovery_head") or not config.get("recovery_only"):
    raise ValueError("Training did not produce the expected frozen recovery model")
destination = Path(destination_value)
config["checkpoint"] = (destination / "best.pth").as_posix()
with tempfile.TemporaryDirectory(dir=destination.parent, prefix=f".{modelname}.publish.") as staging:
    staged = Path(staging)
    shutil.copy2(checkpoint_path, staged / "best.pth")
    (staged / "train_config.json").write_text(json.dumps(config, indent=2) + "\n")
    destination.mkdir(parents=True, exist_ok=True)
    os.replace(staged / "best.pth", destination / "best.pth")
    # Config is the completion marker used when resuming this script.
    os.replace(staged / "train_config.json", destination / "train_config.json")
PY
done

if (( DRY_RUN )); then
  printf '\nDry run: %d model(s) selected for training; %d completed model(s) skipped.\n' "$trained" "$skipped"
else
  printf '\nFinished: %d recovery model(s) trained; %d completed model(s) skipped.\n' "$trained" "$skipped"
fi
