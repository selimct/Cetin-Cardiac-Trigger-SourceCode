#!/usr/bin/env bash
# Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Developed at the German Cancer Research Center (DKFZ).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"
LOPO_PYTHON="${LOPO_PYTHON:-.venv/bin/python}"

METRICS_SUMMARY="models/lopo_metrics_overview.txt"
FINISHED_MODELS_ROOT="finished_models"
RECIPE_DIR="config/lopo_recipes"
PATIENT_ROOT="${PATIENT_ROOT:-data/live}"
DRY_RUN=0
OVERWRITE=0
RUN_COUNT=0
ONLY_MODELS=()
RECOVERY_HEAD=1
MODEL_KIND="recovery"
RECOVERY_MAX_FP_INCREASE="0.10"
RECOVERY_ALLOWANCE_SET=0
MAX_PARALLEL=1
START_DELAY_MINUTES=20
MAX_EPOCHS=""
BASE_SEED=42
RECIPES=(ecg3 slope_ecg3 slope_slow_ecg3 slope_ecg3_cqm4ecg slope_slow_ecg3_cqm4ecg)
SELECTED_MODELS=()
SOURCE_CONFIGS=()
EPOCH_LIMITS=()
SKIPPED_MODELS=()
TRAINING_COMMAND=()
ACTIVE_PIDS=()
declare -A JOB_MODELS=()

usage() {
  cat <<'USAGE'
Usage: ./train_models_lopo.sh [options]
       ./train_models_lopo.sh delayed-100hz [options]
       ./train_models_lopo.sh recovery-only [recovery LOPO options]
       ./train_models_lopo.sh reselect-thresholds [reselection options]

Train fresh two-output nested LOPO models for ecg3, slope_ecg3,
slope_slow_ecg3, slope_ecg3_cqm4ecg and slope_slow_ecg3_cqm4ecg.
Read architecture and preprocessing settings from config/lopo_recipes/MODEL.json,
with new random weights for the LSTM and both heads.
Publish under finished_models/MODEL/recovery/.
The delayed-100hz subcommand instead trains one previous-frame primary at
100 Hz and publishes finished_models/MODEL/delayed_100hz/ with model name
MODEL_delayed_100hz. It preserves inherited physical window/history durations,
uses fresh weights, and disables the recovery head.
The command in build_training_command sets LR to 0.001 for every model,
overriding the recipe's learning rate.
No source checkpoint is loaded and no HPO search is performed.
Set LOPO_PYTHON to use an existing environment outside .venv (for example,
/home/orin/envs/jet/bin/python on the Jetson).

Options:
  --patient-root DIR  Patient directory root. Default: data/live, or the
                      PATIENT_ROOT environment variable.
  --recipe-dir DIR    Training recipe directory. Default: config/lopo_recipes.
                      Recipes are independent of models/ and finished_models/.
  --only MODEL        Run only this model. May be repeated.
  --recovery-head     Accepted for compatibility in the default recovery mode.
                      Incompatible with delayed-100hz. --only accepts the recipe
                      name or MODEL_recovery / MODEL_delayed_100hz for its mode.
  --recovery-max-fp-increase FRACTION
                      Validation FP allowance for recovery. Default: 0.10 (10%).
                      Primary threshold stays fixed during recovery calibration.
  --parallel N        Maximum simultaneous training jobs. Default: 1 (sequential).
  --start-delay-minutes N
                      Minimum time between starts in parallel mode. Default: 20.
                      Use 0 to start jobs immediately up to the parallel limit.
  --num-epochs N      Override the maximum epoch search budget. Otherwise use
                      configured_max_epochs from each recipe, falling
                      back to its num_epochs, then 35.
  --seed N            Base training seed. Default: 42.
  --overwrite         Force retraining even when an identical completed run exists.
  --dry-run           Print commands without training or writing summaries.
  --help              Show this help.

Parallel jobs share the available GPU and CPU resources. The start delay does
not cap GPU memory use. Failures stop the remaining jobs and return an error.
Changed settings, recording contents or training code automatically replace
the same model's existing nested_lopo output. Identical completed runs are
reported and skipped. Incomplete/unverifiable runs are replaced as well.

Examples:
  ./train_models_lopo.sh --dry-run
  ./train_models_lopo.sh --only ecg3
  ./train_models_lopo.sh --parallel 4 --start-delay-minutes 20 --dry-run
  ./train_models_lopo.sh --parallel 2 --start-delay-minutes 20
  ./train_models_lopo.sh --only slope_slow_ecg3
  ./train_models_lopo.sh --patient-root data/live

For two-output threshold calibration, use:
  ./update_model_thresholds.sh --help
To preserve the original detector, use recovery-only. It adds frozen-head
nested LOPO to finished_models sources and publishes MODEL_lopo_recovery.
The reselect-thresholds subcommand supports single-output models only.
USAGE
}

die() {
  printf 'train_models_lopo.sh: %s\n' "$*" >&2
  exit 2
}

init_metrics_summary() {
  mkdir -p models
  printf "%-40s %7s %10s %10s %10s %10s %10s %10s %12s %12s %12s\n" \
    "model" "epoch" "threshold" "recov th" "Se" "+P" "DER" "macro DER" \
    "PE/target" "FP/pred" "FN/target" > "$METRICS_SUMMARY"
  printf "%-40s %7s %10s %10s %10s %10s %10s %10s %12s %12s %12s\n" \
    "-----" "-----" "---------" "--------" "---" "---" "---" "---------" \
    "---------" "-------" "---------" >> "$METRICS_SUMMARY"
}

append_metrics_overview() {
  local modelname="$1"
  local summary="models/${modelname}_cv/nested_lopo/summary.json"

  "$LOPO_PYTHON" -c '
import json
from pathlib import Path
import sys

model, path = sys.argv[1:]
with open(path, encoding="utf-8") as stream:
    summary = json.load(stream)

event = summary["pooled_event_position"]
best_model = summary["best_model"]
epoch = int(best_model["selected_epoch"])
threshold = float(best_model["threshold"])
config = json.loads(Path(path).with_name("train_config.json").read_text())
recovery_threshold = config.get("recovery_threshold")
recovery = (
    f"{float(recovery_threshold):.4f}"
    if recovery_threshold is not None
    else "disabled" if config.get("recovery_head") else "off"
)
se = 100.0 * event["exact_sensitivity"]
pp = 100.0 * event["exact_positive_predictive_value"]
der = 100.0 * event["unique_detection_error"]
macro_der = 100.0 * summary["macro_patient_metrics"][
    "unique_detection_error"
]["mean"]
targets = event["num_target_events"]
predictions = event["num_predicted_events"]
pe_rate = 100.0 * event["position_error"] / targets if targets else 0.0
fp_rate = (
    100.0 * event["unmatched_false_positive"] / predictions
    if predictions
    else 0.0
)
fn_rate = 100.0 * event["unmatched_false_negative"] / targets if targets else 0.0
print(
    f"{model:<40} "
    f"{epoch:7d} "
    f"{threshold:10.4f} "
    f"{recovery:>10} "
    f"{se:9.2f}% "
    f"{pp:9.2f}% "
    f"{der:9.2f}% "
    f"{macro_der:9.2f}% "
    f"{pe_rate:11.2f}% "
    f"{fp_rate:11.2f}% "
    f"{fn_rate:11.2f}%"
)
' "$modelname" "$summary" >> "$METRICS_SUMMARY"
}

publish_best_model() {
  local modelname="$1"
  local output_dir="$2"
  local finished_dir="${FINISHED_MODELS_ROOT}/${modelname%_${MODEL_KIND}}/${MODEL_KIND}"

  "$LOPO_PYTHON" - "$modelname" "$output_dir" "$finished_dir" "${3:-always}" <<'PY'
from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import shutil
import sys
import tempfile

from dkfz.live.onnx_export import build_model_from_config
from dkfz.live.training_config import TRAINING_DEFAULTS


modelname, output_value, finished_value, mode = sys.argv[1:]
repo_root = Path.cwd().resolve()
output_dir = Path(output_value)
finished_dir = Path(finished_value)
source_checkpoint = output_dir / "best.pth"
source_config = output_dir / "train_config.json"

for label, path in (
    ("best checkpoint", source_checkpoint),
    ("training config", source_config),
):
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")

if finished_dir.is_symlink():
    raise ValueError(f"Refusing symlinked finished-model directory: {finished_dir}")
if finished_dir.exists() and not finished_dir.is_dir():
    raise ValueError(f"Finished-model path is not a directory: {finished_dir}")
finished_dir.parent.mkdir(parents=True, exist_ok=True)

with source_config.open(encoding="utf-8") as stream:
    config = json.load(stream)

if modelname.endswith("_delayed_100hz"):
    from dkfz.live.prediction_contract import prediction_delay
    if prediction_delay(config) != 1:
        raise ValueError("Only separately trained delayed_100hz weights can be published in delayed_100hz/")

config["modelname"] = modelname
config["base_modelname"] = modelname
config["prepped_file"] = None
for external_artifact_key in (
    "selection_search_file",
    "hpo_search_config",
    "hpo_source_config",
    "hpo_search_file",
    "hpo_selected_parameters_file",
):
    config.pop(external_artifact_key, None)
finished_checkpoint = (repo_root / finished_dir / "best.pth").resolve()
try:
    config["checkpoint"] = finished_checkpoint.relative_to(repo_root).as_posix()
except ValueError:
    config["checkpoint"] = finished_checkpoint.as_posix()

if mode == "if-needed":
    # Runtime threshold changes can legitimately differ from the training copy.
    # Preserve them when weights and the model/preprocessing contract still match.
    mutable = {
        "manual_threshold", "min_distance", "error_association_frames",
        "threshold_objective", "threshold_sensitivity_efficiency",
        "recovery_max_fp_increase", "inputs", "threshold_inputs", "prepped_file",
        "save_last", "save_prepped", "use_prepped", "prep_workers", "debug_timing",
    }
    contract_keys = (set(TRAINING_DEFAULTS) - mutable) | {"input_dim", "feature_cols"}
    def digest(path):
        value = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                value.update(block)
        return value.digest()
    try:
        published = json.loads((finished_dir / "train_config.json").read_text())
        current = (
            isinstance(published, dict)
            and all(published.get(key) == config.get(key) for key in contract_keys)
            and digest(finished_dir / "best.pth") == digest(source_checkpoint)
        )
    except (OSError, ValueError):
        current = False
    if current:
        print(f"Published model already current: {finished_dir}")
        raise SystemExit(0)

with tempfile.TemporaryDirectory(
    dir=finished_dir.parent,
    prefix=f".{modelname}.publish.",
) as stage_value:
    stage_dir = Path(stage_value)
    staged_checkpoint = stage_dir / "best.pth"
    staged_config = stage_dir / "train_config.json"
    shutil.copy2(source_checkpoint, staged_checkpoint)
    with staged_config.open("w", encoding="utf-8") as stream:
        json.dump(config, stream, indent=2)
        stream.write("\n")

    # Validate model shapes and checkpoint contents before publishing either file.
    build_model_from_config(config, staged_checkpoint)

    finished_dir.mkdir(parents=True, exist_ok=True)
    os.replace(staged_checkpoint, finished_dir / "best.pth")
    os.replace(staged_config, finished_dir / "train_config.json")
PY

  printf 'Finished model: %s/{best.pth,train_config.json}\n' "$finished_dir"
}

should_run() {
  local modelname="$1"
  local selected

  if (( ${#ONLY_MODELS[@]} == 0 )); then
    return 0
  fi
  for selected in "${ONLY_MODELS[@]}"; do
    if [[ "$selected" == "$modelname" ]]; then
      return 0
    fi
  done
  return 1
}

build_training_command() {
  local modelname="${1}_${MODEL_KIND}"
  local source_config="$2"
  local epoch_limit="$3"
  local output_dir="models/${modelname}_cv/nested_lopo"
  TRAINING_COMMAND=(
    "$LOPO_PYTHON" run/train_lopo.py
    --patient-root "$PATIENT_ROOT"
    --config "$source_config"
    --modelname "$modelname"
    --num-epochs "$epoch_limit"
    --lr 0.001
    --seed "$BASE_SEED"
    --prepped-file "$output_dir/patient_frames.parquet"
    --save-prepped
    --no-use-prepped
    --set recovery_only=false
    --set manual_threshold=null
  )
  if (( RECOVERY_HEAD )); then
    TRAINING_COMMAND+=(--recovery-head --recovery-max-fp-increase "$RECOVERY_MAX_FP_INCREASE")
  elif [[ "$MODEL_KIND" == delayed_100hz ]]; then
    TRAINING_COMMAND+=(--prediction-mode delayed_100hz --no-recovery-head)
  fi
  if (( OVERWRITE )); then
    TRAINING_COMMAND+=(--overwrite)
  fi
}

run_one() {
  local modelname="${1}_${MODEL_KIND}"
  local family_dir="models/${modelname}_cv"
  local output_dir="${family_dir}/nested_lopo"
  local log="${family_dir}/nested_lopo.log"
  build_training_command "$@"
  local -a command=("${TRAINING_COMMAND[@]}")

  printf '\nRunning LOPO model: %s\n' "$modelname"
  printf 'Command:'
  printf ' %q' "${command[@]}"
  printf '\n'
  printf 'Publish target: %s/%s/%s/{best.pth,train_config.json}\n' \
    "$FINISHED_MODELS_ROOT" "$1" "$MODEL_KIND"

  if (( DRY_RUN )); then
    return
  fi

  mkdir -p "$family_dir"
  local model_lock
  exec {model_lock}>"$family_dir/.nested_lopo.lock"
  flock -n "$model_lock" || die "training or publication already running for $modelname"
  command+=(--run-lock-fd "$model_lock")
  {
    printf 'Command:'
    printf ' %q' "${command[@]}"
    printf '\n\n'
    "${command[@]}"
  } 2>&1 | tee "$log"

  publish_best_model "$modelname" "$output_dir"
  exec {model_lock}>&-
  printf 'LOPO summary: %s\n' "$output_dir/summary.json"
  printf 'Log: %s\n' "$log"
}

publish_skipped_model() {
  local modelname="${1}_${MODEL_KIND}"
  local family_dir="models/${modelname}_cv"
  local model_lock
  exec {model_lock}>"$family_dir/.nested_lopo.lock"
  flock -n "$model_lock" || die "training or publication already running for $modelname"
  publish_best_model "$modelname" "$family_dir/nested_lopo" if-needed
  exec {model_lock}>&-
}

prepare_runs() {
  local recipe requested matched config epoch_limit output_dir decision action reason
  # Validate every selection before starting any training or writing summaries.
  for requested in "${ONLY_MODELS[@]}"; do
    matched=0
    for recipe in "${RECIPES[@]}"; do
      if [[ "$requested" == "$recipe" || "$requested" == "${recipe}_${MODEL_KIND}" ]]; then
        matched=1
      fi
    done
    (( matched )) || die "unknown --only model: $requested"
  done
  for recipe in "${RECIPES[@]}"; do
    if ! should_run "$recipe" && ! should_run "${recipe}_${MODEL_KIND}"; then
      continue
    fi
    config="$RECIPE_DIR/$recipe.json"
    [[ -f "$config" ]] || die "missing training recipe: $config"
    epoch_limit="$("$LOPO_PYTHON" - "$config" "$MAX_EPOCHS" <<'PY'
import json
from pathlib import Path
import sys

config = json.loads(Path(sys.argv[1]).read_text())
value = sys.argv[2] or config.get("configured_max_epochs", config.get("num_epochs", 35))
if isinstance(value, bool) or int(value) != float(value) or int(value) < 1:
    raise SystemExit("Maximum training epochs must be a positive integer")
print(int(value))
PY
)"
    output_dir="models/${recipe}_${MODEL_KIND}_cv/nested_lopo"
    build_training_command "$recipe" "$config" "$epoch_limit"
    decision="$(PYTHONDONTWRITEBYTECODE=1 "${TRAINING_COMMAND[@]}" --check-existing)"
    read -r action reason <<< "$("$LOPO_PYTHON" -c '
import json, sys
decision = json.loads(sys.argv[1])
print(decision["action"], decision["reason"])
' "$decision")"
    if [[ "$action" == reuse ]]; then
      printf 'Skipping %s_%s: %s; use --overwrite to retrain.\n' "$recipe" "$MODEL_KIND" "$reason"
      SKIPPED_MODELS+=("$recipe")
      continue
    fi
    if [[ "$action" == replace || "$reason" == 'forced by --overwrite' ]]; then
      printf 'Will replace %s: %s\n' "$output_dir" "$reason"
    fi
    SELECTED_MODELS+=("$recipe")
    SOURCE_CONFIGS+=("$config")
    EPOCH_LIMITS+=("$epoch_limit")
  done
  RUN_COUNT=${#SELECTED_MODELS[@]}
  (( RUN_COUNT + ${#SKIPPED_MODELS[@]} > 0 )) || die "no model configurations selected"
}

stop_jobs() {
  local pid
  for pid in "${ACTIVE_PIDS[@]}"; do
    # Job control gives each worker its own process group, including tee/Python.
    kill -TERM -- "-$pid" 2>/dev/null || true
  done
  for pid in "${ACTIVE_PIDS[@]}"; do
    wait "$pid" 2>/dev/null || true
  done
}

run_selected_models() {
  local index=0 pid status next_start=0 delay_seconds=0
  local -a running=()
  if (( DRY_RUN )); then
    for (( index=0; index<RUN_COUNT; index++ )); do
      if (( MAX_PARALLEL > 1 )); then
        printf 'Earliest start: +%d minute(s), parallel limit: %d\n' \
          "$((index * START_DELAY_MINUTES))" "$MAX_PARALLEL"
      fi
      run_one "${SELECTED_MODELS[index]}" "${SOURCE_CONFIGS[index]}" "${EPOCH_LIMITS[index]}"
    done
    return
  fi
  if (( MAX_PARALLEL > 1 )); then
    delay_seconds=$((START_DELAY_MINUTES * 60))
  fi
  set -m
  trap stop_jobs EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM
  while (( index < RUN_COUNT || ${#ACTIVE_PIDS[@]} > 0 )); do
    running=()
    for pid in "${ACTIVE_PIDS[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then
        running+=("$pid")
      elif wait "$pid"; then
        # Only the parent writes this shared summary, after a successful job.
        append_metrics_overview "${JOB_MODELS[$pid]}"
        unset 'JOB_MODELS[$pid]'
      else
        status=$?
        printf 'Training failed for %s (exit %d)\n' "${JOB_MODELS[$pid]}" "$status" >&2
        return "$status"
      fi
    done
    ACTIVE_PIDS=("${running[@]}")
    if (( index < RUN_COUNT && ${#ACTIVE_PIDS[@]} < MAX_PARALLEL && SECONDS >= next_start )); then
      run_one "${SELECTED_MODELS[index]}" "${SOURCE_CONFIGS[index]}" "${EPOCH_LIMITS[index]}" &
      pid=$!
      ACTIVE_PIDS+=("$pid")
      JOB_MODELS[$pid]="${SELECTED_MODELS[index]}_${MODEL_KIND}"
      printf 'Started %s (PID %d)\n' "${JOB_MODELS[$pid]}" "$pid"
      next_start=$((SECONDS + delay_seconds))
      index=$((index + 1))
      continue
    fi
    if (( index < RUN_COUNT || ${#ACTIVE_PIDS[@]} > 0 )); then
      sleep 1
    fi
  done
  trap - EXIT INT TERM
  set +m
}

main() {
if [[ "${1:-}" == delayed-100hz ]]; then
  shift
  MODEL_KIND="delayed_100hz"
  RECOVERY_HEAD=0
  METRICS_SUMMARY="models/lopo_delayed_100hz_metrics_overview.txt"
fi
if [[ "${1:-}" == "recovery-only" || "${1:-}" == "recovery-lopo" ]]; then
  shift
  exec bash train_finished_models_recovery.sh --lopo "$@"
fi
if [[ "${1:-}" == "reselect-thresholds" ]]; then
  shift
  exec "$LOPO_PYTHON" run/post_reselect_thresholds.py "$@"
fi

while (( $# > 0 )); do
  case "$1" in
    --patient-root)
      (( $# >= 2 )) || die "--patient-root requires a directory"
      PATIENT_ROOT="$2"
      shift 2
      ;;
    --recipe-dir)
      (( $# >= 2 )) || die "--recipe-dir requires a directory"
      RECIPE_DIR="$2"
      shift 2
      ;;
    --only)
      (( $# >= 2 )) || die "--only requires a model name"
      ONLY_MODELS+=("$2")
      shift 2
      ;;
    --recovery-head)
      [[ "$MODEL_KIND" != delayed_100hz ]] || die "delayed-100hz uses one primary output; --recovery-head is incompatible"
      RECOVERY_HEAD=1
      shift
      ;;
    --recovery-max-fp-increase)
      (( $# >= 2 )) || die "--recovery-max-fp-increase requires a fraction"
      RECOVERY_MAX_FP_INCREASE="$2"
      RECOVERY_ALLOWANCE_SET=1
      shift 2
      ;;
    --parallel)
      (( $# >= 2 )) || die "--parallel requires a job count"
      MAX_PARALLEL="$2"
      shift 2
      ;;
    --start-delay-minutes)
      (( $# >= 2 )) || die "--start-delay-minutes requires a duration"
      START_DELAY_MINUTES="$2"
      shift 2
      ;;
    --num-epochs)
      (( $# >= 2 )) || die "--num-epochs requires a maximum epoch count"
      MAX_EPOCHS="$2"
      shift 2
      ;;
    --seed)
      (( $# >= 2 )) || die "--seed requires an integer"
      BASE_SEED="$2"
      shift 2
      ;;
    --overwrite)
      OVERWRITE=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      die "unknown argument $1"
      ;;
  esac
done

[[ "$MAX_PARALLEL" =~ ^[1-9][0-9]*$ ]] || die "--parallel must be a positive integer"
[[ "$START_DELAY_MINUTES" =~ ^(0|[1-9][0-9]*)$ ]] || die "--start-delay-minutes must be a nonnegative integer"
[[ "$BASE_SEED" =~ ^(0|[1-9][0-9]*)$ ]] || die "--seed must be a nonnegative integer"
[[ -z "$MAX_EPOCHS" || "$MAX_EPOCHS" =~ ^[1-9][0-9]*$ ]] || die "--num-epochs must be a positive integer"

if (( RECOVERY_HEAD )); then
  "$LOPO_PYTHON" - "$RECOVERY_MAX_FP_INCREASE" <<'PY'
import math
import sys

try:
    value = float(sys.argv[1])
except ValueError:
    raise SystemExit("--recovery-max-fp-increase must be a finite nonnegative fraction")
if not math.isfinite(value) or value < 0:
    raise SystemExit("--recovery-max-fp-increase must be a finite nonnegative fraction")
PY
elif (( RECOVERY_ALLOWANCE_SET )); then
  die "--recovery-max-fp-increase requires --recovery-head"
fi

[[ -d "$PATIENT_ROOT" ]] || die "patient root is not a directory: $PATIENT_ROOT"

prepare_runs
if (( ! DRY_RUN )); then
  for skipped_model in "${SKIPPED_MODELS[@]}"; do
    publish_skipped_model "$skipped_model"
  done
fi
if (( RUN_COUNT == 0 )); then
  printf 'All selected models already match; no training needed.\n'
  return
fi
if (( ! DRY_RUN )); then
  init_metrics_summary
  for skipped_model in "${SKIPPED_MODELS[@]}"; do
    append_metrics_overview "${skipped_model}_${MODEL_KIND}"
  done
fi
run_selected_models
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
