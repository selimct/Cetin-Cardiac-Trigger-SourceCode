#!/usr/bin/env bash
# Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Developed at the German Cancer Research Center (DKFZ).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

METRICS_SUMMARY="models/lopo_metrics_overview.txt"
FINISHED_MODELS_ROOT="finished_models"
PATIENT_ROOT="${PATIENT_ROOT:-data/live}"
DRY_RUN=0
OVERWRITE=0
RUN_COUNT=0
ONLY_MODELS=()

usage() {
  cat <<'USAGE'
Usage: ./script.sh [options]
       ./script.sh reselect-thresholds [reselection options]

Run the fixed-architecture nested LOPO feature experiments listed at the end
of this script. This uses run/train_lopo.py, not the HPO entrypoint. After each
run, the deployment pair is published under finished_models/MODEL/.

Options:
  --patient-root DIR  Patient directory root. Default: data/live, or the
                      PATIENT_ROOT environment variable.
  --only MODEL        Run only this model. May be repeated.
  --overwrite         Replace each selected model's existing nested_lopo output.
  --dry-run           Print commands without training or writing summaries.
  --help              Show this help.

Examples:
  ./script.sh --dry-run
  ./script.sh --only target_r_ecg3
  ./script.sh --only target_r_slope --only target_r_slope_ecg3
  ./script.sh --patient-root data/live --overwrite --only target_r_ecg3
USAGE
}

die() {
  printf 'script.sh: %s\n' "$*" >&2
  exit 2
}

init_metrics_summary() {
  mkdir -p models
  printf "%-40s %7s %10s %10s %10s %10s %10s %12s %12s %12s\n" \
    "model" "epoch" "threshold" "Se" "+P" "DER" "macro DER" \
    "PE/target" "FP/pred" "FN/target" > "$METRICS_SUMMARY"
  printf "%-40s %7s %10s %10s %10s %10s %10s %12s %12s %12s\n" \
    "-----" "-----" "---------" "---" "---" "---" "---------" \
    "---------" "-------" "---------" >> "$METRICS_SUMMARY"
}

append_metrics_overview() {
  local modelname="$1"
  local summary="models/${modelname}_cv/nested_lopo/summary.json"

  .venv/bin/python -c '
import json
import sys

model, path = sys.argv[1:]
with open(path, encoding="utf-8") as stream:
    summary = json.load(stream)

event = summary["pooled_event_position"]
best_model = summary["best_model"]
epoch = int(best_model["selected_epoch"])
threshold = float(best_model["threshold"])
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
  local finished_dir="${FINISHED_MODELS_ROOT}/${modelname}"

  .venv/bin/python - "$modelname" "$output_dir" "$finished_dir" <<'PY'
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

from dkfz.live.onnx_export import build_model_from_config


modelname, output_value, finished_value = sys.argv[1:]
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

run_one() {
  local modelname="$1"
  shift

  if ! should_run "$modelname"; then
    return
  fi
  RUN_COUNT=$((RUN_COUNT + 1))

  local family_dir="models/${modelname}_cv"
  local output_dir="${family_dir}/nested_lopo"
  local log="${family_dir}/nested_lopo.log"
  local -a command=(
    .venv/bin/python run/train_lopo.py
    --patient-root "$PATIENT_ROOT"
    --modelname "$modelname"
    --save-prepped
  )

  if (( OVERWRITE )); then
    command+=(--overwrite)
  elif (( ! DRY_RUN )) && [[ -d "$output_dir" ]] &&
      [[ -n "$(find "$output_dir" -mindepth 1 -print -quit)" ]]; then
    die "$output_dir is not empty; use --overwrite to replace it"
  fi
  command+=("$@")

  printf '\nRunning LOPO model: %s\n' "$modelname"
  printf 'Command:'
  printf ' %q' "${command[@]}"
  printf '\n'
  printf 'Publish target: %s/%s/{best.pth,train_config.json}\n' \
    "$FINISHED_MODELS_ROOT" "$modelname"

  if (( DRY_RUN )); then
    return
  fi

  mkdir -p "$family_dir"
  {
    printf 'Command:'
    printf ' %q' "${command[@]}"
    printf '\n\n'
    "${command[@]}"
  } 2>&1 | tee "$log"

  publish_best_model "$modelname" "$output_dir"
  append_metrics_overview "$modelname"
  printf 'LOPO summary: %s\n' "$output_dir/summary.json"
  printf 'Log: %s\n' "$log"
}

main() {
if [[ "${1:-}" == "reselect-thresholds" ]]; then
  shift
  exec .venv/bin/python run/post_reselect_thresholds.py "$@"
fi

while (( $# > 0 )); do
  case "$1" in
    --patient-root)
      (( $# >= 2 )) || die "--patient-root requires a directory"
      PATIENT_ROOT="$2"
      shift 2
      ;;
    --only)
      (( $# >= 2 )) || die "--only requires a model name"
      ONLY_MODELS+=("$2")
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

[[ -d "$PATIENT_ROOT" ]] || die "patient root is not a directory: $PATIENT_ROOT"

if (( ! DRY_RUN )); then
  init_metrics_summary
fi

# All runs use the fixed architecture in TRAINING_DEFAULTS. The inner folds
# still select the epoch and threshold; no hyperparameter search is performed.

# run_one "target_r"

run_one "target_r_cqm" --set signal_quality_mode=ecg_puls

run_one "target_r_cqm4ecg" --set signal_quality_mode=ecg

run_one "target_r_ecg3" --set ecg3_col=ecg3_raw

run_one "target_r_ecg3_cqm" \
  --set ecg3_col=ecg3_raw \
  --set signal_quality_mode=ecg_puls

run_one "target_r_ecg3_cqm4ecg" \
  --set ecg3_col=ecg3_raw \
  --set signal_quality_mode=ecg

run_one "target_r_slope" --set ecg_slope_mode=fast_slow

run_one "target_r_slope_cqm" \
  --set ecg_slope_mode=fast_slow \
  --set signal_quality_mode=ecg_puls

run_one "target_r_slope_cqm4ecg" \
  --set ecg_slope_mode=fast_slow \
  --set signal_quality_mode=ecg

run_one "target_r_slope_ecg3" \
  --set ecg_slope_mode=fast_slow \
  --set ecg3_col=ecg3_raw

run_one "target_r_slope_ecg3_cqm" \
  --set ecg_slope_mode=fast_slow \
  --set ecg3_col=ecg3_raw \
  --set signal_quality_mode=ecg_puls

run_one "target_r_slope_ecg3_cqm4ecg" \
  --set ecg_slope_mode=fast_slow \
  --set ecg3_col=ecg3_raw \
  --set signal_quality_mode=ecg

run_one "target_r_slope_fast" --set ecg_slope_mode=fast

run_one "target_r_slope_fast_cqm" \
  --set ecg_slope_mode=fast \
  --set signal_quality_mode=ecg_puls

run_one "target_r_slope_fast_cqm4ecg" \
  --set ecg_slope_mode=fast \
  --set signal_quality_mode=ecg

run_one "target_r_slope_fast_ecg3" \
  --set ecg_slope_mode=fast \
  --set ecg3_col=ecg3_raw

run_one "target_r_slope_fast_ecg3_cqm" \
  --set ecg_slope_mode=fast \
  --set ecg3_col=ecg3_raw \
  --set signal_quality_mode=ecg_puls

run_one "target_r_slope_fast_ecg3_cqm4ecg" \
  --set ecg_slope_mode=fast \
  --set ecg3_col=ecg3_raw \
  --set signal_quality_mode=ecg

(( RUN_COUNT > 0 )) || die "none of the --only model names matched a configured run"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
