#!/usr/bin/env bash
# Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Developed at the German Cancer Research Center (DKFZ).

set -uo pipefail

PYTHON_BIN=".venv/bin/python"
PREDICT_SCRIPT="run/offline_predict.py"
PLOT_SCRIPT="run/create_prediction_plots.py"
OVERVIEW_SCRIPT="run/create_prediction_metrics_overview.py"
MODEL_VARIANT="best"
MODELS_ROOT=""
METRICS_ROOT="metrics"
OVERVIEW=""
ONLY_MISSING=0
SYNTHETIC_GRADIENT=0
EXCLUDE_LOW_FREQUENCIES=0
SAVE_SYNTHETIC=0
THRESHOLD_STEP="0.001"
MAX_ERROR_PLOTS="12"
ERROR_WINDOW_FRAMES="25"

POSITIONALS=()
INPUTS=()
MODELS=()
PREDICT_ARGS=()
OVERVIEW_ARGS=()

usage() {
  cat <<'USAGE'
Usage:
  ./create_prediction_metrics.sh [options] <models-dir> <test-file-or-glob> [...]

Discovers model directories under <models-dir>, simulates live inference for
each model/input pair, writes fixed-threshold dashboard and error-example PDFs,
then creates per-model pooled dashboards, all-input overviews, and a comparison of every
model against the siemens_tr baseline. The configured threshold is evaluated
but is never reselected or changed.

By default every artifact is written under the project's metrics/ folder, with one
flat subdirectory named exactly after each model:

  metrics/<modelname>/...

The root prediction_metrics_overview.csv contains one event-weighted pooled
row per model plus the siemens_tr baseline. Each model directory also receives
prediction_metrics_all_inputs.{txt,csv}, with every input and a pooled row.
Overviews include exact Se/+P/DER and Beat Se/+P/DER. By default, outer matches
count as one FP and one FN, with no oPE columns. Use --include-outer-position-errors
to show oPEs and count them as true positives for Beat metrics alongside PEs.
This option controls overview tables; prediction reports and dashboards retain
their own outer association settings.
The final text footer totals annotated/predicted R peaks and evaluated minutes
across all selected inputs for each system.
prediction_all_events_compare_all_inputs_threshold_dashboard.pdf pools the
selected inputs for each model (compare_best_tuned_all_inputs for tuned models).
The pooled dashboard also gets three separately rendered PDFs with suffixes
_event_errors.pdf, _rates.pdf, and _tradeoff.pdf appended to its filename stem.
All four pooled PDFs are rebuilt even with --only-missing to reflect this run's inputs.

Options:
  --output-dir DIR           Artifact root. Default: metrics
  --output FILE              Overview table path. Default:
                             <output-dir>/prediction_metrics_overview.txt
                             A companion .csv is always written.
  --model MODEL              Only run one model. Repeatable.
                             Default: every model directory with artifacts.
  --model-variant VARIANT    Model variant passed to offline_predict.py.
                             Default: best
  --only-missing             Skip model/input pairs whose event parquet,
                             prediction log, metrics, plots, and event analysis
                             already exist.
  --include-outer-position-errors
                             Include outer position errors in overview metrics
                             and columns. Default: count each as FP + FN.
                             Also works with cached predictions (--only-missing).
  --synthetic-gradient      Evaluate main peaks at 190, 114, 84, 62, 50, 20, 10 Hz,
                             aliasing above Nyquist.
                             Add noise independently (seed 42), then calculate
                             ecg3_syn = ecg2_syn - ecg1_syn. Predictions use
                             ecgN_syn; raw ECG columns remain unchanged.
                             Artifacts go under
                             <output-dir>/synthetic/main_<frequency>hz/<model>/.
                             Summary tables pool all inputs per frequency and model,
                             under <output-dir>/synthetic/ and its model folders.
  --exclude-low-frequencies Omit the 20 Hz and 10 Hz configurations from the sweep.
                             Requires --synthetic-gradient.
  --save-synthetic          Also save each modified input to
                             data/synthetic/main_<frequency>hz/pNN/<stem>_synthetic.parquet.
                             Includes raw ECGs and ecg1_syn/ecg2_syn/ecg3_syn.
                             Requires --synthetic-gradient. Predictions are
                             rerun even with --only-missing to save the data.
  --threshold-step VALUE     Dashboard threshold-grid step. Default: 0.001
  --max-error-plots N        Maximum FP/PE/FN example pages. Default: 12
  --error-window-frames N    Frames around each error example. Default: 25
  --python PATH              Python executable. Default: .venv/bin/python
  --help                     Show this help.
  --                         Forward remaining arguments to offline_predict.py
                             for rare manual overrides.

Examples:
  ./create_prediction_metrics.sh finished_models --only-missing --synthetic-gradient --save-synthetic "data/true/p02/p02_head.parquet" "data/true/p14/p14_heart.parquet" "data/true/p14/p14_head.parquet" "data/true/p16/p16_head.parquet" "data/true/p16/p16_heart.parquet" "data/true/p16/p16_pelvis.parquet"

  ./create_prediction_metrics.sh finished_models --only-missing "data/true/p02/p02_head.parquet" "data/true/p14/p14_heart.parquet" "data/true/p14/p14_head.parquet" "data/true/p16/p16_head.parquet" "data/true/p16/p16_heart.parquet" "data/true/p16/p16_pelvis.parquet"

  ./create_prediction_metrics.sh finished_models 'data/test/p04/*.parquet'

  ./create_prediction_metrics.sh --synthetic-gradient --save-synthetic \
     finished_models 'data/true/p*/*.parquet'

  ./create_prediction_metrics.sh \
     --model target_r_slope_cqm4ecg \
     finished_models \
      data/test/p04/p04_head.parquet
USAGE
}

die() {
  echo "Error: $*" >&2
  exit 2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output)
      [[ $# -ge 2 ]] || die "--output requires a path"
      OVERVIEW="$2"
      shift 2
      ;;
    --output-dir)
      [[ $# -ge 2 ]] || die "--output-dir requires a directory"
      METRICS_ROOT="$2"
      shift 2
      ;;
    --model)
      [[ $# -ge 2 ]] || die "--model requires a model name"
      MODELS+=("$(basename "$2")")
      shift 2
      ;;
    --model-variant)
      [[ $# -ge 2 ]] || die "--model-variant requires a value"
      MODEL_VARIANT="$2"
      shift 2
      ;;
    --only-missing)
      ONLY_MISSING=1
      shift
      ;;
    --include-outer-position-errors)
      OVERVIEW_ARGS+=(--include-outer-position-errors)
      shift
      ;;
    --synthetic-gradient)
      SYNTHETIC_GRADIENT=1
      shift
      ;;
    --exclude-low-frequencies|--exclude-tr-200)
      EXCLUDE_LOW_FREQUENCIES=1
      shift
      ;;
    --save-synthetic)
      SAVE_SYNTHETIC=1
      shift
      ;;
    --threshold-step)
      [[ $# -ge 2 ]] || die "--threshold-step requires a value"
      THRESHOLD_STEP="$2"
      shift 2
      ;;
    --max-error-plots)
      [[ $# -ge 2 ]] || die "--max-error-plots requires an integer"
      MAX_ERROR_PLOTS="$2"
      shift 2
      ;;
    --error-window-frames)
      [[ $# -ge 2 ]] || die "--error-window-frames requires an integer"
      ERROR_WINDOW_FRAMES="$2"
      shift 2
      ;;
    --python)
      [[ $# -ge 2 ]] || die "--python requires an executable path"
      PYTHON_BIN="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      PREDICT_ARGS+=("$@")
      break
      ;;
    -*)
      die "unknown option: $1"
      ;;
    *)
      POSITIONALS+=("$1")
      shift
      ;;
  esac
done

# Honor these switches after -- as well, including artifact/cache separation.
for arg in "${PREDICT_ARGS[@]}"; do
  case "$arg" in
    --synthetic-gradient) SYNTHETIC_GRADIENT=1 ;;
    --save-synthetic) SAVE_SYNTHETIC=1 ;;
    --synthetic-main-frequency-hz|--synthetic-main-frequency-hz=*)
      die "the synthetic sweep sets --synthetic-main-frequency-hz for each configuration" ;;
  esac
done
if [[ "$SAVE_SYNTHETIC" -eq 1 && "$SYNTHETIC_GRADIENT" -eq 0 ]]; then
  die "--save-synthetic requires --synthetic-gradient"
fi
if [[ "$EXCLUDE_LOW_FREQUENCIES" -eq 1 && "$SYNTHETIC_GRADIENT" -eq 0 ]]; then
  die "--exclude-low-frequencies requires --synthetic-gradient"
fi
if [[ "$SYNTHETIC_GRADIENT" -eq 1 ]]; then
  METRICS_ROOT="$METRICS_ROOT/synthetic"
  PREDICT_ARGS+=(--synthetic-gradient)
fi
if [[ "$SAVE_SYNTHETIC" -eq 1 ]]; then
  PREDICT_ARGS+=(--save-synthetic)
fi

[[ ${#POSITIONALS[@]} -ge 2 ]] || {
  usage >&2
  exit 2
}
MODELS_ROOT="${POSITIONALS[0]}"
INPUTS=("${POSITIONALS[@]:1}")
[[ -d "$MODELS_ROOT" ]] || die "models directory not found: $MODELS_ROOT"
if [[ -z "$OVERVIEW" ]]; then
  OVERVIEW="$METRICS_ROOT/prediction_metrics_overview.txt"
fi
[[ -x "$PYTHON_BIN" ]] || die "Python executable not found or not executable: $PYTHON_BIN"
[[ -f "$PREDICT_SCRIPT" ]] || die "Prediction script not found: $PREDICT_SCRIPT"
[[ -f "$PLOT_SCRIPT" ]] || die "Plot script not found: $PLOT_SCRIPT"
[[ -f "$OVERVIEW_SCRIPT" ]] || die "Overview script not found: $OVERVIEW_SCRIPT"

GRADIENT_FREQUENCIES=("")
if [[ "$SYNTHETIC_GRADIENT" -eq 1 ]]; then
  frequency_list="$("$PYTHON_BIN" -c '
from pathlib import Path
import sys
sys.path.insert(0, str(Path.cwd() / "src"))
from dkfz.core.gradient_noise import GRADIENT_MAIN_FREQUENCIES_HZ
print(" ".join(str(f) for f in GRADIENT_MAIN_FREQUENCIES_HZ if not (sys.argv[1] == "1" and f in (20, 10))))
' "$EXCLUDE_LOW_FREQUENCIES")" || die "could not load gradient configurations"
  read -r -a GRADIENT_FREQUENCIES <<< "$frequency_list"
fi

case "$MODEL_VARIANT" in
  best)
    CHECKPOINT_REL="best.pth"
    CONFIG_REL="train_config.json"
    OUTPUT_SUFFIX=""
    ;;
  tuned)
    CHECKPOINT_REL="best_tuned.pth"
    CONFIG_REL="tuning/tuned_config.json"
    OUTPUT_SUFFIX="_best_tuned"
    ;;
  *)
    die "--model-variant must be one of: best, tuned"
    ;;
esac

add_input() {
  local path="$1"
  local abs_path
  abs_path="$(realpath "$path")" || return 1
  if [[ -z "${SEEN_INPUTS[$abs_path]+x}" ]]; then
    RESOLVED_INPUTS+=("$path")
    SEEN_INPUTS["$abs_path"]=1
  fi
}

resolve_inputs() {
  local item path found
  for item in "${INPUTS[@]}"; do
    found=0
    if [[ -d "$item" ]]; then
      while IFS= read -r -d '' path; do
        add_input "$path"
        found=1
      done < <(find "$item" -maxdepth 1 -type f -name '*.parquet' -print0 | sort -z)
    elif [[ -f "$item" ]]; then
      add_input "$item"
      found=1
    else
      while IFS= read -r path; do
        [[ -f "$path" ]] || continue
        add_input "$path"
        found=1
      done < <(compgen -G "$item" | sort)
    fi

    if [[ "$found" -eq 0 ]]; then
      echo "Warning: no parquet files matched: $item" >&2
    fi
  done
}

safe_stem() {
  local stem safe
  stem="$(basename "$1")"
  stem="${stem%.*}"
  safe="$(printf '%s' "$stem" | sed 's/[^A-Za-z0-9_.-]/_/g; s/^[._]*//; s/[._]*$//')"
  if [[ -z "$safe" ]]; then
    printf 'input'
  else
    printf '%s' "$safe"
  fi
}

extract_metrics() {
  local metrics_file="$1"
  local metrics_json="${metrics_file%.txt}.json"
  "$PYTHON_BIN" -c '
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
event = report["event_position"]
print(
    int(report["num_rows"]),
    int(event["num_target_events"]),
    int(event["exact_true_positive"]),
    int(event["position_error"]),
    int(event.get("outer_position_error", 0)),
    int(event["unmatched_false_positive"]),
    int(event["unmatched_false_negative"]),
)
' "$metrics_json"
}

extract_siemens_metrics() {
  local prediction_log="$1"
  "$PYTHON_BIN" -c '
from pathlib import Path
import sys

import pandas as pd

src = Path.cwd() / "src"
if src.is_dir() and str(src) not in sys.path:
    sys.path.insert(0, str(src))

from dkfz.live.eval_metrics import build_siemens_trigger_metrics_report


prediction_log = Path(sys.argv[1])
target_col = "target"
error_association_frames = 1

df = pd.read_parquet(prediction_log)
report = build_siemens_trigger_metrics_report(
    df,
    target_col=target_col,
    error_association_frames=error_association_frames,
)
event = report["event_position"]
false_positive = int(event["unmatched_false_positive"])
false_negative = int(event["unmatched_false_negative"])
true_positive = int(event["exact_true_positive"])
position_error = int(event["position_error"])
outer_position_error = int(event.get("outer_position_error", 0))
print(
    int(report["num_rows"]),
    int(event["num_target_events"]),
    true_positive,
    position_error,
    outer_position_error,
    false_positive,
    false_negative,
)
' "$prediction_log"
}

append_overview_record() {
  local input="$1"
  local system="$2"
  local system_type="$3"
  local timepoints="$4"
  local target_events="$5"
  local true_positive="$6"
  local position_error="$7"
  local outer_position_error="$8"
  local false_positive="$9"
  local false_negative="${10}"
  local config="${11}"

  "$PYTHON_BIN" -c '
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path.cwd() / "src"))
from dkfz.live.metrics_overview import duration_minutes

with Path(sys.argv[12]).open(encoding="utf-8") as stream:
    config = json.load(stream)

record = {
    "input": sys.argv[2],
    "system": sys.argv[3],
    "system_type": sys.argv[4],
    "timepoints": int(sys.argv[5]),
    "duration_minutes": duration_minutes(int(sys.argv[5]), config),
    "num_target_events": int(sys.argv[6]),
    "exact_true_positive": int(sys.argv[7]),
    "position_error": int(sys.argv[8]),
    "outer_position_error": int(sys.argv[9]),
    "unmatched_false_positive": int(sys.argv[10]),
    "unmatched_false_negative": int(sys.argv[11]),
}
if sys.argv[13]:
    record["main_frequency_hz"] = float(sys.argv[13])
with Path(sys.argv[1]).open("a", encoding="utf-8") as stream:
    stream.write(json.dumps(record, sort_keys=True) + "\n")
' "$RECORDS_FILE" "$input" "$system" "$system_type" \
    "$timepoints" "$target_events" "$true_positive" "$position_error" "$outer_position_error" \
    "$false_positive" "$false_negative" "$config" "$main_frequency"
}

RESOLVED_INPUTS=()
declare -A SEEN_INPUTS=()
resolve_inputs
[[ ${#RESOLVED_INPUTS[@]} -gt 0 ]] || die "no parquet inputs found"

if [[ ${#MODELS[@]} -eq 0 ]]; then
  shopt -s nullglob
  for checkpoint in "$MODELS_ROOT"/*/"$CHECKPOINT_REL"; do
    model_dir="$(dirname "$checkpoint")"
    [[ -f "$model_dir/$CONFIG_REL" ]] || continue
    MODELS+=("$(basename "$model_dir")")
  done
  shopt -u nullglob
fi
[[ ${#MODELS[@]} -gt 0 ]] || \
  die "no $MODEL_VARIANT model artifacts found under $MODELS_ROOT"

mkdir -p "$METRICS_ROOT" "$(dirname "$OVERVIEW")"
: > "$OVERVIEW"
RECORDS_FILE="$(mktemp "$METRICS_ROOT/.prediction_metrics_records.XXXXXX")"
cleanup_records() {
  rm -f -- "$RECORDS_FILE"
}
trap cleanup_records EXIT

exit_status=0
declare -A SUCCESSFUL_PREDICTIONS=()

for main_frequency in "${GRADIENT_FREQUENCIES[@]}"; do
  CONFIG_METRICS_ROOT="$METRICS_ROOT"
  CONFIG_PREDICT_ARGS=()
  if [[ -n "$main_frequency" ]]; then
    CONFIG_METRICS_ROOT="$METRICS_ROOT/main_${main_frequency}hz"
    CONFIG_PREDICT_ARGS=(--synthetic-main-frequency-hz "$main_frequency")
    echo "Evaluating synthetic main frequency $main_frequency Hz" >&2
  fi

  for input in "${RESOLVED_INPUTS[@]}"; do
    input_name="$(basename "$input")"
    input_stem="$(safe_stem "$input")"
    section_timepoints=""
    section_gt_r_peaks=""
    section_siemens_recorded=0
    section_model_records=0

    for model in "${MODELS[@]}"; do
      model_dir="$MODELS_ROOT/$model"
      checkpoint="$model_dir/$CHECKPOINT_REL"
      config="$model_dir/$CONFIG_REL"

      if [[ ! -f "$checkpoint" || ! -f "$config" ]]; then
        echo "Skipping $model on $input_name: missing $CHECKPOINT_REL or $CONFIG_REL" >&2
        exit_status=1
        continue
      fi

      artifact_dir="$CONFIG_METRICS_ROOT/$model"
      mkdir -p "$artifact_dir"

      output_base="compare${OUTPUT_SUFFIX}_${input_stem}"
      out_events="$artifact_dir/prediction_trigger_events_${output_base}.parquet"
      out_log="$artifact_dir/prediction_all_events_${output_base}.parquet"
      out_metrics="$artifact_dir/prediction_all_events_${output_base}_metrics.txt"
      metrics_json="${out_metrics%.txt}.json"
      run_log="$artifact_dir/prediction_all_events_${output_base}_run.log"
      dashboard="$artifact_dir/prediction_all_events_${output_base}_threshold_dashboard.pdf"
      errors="$artifact_dir/prediction_all_events_${output_base}_threshold_errors.pdf"
      analysis_events="$artifact_dir/prediction_all_events_${output_base}_threshold_events.csv"

      prediction_ok=0
      if [[ "$ONLY_MISSING" -eq 1 && "$SAVE_SYNTHETIC" -eq 0 && -f "$out_events" && -f "$out_log" && -f "$out_metrics" && -f "$metrics_json" ]]; then
        echo "Using existing prediction for $model on $input_name" >&2
        prediction_ok=1
      else
        echo "Running $model on $input_name" >&2
        if "$PYTHON_BIN" "$PREDICT_SCRIPT" \
          "$input" \
          --modelname "$model" \
          --model-variant "$MODEL_VARIANT" \
          --model-dir "$model_dir" \
          --out-events "$out_events" \
          --out-log "$out_log" \
          --out-metrics "$out_metrics" \
          "${PREDICT_ARGS[@]}" \
          "${CONFIG_PREDICT_ARGS[@]}" \
          > "$run_log" 2>&1; then
          prediction_ok=1
        else
          echo "Prediction failed for $model on $input_name; see $run_log" >&2
          exit_status=1
        fi
      fi

      if [[ "$prediction_ok" -eq 1 ]]; then
        SUCCESSFUL_PREDICTIONS["$out_log"]=1
        if [[ "$ONLY_MISSING" -eq 1 && -f "$dashboard" && -f "$errors" && -f "$analysis_events" ]]; then
          echo "Using existing plots for $model on $input_name" >&2
        else
          echo "Creating fixed-threshold plots for $model on $input_name" >&2
          if "$PYTHON_BIN" "$PLOT_SCRIPT" \
            --prediction-log "$out_log" \
            --config "$config" \
            --metrics-json "$metrics_json" \
            --dashboard "$dashboard" \
            --errors "$errors" \
            --events "$analysis_events" \
            --threshold-step "$THRESHOLD_STEP" \
            --max-error-plots "$MAX_ERROR_PLOTS" \
            --error-window-frames "$ERROR_WINDOW_FRAMES" \
            >> "$run_log" 2>&1; then
            :
          else
            echo "Plot creation failed for $model on $input_name; see $run_log" >&2
            exit_status=1
          fi
        fi

        if [[ -f "$out_metrics" ]]; then
          if read -r timepoints gt_r_peaks true_positive position_error outer_position_error false_positive false_negative < <(extract_metrics "$out_metrics"); then
            if [[ -z "$section_timepoints" ]]; then
              section_timepoints="$timepoints"
              section_gt_r_peaks="$gt_r_peaks"
            elif [[ "$section_timepoints" != "$timepoints" || "$section_gt_r_peaks" != "$gt_r_peaks" ]]; then
              echo "Warning: $model on $input_name produced timepoints=$timepoints gt_r_peaks=$gt_r_peaks; section uses timepoints=$section_timepoints gt_r_peaks=$section_gt_r_peaks" >&2
            fi

            if append_overview_record \
                "$input" "$model" "model" "$timepoints" "$gt_r_peaks" \
                "$true_positive" "$position_error" "$outer_position_error" "$false_positive" \
                "$false_negative" "$config"; then
              section_model_records=$((section_model_records + 1))
            else
              echo "Could not record metrics for $model on $input_name" >&2
              exit_status=1
            fi

            if [[ "$section_siemens_recorded" -eq 0 ]]; then
              if read -r siemens_timepoints siemens_gt_r_peaks siemens_true_positive siemens_position_error siemens_outer_position_error siemens_false_positive siemens_false_negative < <(extract_siemens_metrics "$out_log"); then
                if [[ "$section_timepoints" != "$siemens_timepoints" || "$section_gt_r_peaks" != "$siemens_gt_r_peaks" ]]; then
                  echo "Warning: siemens_tr on $input_name produced timepoints=$siemens_timepoints gt_r_peaks=$siemens_gt_r_peaks; section uses timepoints=$section_timepoints gt_r_peaks=$section_gt_r_peaks" >&2
                fi

                if append_overview_record \
                    "$input" "siemens_tr" "baseline" "$siemens_timepoints" \
                    "$siemens_gt_r_peaks" "$siemens_true_positive" \
                    "$siemens_position_error" "$siemens_outer_position_error" "$siemens_false_positive" \
                    "$siemens_false_negative" "$config"; then
                  section_siemens_recorded=1
                else
                  echo "Could not record siemens_tr metrics for $input_name" >&2
                  exit_status=1
                fi
              fi
            fi
          else
            echo "Could not parse metrics for $model on $input_name: $out_metrics" >&2
            exit_status=1
          fi
        else
          echo "No metrics written for $model on $input_name; see $run_log" >&2
          exit_status=1
        fi
      fi
    done

    if [[ "$section_model_records" -eq 0 ]]; then
      echo "No successful metrics for $input_name" >&2
    elif [[ "$section_siemens_recorded" -eq 0 ]]; then
      echo "Warning: no siemens_tr metrics could be computed for $input_name" >&2
    fi
  done

  for model in "${MODELS[@]}"; do
    artifact_dir="$CONFIG_METRICS_ROOT/$model"
    pooled_args=()
    pooled_count=0
    for input in "${RESOLVED_INPUTS[@]}"; do
      input_stem="$(safe_stem "$input")"
      out_log="$artifact_dir/prediction_all_events_compare${OUTPUT_SUFFIX}_${input_stem}.parquet"
      if [[ -n "${SUCCESSFUL_PREDICTIONS[$out_log]+x}" ]]; then
        pooled_args+=(--prediction-log "$out_log" --metrics-json "${out_log%.parquet}_metrics.json")
        pooled_count=$((pooled_count + 1))
      fi
    done
    [[ "$pooled_count" -gt 0 ]] || continue
    pooled_base="$artifact_dir/prediction_all_events_compare${OUTPUT_SUFFIX}_all_inputs"
    echo "Creating pooled dashboard for $model from $pooled_count/${#RESOLVED_INPUTS[@]} inputs" >&2
    if ! "$PYTHON_BIN" "$PLOT_SCRIPT" \
        --pooled \
        "${pooled_args[@]}" \
        --config "$MODELS_ROOT/$model/$CONFIG_REL" \
        --dashboard "${pooled_base}_threshold_dashboard.pdf" \
        --threshold-step "$THRESHOLD_STEP" \
        > "${pooled_base}_run.log" 2>&1; then
      echo "Pooled plot creation failed for $model; see ${pooled_base}_run.log" >&2
      exit_status=1
    fi
  done

done # gradient configurations (one unmodified pass for clean predictions)

if [[ ! -s "$RECORDS_FILE" ]]; then
  echo "No successful prediction metrics were available for an overview" >&2
  exit 1
fi

if ! "$PYTHON_BIN" "$OVERVIEW_SCRIPT" \
    --records "$RECORDS_FILE" \
    --metrics-root "$METRICS_ROOT" \
    --overview-text "$OVERVIEW" \
    "${OVERVIEW_ARGS[@]}" \
    --requested-input-files "${#RESOLVED_INPUTS[@]}"; then
  echo "Could not create pooled prediction metrics overviews" >&2
  exit_status=1
fi

exit "$exit_status"
