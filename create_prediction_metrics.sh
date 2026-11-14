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
CV_METHOD=""
MODELS_ROOT=""
METRICS_ROOT="metrics"
OVERVIEW=""
ONLY_MISSING=0
DISABLE_RECOVERY=0
EXCLUDE_RECOVERY_MODELS=0
SYNTHETIC_GRADIENT=0
EXCLUDE_LOW_FREQUENCIES=0
SAVE_SYNTHETIC=0
SAVE_PREPPED=0
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
Two-output models automatically use their configured recovery threshold. Logs
and fixed-threshold dashboards include the recovery output; no extra flag is needed.
Use --no-recovery to evaluate only the current output of the same trained model.
Those reports go under <output-dir>/current_only/ for comparison with recovery.

By default every artifact is written under the project's metrics/ folder, with one
flat subdirectory named exactly after each model:

  metrics/<modelname>/...

Full prediction/event parquets are temporary unless --save-prepped is given.
Saved parquets go under metrics/prepped/ with the same subdirectory layout.
Small score/annotation caches in metrics/.cache/ allow metrics and plots to be
rebuilt without repeating inference. Existing saved parquets are reused.

The root prediction_metrics_overview.csv contains one event-weighted pooled
row per model plus the siemens_tr baseline. The matching .txt contains the
pooled comparison and compact totals. Per-input comparisons are written to
prediction_metrics_per_input.{txt,csv} beside the overview, grouped by frequency
for synthetic runs. Text comparison tables omit the type column.
Each model directory also receives
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
  --cv-method METHOD        Select nested_lopo, nested_lopo_hpo, or
                             nested_recovery_lopo training artifacts.
                             Default: flat pairs and unambiguous nested runs.
                             --model accepts the declared name or CV family name.
  --no-recovery             Disable recovery events; keep the current prediction
                             and saved calibration. Reports: current_only/.
  --exclude-recovery-models Skip recovery model versions entirely, including
                             frozen and LOPO recovery. Keep normal 50 Hz and
                             delayed 100 Hz models.
  --only-missing             Skip model/input pairs whose cached predictions,
                             metrics, plots, and event analysis
                             already exist and match the saved model/runtime settings.
                             Training settings and calibration audit metadata
                             do not invalidate predictions. Inference/native
                             changes do; plotting changes rebuild plots only.
                             Recovery models require a matching cache key;
                             older single-output caches remain usable.
  --save-prepped             Keep full prediction/event parquets under
                             <output-dir>/prepped/, preserving subdirectories.
                             Matching saved files are reused with --only-missing.
  --include-outer-position-errors
                             Include outer position errors in overview metrics
                             and columns. Default: count each as FP + FN.
                             Also works with cached predictions (--only-missing).
  --synthetic-gradient      Evaluate main peaks at 190, 90, 80, 40, 20, 10 Hz,
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
  # All published normal and recovery models on the six preferred recordings:
  ./metrics.sh normal

  # Only normal 50 Hz and 100 Hz models:
  ./metrics.sh normal --exclude-recovery-models

  # Same recordings with synthetic gradients:
  ./metrics.sh synthetic

  # Also keep full prediction/event parquets:
  ./metrics.sh synthetic --save-prepped

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
    --cv-method)
      [[ $# -ge 2 ]] || die "--cv-method requires a method"
      CV_METHOD="$2"
      shift 2
      ;;
    --only-missing)
      ONLY_MISSING=1
      shift
      ;;
    --no-recovery)
      DISABLE_RECOVERY=1
      shift
      ;;
    --exclude-recovery-models)
      EXCLUDE_RECOVERY_MODELS=1
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
    --save-prepped)
      SAVE_PREPPED=1
      shift
      ;;
    --no-save-prepped)
      SAVE_PREPPED=0
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
FORWARDED_PREDICT_ARGS=()
for arg in "${PREDICT_ARGS[@]}"; do
  case "$arg" in
    --no-recovery) DISABLE_RECOVERY=1 ;;
    --synthetic-gradient) SYNTHETIC_GRADIENT=1 ;;
    --save-synthetic) SAVE_SYNTHETIC=1 ;;
    --save-prepped) SAVE_PREPPED=1; continue ;;
    --no-save-prepped) SAVE_PREPPED=0; continue ;;
    --synthetic-main-frequency-hz|--synthetic-main-frequency-hz=*)
      die "the synthetic sweep sets --synthetic-main-frequency-hz for each configuration" ;;
  esac
  FORWARDED_PREDICT_ARGS+=("$arg")
done
PREDICT_ARGS=("${FORWARDED_PREDICT_ARGS[@]}")
ARTIFACT_ROOT="$METRICS_ROOT"
PREPPED_ROOT="$ARTIFACT_ROOT/prepped"
CACHE_ROOT="$ARTIFACT_ROOT/.cache"
if [[ "$DISABLE_RECOVERY" -eq 1 ]]; then
  METRICS_ROOT="$METRICS_ROOT/current_only"
  PREDICT_ARGS+=(--no-recovery)
fi
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

prediction_cache_info() {
  "$PYTHON_BIN" - "$config" "$checkpoint" "$input" "$PREDICT_SCRIPT" \
    "$MODEL_VARIANT" "${PREDICT_ARGS[@]}" "${CONFIG_PREDICT_ARGS[@]}" <<'PY'
import importlib.util
from pathlib import Path
import sys

spec = importlib.util.spec_from_file_location(
    "prediction_fingerprint", Path("src/dkfz/live/prediction_fingerprint.py")
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
mode, key = module.prediction_cache_info(*sys.argv[1:5], sys.argv[5:])
print(mode, key, module.plot_source_key(Path("run/create_prediction_plots.py")))
PY
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

src = Path.cwd() / "src"
if src.is_dir() and str(src) not in sys.path:
    sys.path.insert(0, str(src))

from dkfz.live.eval_metrics import build_siemens_trigger_metrics_report
from dkfz.live.prediction_cache import read_prediction_frame


prediction_log = Path(sys.argv[1])
target_col = "target"
error_association_frames = 1

df = read_prediction_frame(prediction_log)
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

discovery_args=(list --root "$MODELS_ROOT" --variant "$MODEL_VARIANT")
[[ -z "$CV_METHOD" ]] || discovery_args+=(--cv-method "$CV_METHOD")
[[ "$EXCLUDE_RECOVERY_MODELS" -eq 0 ]] || discovery_args+=(--exclude-recovery-models)
for model in "${MODELS[@]}"; do
  discovery_args+=(--model "$model")
done
model_listing="$(PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" -m dkfz.live.artifact_discovery "${discovery_args[@]}")" || die "could not resolve model artifacts"
MODELS=()
declare -A MODEL_DIRS=()
while IFS=$'\t' read -r model model_dir; do
  [[ -n "$model" ]] || continue
  MODELS+=("$model")
  MODEL_DIRS["$model"]="$model_dir"
done <<< "$model_listing"
[[ ${#MODELS[@]} -gt 0 ]] || \
  die "no $MODEL_VARIANT model artifacts found under $MODELS_ROOT"

mkdir -p "$METRICS_ROOT" "$(dirname "$OVERVIEW")"
: > "$OVERVIEW"
RECORDS_FILE="$(mktemp "$METRICS_ROOT/.prediction_metrics_records.XXXXXX")"
WORK_DIR="$(mktemp -d "$METRICS_ROOT/.prediction_work.XXXXXX")"
cleanup_records() {
  rm -f -- "$RECORDS_FILE"
  rm -rf -- "$WORK_DIR"
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
      model_dir="${MODEL_DIRS[$model]}"
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
      relative_dir="${artifact_dir#"$ARTIFACT_ROOT"/}"
      prepped_dir="$PREPPED_ROOT/$relative_dir"
      cache_dir="$CACHE_ROOT/$relative_dir"
      saved_events="$prepped_dir/prediction_trigger_events_${output_base}.parquet"
      saved_log="$prepped_dir/prediction_all_events_${output_base}.parquet"
      compact_log="$cache_dir/prediction_all_events_${output_base}.arrow"
      saved_stamp="$prepped_dir/prediction_all_events_${output_base}_cache.key"
      # Adopt old outputs without making the model recompute them.
      for legacy in "$artifact_dir/prediction_trigger_events_${output_base}.parquet" \
                    "$artifact_dir/prediction_all_events_${output_base}.parquet"; do
        if [[ -f "$legacy" && ! -e "$prepped_dir/$(basename "$legacy")" ]]; then
          mkdir -p "$prepped_dir"
          mv -- "$legacy" "$prepped_dir/"
        fi
      done
      if [[ "$SAVE_PREPPED" -eq 1 ]]; then
        mkdir -p "$prepped_dir"
        out_events="$saved_events"
        out_log="$saved_log"
      else
        temporary_dir="$WORK_DIR/$relative_dir"
        mkdir -p "$temporary_dir"
        out_events="$temporary_dir/prediction_trigger_events_${output_base}.parquet"
        out_log="$temporary_dir/prediction_all_events_${output_base}.parquet"
      fi
      out_metrics="$artifact_dir/prediction_all_events_${output_base}_metrics.txt"
      metrics_json="${out_metrics%.txt}.json"
      run_log="$artifact_dir/prediction_all_events_${output_base}_run.log"
      dashboard="$artifact_dir/prediction_all_events_${output_base}_threshold_dashboard.pdf"
      errors="$artifact_dir/prediction_all_events_${output_base}_threshold_errors.pdf"
      analysis_events="$artifact_dir/prediction_all_events_${output_base}_threshold_events.csv"
      cache_stamp="$artifact_dir/prediction_all_events_${output_base}_cache.key"
      plot_stamp="$artifact_dir/prediction_all_events_${output_base}_plots_cache.key"

      if ! cache_info="$(prediction_cache_info)"; then
        echo "Could not read model/cache settings for $model on $input_name" >&2
        exit_status=1
        continue
      fi
      read -r output_mode cache_key plot_source_key <<< "$cache_info"
      plot_key="$(printf '%s\0' "$cache_key" "$plot_source_key" "$THRESHOLD_STEP" "$MAX_ERROR_PLOTS" "$ERROR_WINDOW_FRAMES" | sha256sum)"
      cache_matches=0
      if [[ -f "$cache_stamp" ]]; then
        [[ "$(<"$cache_stamp")" == "$cache_key" ]] && cache_matches=1
      elif [[ "$output_mode" == "single" ]]; then
        # Retain compatibility with pre-feature single-output cached reports.
        cache_matches=1
      fi

      prediction_ok=0
      prediction_reused=0
      analysis_log="$compact_log"
      saved_matches=0
      if [[ -f "$saved_stamp" ]]; then
        [[ "$(<"$saved_stamp")" == "$cache_key" ]] && saved_matches=1
      elif [[ "$cache_matches" -eq 1 && ! -f "$compact_log" && -f "$saved_log" ]]; then
        # Adopt a legacy parquet whose cache policy was validated above.
        printf '%s\n' "$cache_key" > "$saved_stamp"
        saved_matches=1
      fi
      if [[ ! -f "$compact_log" && "$saved_matches" -eq 1 && -f "$saved_log" ]]; then
        analysis_log="$saved_log"
      fi
      storage_complete=1
      if [[ "$SAVE_PREPPED" -eq 1 && ( "$saved_matches" -eq 0 || ! -f "$saved_events" || ! -f "$saved_log" ) ]]; then
        storage_complete=0
      fi
      if [[ "$ONLY_MISSING" -eq 1 && "$SAVE_SYNTHETIC" -eq 0 && "$cache_matches" -eq 1 && "$storage_complete" -eq 1 && -f "$analysis_log" && -f "$out_metrics" && -f "$metrics_json" ]]; then
        echo "Using existing prediction for $model on $input_name" >&2
        prediction_ok=1
        prediction_reused=1
      else
        echo "Running $model on $input_name ($output_mode)" >&2
        rm -f -- "$cache_stamp" "$plot_stamp"
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
          analysis_log="$out_log"
        else
          echo "Prediction failed for $model on $input_name; see $run_log" >&2
          exit_status=1
        fi
      fi

      if [[ "$prediction_ok" -eq 1 ]]; then
        if [[ "$prediction_reused" -eq 0 && "$SAVE_PREPPED" -eq 1 ]]; then
          printf '%s\n' "$cache_key" > "$saved_stamp"
        fi
        if [[ "$prediction_reused" -eq 0 || ! -f "$compact_log" ]]; then
          if ! PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" - "$analysis_log" "$compact_log" "$metrics_json" <<'PY'
import json
from pathlib import Path
import sys
from dkfz.live.prediction_cache import cache_prediction_frame
report = json.loads(Path(sys.argv[3]).read_text())
cache_prediction_frame(Path(sys.argv[1]), Path(sys.argv[2]), report.get("target_col", "target"))
PY
          then
            echo "Could not cache predictions for $model on $input_name" >&2
            exit_status=1
            continue
          fi
        fi
        analysis_log="$compact_log"
        if [[ "$SAVE_PREPPED" -eq 0 ]]; then
          rm -f -- "$out_log" "$out_events"
        fi
        printf '%s\n' "$cache_key" > "$cache_stamp"
        SUCCESSFUL_PREDICTIONS["$artifact_dir/$output_base"]="$analysis_log"
        plots_match=0
        if [[ -f "$plot_stamp" ]]; then
          [[ "$(<"$plot_stamp")" == "$plot_key" ]] && plots_match=1
        elif [[ "$output_mode" == "single" ]]; then
          plots_match=1
        fi
        if [[ "$ONLY_MISSING" -eq 1 && "$prediction_reused" -eq 1 && "$plots_match" -eq 1 && -f "$dashboard" && -f "$errors" && -f "$analysis_events" ]]; then
          echo "Using existing plots for $model on $input_name" >&2
          printf '%s\n' "$plot_key" > "$plot_stamp"
        else
          echo "Creating fixed-threshold plots for $model on $input_name" >&2
          if "$PYTHON_BIN" "$PLOT_SCRIPT" \
            --prediction-log "$analysis_log" \
            --config "$config" \
            --metrics-json "$metrics_json" \
            --dashboard "$dashboard" \
            --errors "$errors" \
            --events "$analysis_events" \
            --threshold-step "$THRESHOLD_STEP" \
            --max-error-plots "$MAX_ERROR_PLOTS" \
            --error-window-frames "$ERROR_WINDOW_FRAMES" \
            >> "$run_log" 2>&1; then
            printf '%s\n' "$plot_key" > "$plot_stamp"
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
              if read -r siemens_timepoints siemens_gt_r_peaks siemens_true_positive siemens_position_error siemens_outer_position_error siemens_false_positive siemens_false_negative < <(extract_siemens_metrics "$analysis_log"); then
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
      output_base="compare${OUTPUT_SUFFIX}_${input_stem}"
      prediction_id="$artifact_dir/$output_base"
      if [[ -n "${SUCCESSFUL_PREDICTIONS[$prediction_id]+x}" ]]; then
        pooled_args+=(--prediction-log "${SUCCESSFUL_PREDICTIONS[$prediction_id]}" --metrics-json "$artifact_dir/prediction_all_events_${output_base}_metrics.json")
        pooled_count=$((pooled_count + 1))
      fi
    done
    [[ "$pooled_count" -gt 0 ]] || continue
    pooled_base="$artifact_dir/prediction_all_events_compare${OUTPUT_SUFFIX}_all_inputs"
    echo "Creating pooled dashboard for $model from $pooled_count/${#RESOLVED_INPUTS[@]} inputs" >&2
    if ! "$PYTHON_BIN" "$PLOT_SCRIPT" \
        --pooled \
        "${pooled_args[@]}" \
        --config "${MODEL_DIRS[$model]}/$CONFIG_REL" \
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
