# Previous-frame recovery

The optional recovery head gives uncertain beats a second chance after one
additional frame arrives. One LSTM pass through the window ending at frame `t`
produces two logits, in this order:

- `current`: target at `t`, with the existing prediction and trigger timing.
- `previous`: target at `t - 1`, using frame `t` as additional context.

At 50 Hz the recovery output has 20 ms of lookahead relative to its target.
The recovered event is located at `t - 1` and emitted at `t`; emission times
are never backdated. This is one model with two small output heads, rather
than two LSTM evaluations.

After updating the source, rebuild the native runtime and Python extension:

```bash
bash run/rebuild_native.sh --python-extension
```

Run this after updating an existing installation, before using the native trigger API.

## Train a two-output model

The current training approach implemented by `train_models_lopo.sh` is:

1. Load each model's architecture and preprocessing settings from its recipe. Train
   the shared LSTM and both output heads from new random weights.
2. Use the wrapper's shared learning rate (currently `0.001`) and base seed
   (default `42`). Architecture parameters are fixed; no HPO or repeated-seed
   search is performed.
3. Within each outer training split, use inner patient-wise LOPO validation to
   select the epoch and primary detection threshold. Keep the outer test patient
   excluded from training and selection.
4. Hold the selected epoch and primary threshold fixed, then calibrate recovery
   from the corresponding inner-validation predictions, using the configured
   relative FP allowance (default `10%`).
5. Evaluate on each outer test patient. Select the deployment epoch and primary
   threshold from pooled inner-validation searches, calibrate recovery at that
   epoch, and train the final model on all supplied patients. Publish it under
   `finished_models/MODEL/recovery/`.

This is the default workflow of the wrapper. The `recovery-only` subcommand
instead adds a recovery head to frozen source weights.

The root training wrapper trains fresh two-output models for `ecg3`,
`slope_ecg3`, `slope_slow_ecg3`, `slope_ecg3_cqm4ecg` and
`slope_slow_ecg3_cqm4ecg`. It reads architecture and preprocessing settings from
each model's `config/lopo_recipes/MODEL.json`. These recipes preserve the settings
from the original finished models and do not depend on saved checkpoints or
output directories. Use `--recipe-dir DIR` to select another recipe directory.
The explicit `--lr 0.001`
in `build_training_command` in `train_models_lopo.sh` overrides each recipe's
learning rate; edit that value to adjust LR for all these jobs.
The LSTM and both heads start with
new weights. Outputs use separate `_recovery` names:

```bash
./train_models_lopo.sh --dry-run
```

Remove `--dry-run` to train all five models sequentially. Use `--only ecg3`
(repeatable) to select a subset. `--recovery-head` remains accepted but is
already enabled by default. Source checkpoints and preprocessing caches are
not reused. Each recipe's `configured_max_epochs` restores the original search
budget (35 for these models), rather than its previously selected duration.
`--num-epochs` overrides that maximum; the base seed defaults to 42.

The wrapper can start with both `models/` and `finished_models/` absent or empty.
Keep `config/lopo_recipes/` and the raw annotated recordings. It recreates training
outputs under `models/MODEL_recovery_cv/nested_lopo/` and publishes the final
weights and matching config under `finished_models/MODEL/recovery/`. Only these
five two-output models are trained; clearing the directories also removes any
other models previously stored there. The wrapper does not clear directories
itself.

To allow up to four overlapping jobs with at least 20 minutes between starts:

```bash
./train_models_lopo.sh --parallel 4 --start-delay-minutes 20 --dry-run
./train_models_lopo.sh --parallel 4 --start-delay-minutes 20
```

The earliest starts are 0, 20, 40, 60 and 80 minutes after the first launch,
subject to the parallel job limit.
`--parallel 2` limits overlap to two jobs. Jobs share GPU memory; staggered
starts do not reduce their eventual combined memory requirement. Each job has
its own log and output directory. The parent writes the shared metrics summary.
A failed job or interruption stops the remaining workers and their children.

Existing runs under `models/MODEL_recovery_cv/nested_lopo/` are automatically
replaced when effective training settings, recording contents, training code
or dependency versions differ. Incomplete runs are replaced too. Completed
equivalent runs are reported and skipped before scheduling; use `--overwrite`
to force retraining. The check compares the maximum epoch budget rather than
the previously selected epoch, and ignores operational settings such as cache
paths, preprocessing worker counts and logging options.

Successful new runs save `training_request.json`, including input/code hashes
and completion-artifact checksums. Older runs without this signature cannot be
verified as equivalent and are rebuilt once. `--dry-run` reports these decisions
without deleting outputs or starting training. An active fit holds a per-model
lock, so it cannot be overwritten by a second invocation.
The wrapper retains that lock through publication, and cancellation stops the
worker process group in both sequential and parallel modes. Reused completed
fits also restore a missing or stale published copy without retraining;
matching published weights preserve runtime threshold edits. Basic training
settings are validated before any previous outputs are removed.

Enable the architecture with `--recovery-head`:

```bash
.venv/bin/python run/train_lopo.py \
  --modelname slope_ecg3_recovery \
  --config config/lopo_recipes/slope_ecg3.json \
  --patient-root data/live \
  --recovery-head \
  --recovery-max-fp-increase 0.10
```

The existing patient-wise training, epoch selection, and primary-threshold
selection remain in place. Recovery is calibrated afterward from the selected
inner-validation predictions. Global recovery calibration uses predictions at
the selected global epoch, without consulting the outer test recordings.
The HPO entrypoint supports the same options. `run/train.py` also supports the
new head for the standard training workflow.

The two training targets are `[target[t], target[t - 1]]`. Dataset windows and
recovery state never cross recording boundaries. Recovery is skipped for the
first model output after a reset, when no previous output frame is available.

## Add recovery to existing weights

To add a frozen recovery head to every model in `finished_models/`, use the
root script:

```bash
./train_finished_models_recovery.sh --dry-run
./train_finished_models_recovery.sh
```

It runs ten recovery-only epochs per source model, calibrates with a default
10% relative validation FP allowance, and publishes separate
`finished_models/MODEL/frozen_recovery/` copies. Completed copies are skipped; a failed
job stops the script and is retried on the next invocation. Histories, reports
and `recovery_training.log` remain in `models/MODEL_frozen_recovery/`.
Training writes `models/MODEL_frozen_recovery/best_tuned.pth`; only after successful
completion does the wrapper copy it and the config to `finished_models/`.
The training checkpoint and config remain under `models/`.
Use `--only ecg3 --only slope_ecg3` for a subset, `--num-epochs` to change the
training length, or `--recovery-max-fp-increase` to change the allowance.
The saved data/validation split is reused; keep the six benchmark recordings
for the independent comparison afterward.

CPU recovery-only training caches the frozen LSTM representations and current
logits once per training call, including validation. Later epochs train only the
recovery head. This is enabled by default for the standard deterministic float32
CPU model and datasets, for both the saved-split and nested LOPO workflows. The
cache lives in RAM for that call; source checkpoints and fold data are never
reused across calls. Short final batches use the original forward path to retain
their original numerical kernels; single-epoch jobs also use the original path.
The original weights, two-output loss, optimizer, shuffled batch order,
validation/checkpoint selection, and saved checkpoint format stay
the same. Small floating-point differences from batching remain possible.

GPU training, custom/stochastic models or datasets, and representation caches
over 512 MiB use the original path. To explicitly compare with the original CPU
implementation, add `--no-cache-frozen-recovery` to
`train_finished_models_recovery.sh`, `run/train_finetune.py`, or
`run/train_recovery_lopo.py`. No native rebuild is needed.

The output names distinguish the training modes:

| Mode | Published model directory |
| --- | --- |
| Joint two-output LOPO (`train_models_lopo.sh --recovery-head`) | `finished_models/MODEL/recovery/` |
| Frozen base with saved data split | `finished_models/MODEL/frozen_recovery/` |
| Frozen base with nested LOPO (`--lopo`) | `finished_models/MODEL/lopo_recovery/` |

Declared model names stay `MODEL_recovery`, `MODEL_frozen_recovery` and
`MODEL_lopo_recovery`; training archives under `models/` keep those names.
Discovery also accepts older flat published directories. An existing joint
recovery model does not count as a completed frozen recovery run.

For an initial experiment that preserves the original detector, add a head
and train only that head:

```bash
.venv/bin/python run/train_finetune.py \
  --modelname slope_ecg3_frozen_recovery \
  --config finished_models/slope_ecg3/train_config.json \
  --checkpoint finished_models/slope_ecg3/best.pth \
  --recovery-head --recovery-only \
  --num-epochs 5 \
  --recovery-max-fp-increase 0.10
```

The new model name writes separate artifacts under
`models/slope_ecg3_frozen_recovery/`. Adding the head initializes it from the existing
current head. `--recovery-only` freezes the original LSTM and current head and
retains the source primary threshold, FP threshold, and DER threshold. The
original current scores stay unchanged during this training mode. Omit
`--recovery-only` to fine-tune the shared representation and both heads jointly.

Use independent annotated validation recordings via `--threshold-inputs` when
available. The source checkpoint may already have seen the original training
recordings, including any segments reused for validation. Such a fine-tuning
run is not automatically a patient-held-out evaluation.

## Nested LOPO for a frozen recovery head

Use patient-wise nested recovery selection instead of the within-recording split:

```bash
./train_finished_models_recovery.sh --lopo --dry-run
./train_finished_models_recovery.sh --lopo
```

The main training wrapper also accepts
`./train_models_lopo.sh recovery-only` with the same recovery LOPO options.

Use `--only ecg3 --only slope_ecg3` for a subset. In this mode `--num-epochs`
sets the maximum epoch; the selected epoch comes from inner patient folds.
It trains only recovery heads, using the same source checkpoint, primary
thresholds and fitted preprocessing in every fold. Each outer patient is
excluded from recovery-head training and from its epoch/threshold selection.
Inner patients select the epoch and recovery threshold by event misses, within
the FP allowance, rather than by the unchanged primary predictions. After
outer evaluation, all inner out-of-fold predictions select the deployment
epoch/threshold; the final head is fitted on all configured training patients.
Outer test metrics are never used for selection.

Default grouping uses exactly the original config's training files and their
patient-directory names. It does not include the separate six-file benchmark.
The Python entrypoint can also accept `--patient-root` or `--patient-manifest`.

Outputs are deliberately separate from the simple fine-tuning copies:

- `finished_models/MODEL/lopo_recovery/{best.pth,train_config.json}`
- `models/MODEL_lopo_recovery_cv/nested_recovery_lopo/`: inner/outer checkpoints,
  fold membership, per-epoch threshold audits, outer metrics, and final model.
- `models/MODEL_lopo_recovery_cv/recovery_training.log`

Completed published copies are skipped. An incomplete job is restarted from
scratch on rerun, replacing only its dedicated CV output directory. Before
publishing, the entrypoint verifies that every primary tensor in the final and
outer checkpoints is unchanged from the original source.
All fold checkpoints and the final fitted `best.pth` stay under `models/`.
After the complete run succeeds, the root wrapper publishes a copy of only
the final checkpoint and config to `finished_models/`. A failed training run
does not create a finished-model copy.

These are **recovery-head folds conditional on a pretrained base**. The frozen
LSTM, current head and preprocessing may already have seen the outer patients
during original training. This preserves the original detector but does not
establish a completely patient-unseen full-detector estimate. Configs and
reports record this scope and `base_patient_holdout_verified=false`. A fully
patient-held-out detector requires training the base separately within each
fold, as in normal model training.

```bash
./create_train_metrics.sh models --cv-method nested_recovery_lopo

.venv/bin/python run/train_recovery_lopo.py \
  --config finished_models/ecg3/train_config.json \
  --checkpoint finished_models/ecg3/best.pth \
  --modelname ecg3_lopo_recovery --num-epochs 10
```

## Choose current-only prediction or recovery

A two-output checkpoint supports both choices without retraining. Normally the
current output and calibrated recovery output both participate. To evaluate only
the current output, add `--no-recovery` to the same metrics command:

```bash
./create_prediction_metrics.sh finished_models --model ecg3_recovery \
  --no-recovery data/true/p02/p02_head.parquet
```

Use the full six-recording selection below for the comparison. Current-only
reports are saved under `metrics/current_only/`; ordinary recovery reports remain
under `metrics/`. Omit `--no-recovery` to use the saved recovery policy again.
The Torch and TensorRT offline entrypoints also accept `--no-recovery`.

For native deployment, use the same choice when preparing the model on the Jetson:

```bash
./prepare_model.sh ecg3_recovery --model-dir finished_models/ecg3/recovery \
  --jetson-only --no-recovery
```

The two-output architecture and engine stay intact; only recovery events are
disabled in the deployment manifest. The saved config/calibrated threshold is
preserved. Running preparation without the flag restores the configured policy.
For a persistent config choice, set `recovery_enabled` to `false`, then to `true`
to re-enable it. Keep `recovery_head=true`, which describes the checkpoint shape.
A null recovery threshold still means calibration has not enabled recovery.
Stop the service and any direct inference before preparing, validate afterward,
and restart. In the GUI, choose **Startup recovery** beside the finished model
and use **Deploy and start** for this sequence. Press Check or Enter for updated
information. This works through the existing SSH workflow without a recovery
add-on or live switching; see the [operation guide](README.md#choose-recovery-behavior).
Preparation itself retains its existing Python export/parity requirements.

Both modes compute the same current-frame scores, with the same primary trigger
timing and no added wait. Recovery can affect later events through shared holdoff
when enabled; current-only mode uses the primary detector alone. Joint training
changes the shared LSTM, so current-only mode is not necessarily identical to an
independently trained normal model. Use the frozen `--recovery-only` fine-tuning
option above if the original current scores must stay exactly unchanged.

For independent two-output models, the root wrapper always trains from scratch
and saves `MODEL_recovery` under `finished_models/MODEL/recovery/`. To train a separate
single-output model, use `run/train_lopo.py` with its own model name and config.

## Trigger behavior and calibration

The primary trigger always runs first. Recovery is allowed only when:

- Both the previous and current primary scores are below the primary threshold.
- There is no pending primary candidate or unfinished early-emission tail.
- The previous frame is consecutive and belongs to the same recording.
- The shared holdoff and duplicate checks allow a new event.
- The recovery score exceeds its independently calibrated threshold.

A recovery emits immediately when accepted; it does not start another wait for
the recovery score to fall. It cannot retract an event already emitted. A false
recovery can suppress a following event through the shared holdoff, so calibration
replays the complete combined state machine rather than counting threshold
crossings independently.

The default extra false-positive budget is the larger of 10% of the fixed primary
detector's FP count and 0.5 additional FP per evaluated validation minute:

```text
total FP limit = baseline FP + max(floor(0.10 × baseline FP), floor(0.5 × validation minutes))
```

This is an average over pooled validation, not a strict limit for each two-minute
window. Duration includes real evaluated frames only, excluding artificial gaps
between recordings. Counts include outer position errors as FP/FN, matching the
default prediction overview. Recovery must reduce both FN and combined FP+FN
within the budget. Native early-emission replay must not increase FN or combined
FP+FN and must meet its own FP budget. Otherwise
`recovery_threshold` is saved as `null` and recovery stays disabled.

Configure the rate with `recovery_min_extra_fp_per_minute` or the training flag
`--recovery-min-extra-fp-per-minute`; set it to zero to disable the duration
allowance. `recovery_max_fp_increase` remains configurable. For recalibration,
`run/calibrate_recovery.py` accepts `--min-extra-fp-per-minute`; the mixed
`update_model_thresholds.sh` workflow accepts `--recovery-min-extra-fp-per-minute`.
The config and calibration-search CSV record validation duration and the actual
budgets. These settings do not retrain weights or change the primary policy.

This is a pooled validation constraint, not a guaranteed per-recording or future
deployment FP bound. The primary threshold search and the recovery search are
saved separately. Recovery metadata includes the combined validation metrics.
When the model has a higher FP threshold, calibration also replays the native
service's early-emission mode. A recovery candidate must stay within that mode's
FP budget and must not increase its misses. The search CSV and validation
metadata include this additional audit. Changing primary thresholds or other
runtime policy settings afterward requires a fresh evaluation.

To recalibrate only recovery from saved validation scores:

```bash
./update_model_thresholds.sh recovery \
  --config models/slope_ecg3_frozen_recovery/tuning/tuned_config.json \
  --validation-log models/slope_ecg3_frozen_recovery/tuning/tuned_all_events.parquet \
  --max-fp-increase 0.10 --apply
```

Without `--apply`, this writes the search CSV and reports the choice without
changing the config. Repeat `--validation-log` to pool independent recordings.
The log option is optional when saved validation artifacts are available.
New two-output CV runs save `recovery_validation_predictions.parquet` at the
selected deployment epoch and reference it as `recovery_validation_log` in the
config. These contain inner-validation predictions, never outer test predictions.

The default updater handles mixed single-output and two-output collections:

```bash
./update_model_thresholds.sh finished_models --apply
```

For two-output models it recalibrates only recovery; primary, manual, FP and DER
thresholds remain fixed. Use `--recovery-max-fp-increase` to override the allowance
in this mixed-collection command. For older two-output CV runs without saved
same-epoch predictions, `--rebuild-cv-validation` replays the saved inner
checkpoints at the selected epoch without training. The original training
inputs, preprocessing cache and inner checkpoints must still be available.
For single-output models, that flag retains its existing inner-fold retraining
behavior when validation reconstruction is necessary.

Use validation predictions, not the preferred test benchmark, to choose thresholds.
Missing validation artifacts produce an error; no outer-test or benchmark
fallback is used. The legacy `reselect-thresholds` subcommand remains restricted
to single-output models.

## Export and evaluate

The root preparation script detects the two outputs from the saved config:

```bash
./prepare_model.sh slope_ecg3_recovery \
  --model-dir finished_models/slope_ecg3/recovery --workstation-only
./create_train_metrics.sh models --model slope_ecg3_recovery
```

Training reports include all CV methods by default, including
`nested_recovery_lopo`; `--cv-method` restricts them. Prediction metrics discover
both published flat model pairs and final pairs inside nested CV folders.
Preparation searches `models/`, then `finished_models/`, and then unique nested
CV matches, so `--model-dir` is optional. Use the config's declared model name
for preparation. Both prediction and preparation accept
`--cv-method nested_recovery_lopo` (or the corresponding normal/HPO method) to
resolve ambiguous nested runs; preparation also accepts an exact `--model-dir`.

Training overviews include the recovery threshold (`disabled` when calibration
keeps it off), and CSV/JSON reports retain the recovery settings. The
`create_prediction_metrics.sh` wrapper automatically evaluates the configured
combined detector. Its `--only-missing` cache checks include checkpoint contents,
runtime configuration, input identity, and prediction arguments. Training settings
and calibration audit metadata do not invalidate predictions; changing the selected
recovery threshold does. Inference code and the installed native extension are
included in prediction keys. Plot settings and plotting dependencies have a separate
key, so plot changes reuse predictions while rebuilding plots.
Legacy single-output caches can still be reused, but recovery
models require a matching key and rerun old predictions on first use.

ONNX exports one tensor named `logit`, shaped `[batch, 2]`. The order is current,
then previous. Old models retain their scalar-per-window output and checkpoint
format. Runtime configuration declares `recovery_head=true` and the optional
`recovery_threshold`. A two-output model with a null recovery threshold still
produces both scores, but only its current head emits events.

For the fine-tuned example:

```bash
.venv/bin/python run/jetson_export_onnx.py \
  --modelname slope_ecg3_frozen_recovery --model-variant tuned

./create_prediction_metrics.sh models \
  --model slope_ecg3_frozen_recovery --model-variant tuned \
  data/true/p02/p02_head.parquet \
  data/true/p14/p14_heart.parquet data/true/p14/p14_head.parquet \
  data/true/p16/p16_head.parquet data/true/p16/p16_heart.parquet \
  data/true/p16/p16_pelvis.parquet
```

The Python Torch/TensorRT predictors and native deployment manifest/pipeline
support both outputs. Build a matching TensorRT engine from the new ONNX model;
the old scalar engine cannot supply recovery scores. Native runtime output
counts are checked against the manifest.

Saved frame logs keep `prob` as the current score and add `recovery_prob`, whose
target is the previous frame, plus `recovery_emitted_now`. Event logs retain
separate peak and actual emission positions/times. Prediction dashboards replay
the combined detector; error plots show the recovery curve shifted to its
target frame.

Enabling the architecture alone is not evidence of reduced misses. Train and
calibrate it, then compare FN/FP and emission timing on the six preferred
recordings and their configured gradient variants. Existing finished models
are not modified or activated by adding this implementation.
