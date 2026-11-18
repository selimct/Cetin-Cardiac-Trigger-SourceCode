# 100 Hz predictor with a one-frame delayed target

The default `delayed_100hz` mode uses separately trained weights in training,
Torch/TensorRT offline prediction, preparation, and the native UART runtime.
Current-frame 50 Hz is an explicit opt-in; supported contract-3 models retain
their saved timing. Use `./train_models_lopo.sh normal` or
`--prediction-mode current` for fresh 50 Hz training.
Use the [deployment and qualification guide](jetson-model-deployment.md) for
target engine builds, accuracy checks, sustained timing, and activation.

## Training objective

Use one stream of nonoverlapping four-sample frames at 400 Hz:

```text
F0 = samples 0..3
F1 = samples 4..7
F2 = samples 8..11
...

At completion of Ft:
    input  = the most recent 100 frames, ending at Ft
    target = whether F(t-1) contains the annotated R-peak
```

There is one inference every 10 ms and one primary prediction stream. A
100-frame window preserves the current 50-frame/50 Hz model's one-second raw
input span. The target's future context replaces part of its historical
context when the total input span is held fixed.

Use the **latest** LSTM state, after consuming Ft, to predict F(t-1). Taking the
penultimate LSTM state would discard Ft's information and defeat the added
context. The existing recovery head demonstrates the previous-frame target
concept; this mode makes that delayed target the sole primary objective
and trains the backbone for it.

The raw annotation file remains unchanged. Only the training-window target
lookup changes to the penultimate frame. This requires separate training and
new weights: the input feature dimension changes, and the supervision changes.
For ECG3 plus fast/slow slopes and pulse, the current 29-feature/eight-sample
input becomes 17 features with four-sample frames. Existing weights cannot be
loaded unchanged into that input layer.

## Timing and preprocessing requirements

The penultimate target frame always has one complete subsequent frame available.
That is 10 ms after its end, or 10–17.5 ms after its R sample, depending on which
of its four samples contains the annotation. This is the earliest possible
algorithmic decision, before threshold-excursion waiting, processing, UART,
output scheduling and GPIO delays. It provides a guaranteed amount of future
context relative to the target; it does not guarantee correct classification.

Candidate positions must refer to F(t-1), while decision/emission timestamps
refer to Ft or the later frame that actually causes emission. The runtime must
not backdate emission by one frame. Holdoff and duplicate suppression must use
actual emission time on the raw-sample axis. A default 200 ms holdoff is 20 new
frames; the evaluated model's explicit 400 ms holdoff would be 40 new frames.
Prefer configuring this in milliseconds to avoid ambiguity across model rates.

Keep preprocessing horizons in physical time: ECG normalization still uses the
same number of raw samples; a 500-frame pulse normalization horizon at 50 Hz
becomes 1,000 frames at 100 Hz. Pulse quality history horizons need analogous
scaling. Pulse samples remain sample-and-hold; their physical acquisition rate
does not double. Slow slopes retain their trailing 40 raw samples (100 ms by
default). Fast slopes use the latest eight raw samples (20 ms) at both frame
rates, updated every frame. At 100 Hz, successive fast-slope windows overlap by
four samples. The first frame has zero fast slopes until eight samples are
available; resets and gaps restart that warmup. Both fits remain causal.
Signal-quality calibration must be fit on the training split for the new
framing, rather than copied blindly.

The delayed primary has an explicit target-offset contract in every path.
Candidate coordinates refer to the target; emission coordinates refer to the
actual observation. Resets and raw-index gaps/reversals discard normalization,
quality histories, incomplete frames, model windows, candidates and holdoff,
then warm up again. EOF does not invent a subsequent frame; flushing uses the
actual last raw sample, including a partial final frame. Sequence changes also
reset implicit sample indices. Existing current-frame reset behavior is retained.

## Controlled comparison

Use unchanged raw annotations and the same patient/split boundaries for:

1. The original 50 Hz, current-frame predictor.
2. A newly trained 100 Hz, current-frame predictor.
3. A newly trained 100 Hz, previous-frame predictor.

This distinguishes the effect of smaller frames from the effect of future
context. A separate future 50 Hz previous-frame experiment could test 20 ms of
future context; that variant is not part of this preset. Use multiple training seeds, select thresholds using validation
data only, then evaluate once on held-out recordings. Compare events against
raw annotation times, with the same timing tolerance, evaluation intervals and
physical holdoff. Report early/late timing errors, beat matches, false
positives/misses, and actual emission latency. Match by raw positions, not
frame indices that have different durations.

Higher accuracy is plausible, not established. Ten milliseconds may still be
insufficient to distinguish some waveform patterns, and false early predictions
can still initiate holdoff.

This is also not automatically a compute-efficiency improvement. Re-evaluating
100-step windows at 100 Hz entails four times the recurrent time steps per
second of 50-step windows at 50 Hz. Fewer input features reduce some arithmetic,
but recurrent state computation remains. Measure the actual cost on the Jetson;
a stateful recurrent design would be a separate training/runtime change.

## Train the model

Published variants use the same family layout as recovery models:

```text
finished_models/ecg3/
├── best.pth                     # Original model
├── train_config.json
├── recovery/                    # Existing recovery variant
└── delayed_100hz/
    ├── best.pth                 # Separately trained 100 Hz model
    └── train_config.json        # modelname: ecg3_delayed_100hz
```

The LOPO wrapper trains and publishes this layout. Preview without training:

```bash
LOPO_PYTHON=/home/orin/envs/jet/bin/python \
    ./train_models_lopo.sh delayed-100hz --only ecg3 --dry-run
```

Remove `--dry-run` to train that variant and publish after successful completion.
Omit `--only` to use all configured model recipes. Training artifacts stay in
`models/ecg3_delayed_100hz_cv/nested_lopo/`. Each recipe declares the 100 Hz contract and its physical window/history
durations. The 50 Hz opt-in converts these histories to the coarser frame grid. This command uses fresh
weights and disables recovery. Original and recovery artifacts stay in place.
The wrapper without a subcommand uses this delayed 100 Hz mode.
Use `normal` for 50 Hz or `recovery` for joint recovery training.

Metrics, preparation/export tools and the GUI discover `delayed_100hz/` as an
independent sibling. Use the unique name `ecg3_delayed_100hz` or the exact source
directory `finished_models/ecg3/delayed_100hz`; selecting `ecg3` selects the base.
After publishing, for example:

```bash
.venv/bin/python run/offline_predict.py data/evaluation/session.parquet \
    --modelname ecg3_delayed_100hz
```

The directory appears after successful training/publication. Existing 50 Hz
weights are never copied there as a substitute for a trained 100 Hz model.

For individual custom training runs instead of the publication wrapper:

Use a **new model name** and [the delayed recipe](../config/experiments/delayed_100hz.json):

```bash
.venv/bin/python run/train.py data/live/p01/p01_out.parquet \
    --threshold-inputs data/live/sp01/sp01_out.parquet \
    --config config/experiments/delayed_100hz.json \
    --modelname delayed_100hz_v1

.venv/bin/python run/train_lopo.py --patient-root data/live \
    --config config/experiments/delayed_100hz.json \
    --modelname delayed_100hz_v1 --no-use-prepped
```

Replace the example inputs with the intended patient-wise splits. The recipe
sets `prediction_mode=delayed_100hz`, `prediction_delay_frames=1`, `raw_hz=400`,
`frame_size_raw=4`, `window_size=100`, and one primary output. Recovery is
incompatible with the delayed primary. Train/prepare reject a model directory
with a different saved target contract; fine-tuning requires the same target
and architecture as its source checkpoint.

`train.py`, `train_finetune.py`, `train_lopo.py` and
`train_lopo_hpo.py` accept `--prediction-mode current|delayed_100hz` and
`--prediction-delay-frames 0|1`. The preset converts inherited eight-sample
settings by doubling the window and holdoff (`min_distance`),
pulse-normalization and pulse-quality frame counts. Saved `holdoff_frames` or
`holdoff_ms` overrides retain their physical duration. Explicit CLI frame counts
win; an already delayed config is not scaled twice. The default is **20 frames
= 200 ms**; an explicit new `--min-distance 10` means **100 ms** at 100 Hz.
Runtime `holdoff_frames` always uses the trained model's frame rate.

## TP, PE and oPE definitions (option 2)

Both 100 Hz controls use these one-to-one event tiers. Offsets are the absolute
difference between the selected probability-peak frame and the annotated target
frame; emission time does not determine the category.

| Category | Standard 50 Hz model | Four-sample 100 Hz model |
| --- | --- | --- |
| TP | Same frame | Same frame or ±1 frame |
| PE | ±1 frame | ±2–3 frames |
| oPE | ±2 frames | ±4–5 frames |
| Unmatched FP/FN | Beyond ±2 frames | Beyond ±5 frames |

This preserves approximately 20 ms-wide positional-error bands when using
10 ms frames. The 100 Hz outer frame offsets are 10/30/50 ms for TP/PE/oPE.
These are model-grid offsets, not exact raw-time acceptance limits: the normal
model's coarser bins make identical physical-time acceptance impossible under
these frame-based definitions. This intentionally replaces the earlier
100 Hz 0/2/4 radii, which made the positional tiers narrower.

Each target and prediction is used at most once. Identical-frame matches get
priority, then remaining tolerant TPs, PE and oPE. Error costs remain
`20 FP + 10 FN + PE + 5 oPE`. Legacy JSON/CSV keys such as `exact_true_positive`
and `exact_sensitivity` are retained for compatibility; at 100 Hz they count
TPs within the recorded `exact_tolerance_frames=1` radius.

Metrics schema **4** records the radii. Threshold/epoch selection, CV pooling,
offline reports and plots all use the same policy. Old 100 Hz configurations
are interpreted using the new defaults when re-evaluated, without editing their
weights or saved threshold. Historical schema-3 counts cannot be relabelled as
schema 4; recompute them from saved scores/annotations. Mixing tolerance policies
in pooled threshold searches is rejected. Explicit PE-radius overrides remain
supported; the recipes use the requested default of three frames.

Training labels, supervision delay, network loss, trigger holdoff and GPIO
behavior do not change. A running process using the old code still selects
thresholds/epochs under its loaded old policy. Re-evaluate its validation results
under the new policy afterward; an optimal threshold or selected epoch can
change. No checkpoint is overwritten or calibration automatically applied by
this metric update.

The comparison tool's fixed ±20 ms **raw-time diagnostic** remains separate
from these frame tiers and records their radii alongside its raw-event counts.
Its `matched` count should not be read as the frame-tier TP count.

## Preprocessing horizons

ECG normalization and slow slope history retain their raw-sample horizons.
Pulse normalization defaults to 1,000 frames; quality flat/median histories
default to 20/400 frames. When the input holds a pulse value across eight ECG
samples, both four-sample frames reuse that raw measurement. The UART decoder
publishes auxiliary updates at four-ECG-sample group boundaries, and consecutive
pulse-bearing groups can update on successive boundaries; eight samples are
therefore not guaranteed to share one pulse value. Each accepted model frame
contains four ECG samples per lead and one normalized pulse scalar when pulse
input is enabled. If pulse alignment is enabled and input starts
inside a pulse block, initial inconsistent samples are discarded before regular
four-sample framing resumes. Pulse alignment never enlarges frames to eight
samples. Signal-quality statistics are fitted on the new training frames.

Prepped caches fingerprint the mode, target, frame size, normalization and
quality histories. Delayed models reject legacy unmarked caches: regenerate
from raw recordings. Weights, raw annotation files and held-out data are never
rewritten by the mode conversion.

## Prediction, export and GUI

Runtime prediction reads the target contract from the saved model config;
there is no flag to reinterpret old weights. Use the existing offline commands:

```bash
.venv/bin/python run/offline_predict.py data/evaluation/session.parquet \
    --modelname delayed_100hz_v1 --model-dir models/delayed_100hz_v1

.venv/bin/python run/offline_predict_trt.py data/evaluation/session.parquet \
    --modelname delayed_100hz_v1 --model-dir models/delayed_100hz_v1
```

TensorRT requires an engine built for the executing target. Delayed offline TRT
uses sequential causal inference even if an offline batch size was requested;
legacy batching is unchanged. Rebuild both native components before use:

```bash
bash run/rebuild_native.sh --python-extension
```

ONNX exports carry target metadata, checked during preparation even if ONNX
numerical parity is explicitly skipped. Delayed deployments and nondefault
quality histories use manifest schema `dkfz-live-deployment-v2`. Old native
binaries reject v2, preventing silent current-frame interpretation. Existing
default current-frame manifests remain v1. Native config validation enforces
the mode, delay, raw rate, frame size, window and single-output contract.

Use the [existing preparation workflow](terminal.md#prepare-and-deploy-a-model)
after training/evaluation. Build TensorRT engines **on the target Jetson**.
The TRT parity check derives its defaults from the new config: 100 evaluations/s
and a 10 ms p99 inference budget. This inference-only check does not establish
end-to-end real-time performance.

The GUI's **Finished models** selector identifies frame rate, window duration
and previous-frame primary target. **Trained prediction** displays the selected
contract; deployed detector information shows the target and physical holdoff.
Probability telemetry labels the previous-frame primary. This is a read-only
training property: select separately trained weights to change it. The usual
explicit deployment/start actions remain required.

The integrated delayed runtime also resets at a UART decoder resynchronization
boundary, even though decoded sample indices remain consecutive. The wire
protocol has no absolute sample counter: entirely missing intact groups cannot
be detected from bytes alone, and positions after lost input cannot be mapped
to an uninterrupted recording without an external timing reference.

Log `frame_idx`, `raw_idx`, `raw_frame_end_idx` and `time_ms` identify the target
frame. `observed_frame_idx`, `observed_raw_idx`, `observed_raw_frame_end_idx` and
`observed_time_ms` identify the newest consumed frame. Events retain separate
selected `peak_*` and actual `emitted_on_*` coordinates. Public names ending in
`_50hz` remain for compatibility and mean model-frame indices. Native host
decision/GPIO-assertion monotonic timestamps are separate from sample timestamps.

## Compare saved models on raw time

Train the [current_100hz control](../config/experiments/current_100hz.json) with
the same splits and physical settings to separate framing from target delay.
Use saved validation-selected thresholds; the comparison tool does no tuning:

```bash
.venv/bin/python run/compare_prediction_models.py data/evaluation/session.parquet \
    --model baseline models/original/train_config.json models/original/best.pth \
    --model current100 models/current100/train_config.json models/current100/best.pth \
    --model delayed100 models/delayed100/train_config.json models/delayed100/best.pth \
    --output-dir build/delayed-comparison
```

`--backend tensorrt` accepts target-built engines instead of checkpoints.
The tool intersects observable target intervals after warmup, guards edges by
the fixed matching radius (default ±20 ms), and matches merged events against
unchanged raw annotations. It saves hashes, probabilities, events, recovered
baseline misses, new misses, added/removed false events, signed/absolute timing
errors, and stream-trigger latency. False events across models are paired on
raw positions within the same fixed tolerance before additions are counted.
Miss diagnostics retain R position within its frame, target probability and
preceding premature-trigger/holdoff evidence. A frame-end R with a low score
does not establish that framing caused the miss.

Delayed offline metrics add `raw_events`. Existing frame-based reports remain
model-local; use this shared-interval tool for cross-rate comparisons. It does
not claim physical GPIO latency from offline emission times.

## Sustained Jetson latency measurement

Use [paced and live UART measurements](jetson-model-deployment.md#measure-sustained-processing)
after model qualification. That guide explains processing, host-decode,
backlog, deadline, and physical-output timing and their measurement limits.
