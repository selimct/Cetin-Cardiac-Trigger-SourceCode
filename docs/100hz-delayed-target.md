# Optional 100 Hz previous-frame prediction

The delayed model uses four raw samples per feature frame at 400 Hz, producing
100 frames per second. At completion of frame `t`, the latest LSTM state predicts
the target at `t - 1`. This adds one frame (10 ms) of context. Event positions
refer to the predicted frame; emission times are never backdated.

This requires separately trained weights and a matching configuration. Changing
the frame size or prediction mode of a trained 50 Hz model does not convert it.
The normal current-frame 50 Hz model and optional two-output
[recovery model](recovery-head.md) remain separate choices.

## Train and publish

Use the [workstation setup and annotation workflow](training.md) first. Preview
the delayed training/publication command:

```bash
./train_models_lopo.sh delayed-100hz --only ecg3 --dry-run
```

Remove `--dry-run` to train. The published directory is
`finished_models/ecg3/delayed_100hz/`. Recipes are under `config/lopo_recipes/`;
`config/experiments/delayed_100hz.json` and `current_100hz.json` describe delayed
and current-frame 100 Hz settings for controlled comparisons.

A delayed configuration declares `prediction_mode=delayed_100hz`,
`prediction_delay_frames=1`, `raw_hz=400`, `frame_size_raw=4`, and preprocessing
contract 4. A one-second window contains 100 frames. Physical normalization
history and holdoff durations must be preserved when changing the model rate.
The default holdoff is 200 ms: 20 frames at 100 Hz or 10 at 50 Hz. Explicit saved
holdoff settings take precedence.

After updating an existing checkout, rebuild both native components:

```bash
bash run/rebuild_native.sh --python-extension
```

Training caches must match the preprocessing contract and prediction settings;
regenerate incompatible caches from raw recordings. Keep each checkpoint with
its own unedited training configuration.

## Evaluate recorded input

Use raw-rate parquet recordings with reviewed annotations from patients excluded
from training when measuring accuracy:

```bash
.venv/bin/python run/offline_predict.py data/evaluation/session.parquet \
    --modelname ecg3_delayed_100hz_full \
    --model-dir finished_models/ecg3/delayed_100hz
```

Replace the model identity with the exact `modelname` in the selected config.
`run/offline_predict_trt.py` provides the TensorRT alternative after building an
engine on the target Jetson. Predictions and metrics stay with the resolved
model directory.

Metrics schema 4 uses the following absolute frame offsets:

| Rate | TP | PE | oPE |
| --- | --- | --- | --- |
| 50 Hz | 0 | 1 | 2 |
| 100 Hz | 0–1 | 2–3 | 4–5 |

Use `run/compare_prediction_models.py --help` to compare models on a common
raw-time interval, including sample-axis emission latency. Frame categories and
raw-time diagnostics measure different things. Do not relabel historical metric
counts or tune thresholds on held-out test recordings.

## Prepare and qualify on the Jetson

Transfer the selected checkpoint and matching config, then use the
[isolated preparation workflow](jetson-model-deployment.md) with `--prepare-only`
and representative real recordings. The Jetson UART runtime derives ECG3 as
ECG2 − ECG1; confirm the same lead definition for training and parity inputs.

Preparation checks numerical and native-stream parity and retains its evidence
in a new bundle. The replay executable `native_live_runtime` and its Python JSONL
reference are required by these checks. They are included in this distribution.
Development regression suites and historical implementation reports are not
bundled.

Use [model qualification](model-qualification.md) to evaluate fixed-weight
accuracy and sustained processing under the candidate
[service scheduling profile](service-latency.md). A paced replay measures host
processing, rather than physical acquisition or GPIO edges. Check sample and
evaluation counts, invalid-input/reset/loss counters, backlog and deadlines.
`run/summarize_inference_latency.py` summarizes native probability/latency logs.

Neither numerical parity nor a successful timing run establishes accuracy on
unseen patients or a universal worst-case timing bound. Measure physical input
and output timing separately before enabling triggering. Activation and service
startup remain explicit operator actions after review.
