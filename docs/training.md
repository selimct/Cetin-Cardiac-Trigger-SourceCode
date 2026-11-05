<!--
# Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Developed at the German Cancer Research Center (DKFZ).
-->

# Train your own model

The standard workflow is:

```text
Recordings → raw-rate parquet → reviewed R-peak annotations
           → patient-wise LOPO training and evaluation → deployment
```

The model learns to detect R-peaks from ECG and, by default, pulse input.
Training applies the causal preprocessing used for live prediction, converting
400 Hz raw samples into 50 Hz model features. Supply raw-rate recordings; the
training script builds the features and model windows. Use
[`run/train_lopo.py`](../run/train_lopo.py) for nested leave-one-patient-out
(LOPO) training and evaluation.

Run commands from the repository root. Examples use `my_model` as a new model
name and illustrative patient paths. Script documentation and `--help` describe
additional settings.

## Workstation setup

Use Python 3.10 or 3.11 and a C++ build toolchain. From a fresh checkout:

```bash
bash setup.sh --profile workstation
```

This creates or reuses `.venv`, installs the training/export dependencies, builds
the shared native processing code, and runs the checks. Use `.venv/bin/python`
for subsequent Python commands. A workstation build can run without TensorRT;
the deployment engine is built on the Jetson.

## Use the annotated dataset

Part of the annotated data is available in the
[ECG/POX Data repository](https://github.com/selimct/Cetin-UHF-ECGPOX-Data),
linked by the data badge in the project's README:

[![ECG/POX Data](https://raw.githubusercontent.com/selimct/DKFZ-MRI-Cardiac-Trigger-GUI/refs/heads/main/packaging/data-badge.svg)](https://github.com/selimct/Cetin-UHF-ECGPOX-Data)

The dataset is maintained as its own repository with the same structure as
`data/live/`: patient folders sit directly at its root. For a fresh data
directory, clone it into this project with:

```bash
git clone https://github.com/selimct/Cetin-UHF-ECGPOX-Data.git data/live
```

This places recordings under paths such as `data/live/p01/p01_out.parquet`.
These files are already annotated; continue with
[patient grouping](#group-recordings-by-patient) and training. The next section
covers preparing additional recordings yourself.

## Prepare your own recordings

For paired scanner/PMU logs, put each recording's `.ecg` and `.puls` files
together under `data/raw/<patient>/`. Convert one recording with:

```bash
.venv/bin/python run/prep_read_log_to_parquet.py data/raw/p01/session.ecg
```

This writes `data/raw/p01/session.parquet` with synchronized signals and
source offsets removed. The converter creates a placeholder `target` column;
it does not provide reviewed R-peak labels. For UART recordings, use the
[capture converter](hardware.md#record-input-for-troubleshooting) instead.

Create an initial annotation proposal:

```bash
.venv/bin/python run/prep_window_annotate.py data/raw/p01/session.parquet \
    --ecg-col ecg1_raw
```

The explicit ECG column lets this example use the converter output directly.
Choose an ECG lead and annotation settings appropriate for the recording. The
tool uses pulse timing to propose R-peaks and writes
`data/flagged/p01/session.parquet`.

Inspect and correct the proposed labels before training. In `target`, a value
of 1 marks an R-peak sample and 0 marks other samples. Automatically proposed
labels and scanner trigger markers are not a substitute for reviewed ground
truth. Keep the original recordings so corrections remain traceable.

For the standard configuration, prepared files contain:

| Column | Meaning |
| --- | --- |
| `time_ms` | Sample time in milliseconds. |
| `ecg1_raw`, `ecg2_raw`, `ecg3_raw` | Raw-rate ECG channels after source calibration. |
| `puls_raw` | Raw-rate pulse signal after source calibration. |
| `target` | Reviewed binary R-peak labels. |
| `raw_idx` | Sample index, preserving recording continuity. |

The supplied PMU and UART converters derive `ecg3_raw` as ECG2 minus ECG1.
The current default training configuration uses all three ECG columns. Other
channel choices are configured in the training script. Do not supply
pre-normalized features or subtract source offsets a second time.

Keep separate recordings identifiable. Discontinuous sample indices and sequence
boundaries reset preprocessing; joining unrelated recordings into one continuous
stream would carry signal history across that boundary.

## Group recordings by patient

Use the dataset's existing layout, with one directory per patient directly
under `data/live/`. For example:

```text
data/live/
├── p01/
│   ├── p01_out.parquet
│   ├── p01_head.parquet
│   └── p01_head_gradient.parquet
├── p05/
│   └── p05_out.parquet
└── p06/
    └── p06_out.parquet
```

This shows only a few example recordings. Each patient directory can contain
multiple parquet files. Add your own reviewed recordings using the same layout,
keeping every recording from the same person together, including repeat
sessions. The script includes all parquet-containing patient directories under
the selected root, requires at least three patients, and creates the training,
validation, and test splits itself.

## Train with leave-one-patient-out evaluation

Choose a new model name so an earlier run's artifacts are kept separately:

```bash
.venv/bin/python run/train_lopo.py \
    --modelname my_model \
    --patient-root data/live
```

For each outer fold, the script holds out one patient for testing. Inner folds
use the remaining patients to select the training duration and detection
threshold. The held-out test patient does not participate in that selection.
Every patient takes a turn as the outer test patient.

After evaluating the folds, the script selects a training duration and threshold
from the inner-fold results and trains a final deployment model on all supplied
patients. Nested LOPO trains many models, so expect it to take longer than a
single training run.

With the example name, results are written to
`models/my_model_cv/nested_lopo/`. The saved model name is `my_model_cv`:

| Output | Use |
| --- | --- |
| `best.pth` | Final deployment model trained on all supplied patients. |
| `train_config.json` | Matching model, preprocessing, and selected threshold settings. |
| `summary.json` | Overall held-out evaluation and final model information. |
| `outer_results.csv` | Held-out results for each test patient. |
| `outer/` | Individual fold checkpoints, predictions, and evaluation artifacts. |
| `patient_frames.parquet` | Cached patient features, saved by default for reuse. |

Keep `best.pth` and `train_config.json` together. A checkpoint alone does not
describe how input signals must be processed. Rebuild cached features after
changing recordings or preprocessing settings.

## Evaluate before deployment

Start with `summary.json` and `outer_results.csv`. These report performance on
patients held out during the outer folds. Inspect individual patients' results
under `outer/` to understand where detection succeeds or fails.

Review missed beats, extra triggers, and timing errors alongside the signal
traces. Sensitivity describes how many reference beats were detected; positive
predictive value describes how many detections were correct. Event timing and
physical output delay also matter for triggering, so inspect those separately
from detection counts.

Model preparation checks agreement between PyTorch, ONNX, and TensorRT. That
agreement does not replace evaluation on held-out recordings or measurement of
the deployed system's physical timing.

The final `best.pth` has seen all supplied patients. Running it on those same
recordings is useful for inspection, but its results are not held-out test
performance. For additional independent evaluation, use new patients with the
[offline prediction command](README.md#use-a-model-on-a-recording).

## Deploy the result

Follow [Prepare and deploy a model](README.md#prepare-and-deploy-a-model) using
model name `my_model_cv` and source directory
`models/my_model_cv/nested_lopo/`. Deploy its final `best.pth` with the matching
`train_config.json`. Routine live threshold adjustments are covered in
[the operation guide](README.md#adjust-the-detection-threshold).

## Export model settings for a paper

Create an editable Word table from a saved configuration:

```bash
.venv/bin/python run/create_model_table.py --config finished_models/ecg3/train_config.json
```

The report is written to `dbg/ecg3/ecg3_parameters.docx`. You can also pass a
model directory or use `--modelname ecg3` to find an unambiguous config in
`finished_models/` or `models/` (including nested LOPO directories). With an
explicit config, `--modelname` overrides the report's output directory name.

Pass multiple model directories, names, or config paths to create **one Word
document** with each model's table starting on a new page:

```bash
.venv/bin/python run/create_model_table.py finished_models/ecg3 finished_models/slope_ecg3
```

This writes `dbg/model_tables/ecg3__slope_ecg3_parameters.docx`. Input order is
preserved. Captions are numbered `Table S1`, `Table S2`, etc., restarting at 1
in each document. Each caption reads “Model architecture and training parameters
for the <modelname> model.” You may also repeat `--config` for multiple JSON files. Use explicit
paths if a short model name matches more than one saved configuration.
For a custom combined filename, append `--modelname paper_models` to write
`dbg/model_tables/paper_models_parameters.docx`.

The compact two-column table groups the main architecture, input/preprocessing,
training, validation, and detection settings into about 15 rows. Inactive features
and implementation details are omitted. Missing settings use the shared
`TRAINING_DEFAULTS` and are marked with an asterisk. These current defaults
cannot establish historical settings that were not saved.

Single-model reports are named `dbg/<modelname>/<modelname>_parameters.docx`. Configs and
model artifacts are only read; the script writes exclusively under `dbg/` and
does not start training or require additional Word libraries.

For LOPO models, the report also looks for the saved
`models/<modelname>_cv/<cv_method>/patient_frames.parquet` cache (or its recorded
location). It reads only patient, sequence, frame-index, and recording identifiers.
The final-fit update count uses valid windows within each sequence, rounds batches
up per epoch, and multiplies by the selected epoch count. It excludes CV fits.
The count is nominal: mixed-precision overflow skips are not recorded. If the
cache is absent, the row is omitted; no frames are regenerated.
