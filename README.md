<!--
# Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Developed at the German Cancer Research Center (DKFZ).
-->

# A Real-Time Cardiac Triggering System for Ultra-High-Field MRI using Machine Learning

[![Build and distribution check](https://github.com/selimct/Cetin-Cardiac-Trigger-SourceCode/actions/workflows/build.yml/badge.svg)](https://github.com/selimct/Cetin-Cardiac-Trigger-SourceCode/actions/workflows/build.yml)
[![Trigger GUI](https://raw.githubusercontent.com/selimct/DKFZ-MRI-Cardiac-Trigger-GUI/refs/heads/main/packaging/github-badge.svg)](https://github.com/selimct/DKFZ-MRI-Cardiac-Trigger-GUI)
[![ECG/POX Data](https://raw.githubusercontent.com/selimct/DKFZ-MRI-Cardiac-Trigger-GUI/refs/heads/main/packaging/data-badge.svg)](https://github.com/selimct/Cetin-UHF-ECGPOX-Data.git)
[![ParquetPlot](https://raw.githubusercontent.com/selimct/ParquetPlot/refs/heads/main/resources/parquetplot-gitlab-badge.svg)](https://github.com/selimct/ParquetPlot)

This project presents a machine-learning-based hybrid ECG-POX R-peak detector for real-time cardiac triggering in ultra-high-field MRI. It was developed by Furkan Selim Cetin as part of their master's thesis at DKFZ Heidelberg.

The repository includes the research workflow and the real-time triggering system, from recorded physiological signals to a trained model and its use during live acquisition.

This public distribution includes setup, data preparation, model training,
evaluation, deployment and live operation. Recordings and trained model
artifacts are supplied separately; `models/`, `finished_models/` and metric
outputs are created or populated by the user. Development test suites,
internal patches and deprecated executables are omitted.

## Project scope

- **Data preparation:** converting recordings and preparing reviewed R-peak
  annotations for model development.
- **Model development:** training and evaluating models for causal ECG
  R-peak detection.
- **Real-time triggering:** deploying trained models to an embedded system
  that processes incoming signals and produces cardiac trigger pulses.
- **Validation:** comparing model behavior across research and deployment
  environments and evaluating detection accuracy and timing.

## Documentation

The guides in [docs/](docs/README.md) cover setup, model training, and operation
with the Trigger GUI.

- [Project layout, deployment, and GUI operation](docs/README.md)
- [Hardware connections and Jetson setup](docs/hardware.md)
- [Annotated data and leave-one-patient-out training](docs/training.md)
- [Terminal alternatives](docs/terminal.md)
- [Optional 100 Hz previous-frame predictor](docs/100hz-delayed-target.md)

## Prediction metrics

Run the preferred six-recording benchmark from the project root:

```bash
./metrics.sh normal
./metrics.sh synthetic
```

Both commands evaluate every published model in `finished_models/`, including
normal models and their recovery and `delayed_100hz/` variants. They use p02 head,
p14 heart/head,
and p16 head/heart/pelvis. `./metrics.sh` defaults to the normal run.

To evaluate only normal 50 Hz and delayed 100 Hz models, skipping all recovery
versions (including frozen and LOPO recovery), add `--exclude-recovery-models`:

```bash
./metrics.sh normal --exclude-recovery-models
./metrics.sh synthetic --exclude-recovery-models
```

The option also works with `create_prediction_metrics.sh`. It filters model
selection; `--no-recovery` instead disables recovery events within the selected
trained models.

Both modes use `--only-missing`. By default, full prediction/event parquets are
temporary; small score/annotation caches under `metrics/.cache/` allow reports
and plots to be rebuilt without repeating inference. Synthetic runs do not save
modified source recordings by default.

Add `--save-prepped` to keep full prediction/event parquets. Normal outputs go
under `metrics/prepped/<model>/`; synthetic outputs go under
`metrics/prepped/synthetic/main_<frequency>hz/<model>/`. Matching saved files
are reused on subsequent runs. Changing input data, model weights, or prediction
settings invalidates the cache; changing only this saving flag does not.
Prediction caches track runtime settings and inference/native code separately
from training settings and calibration audit metadata. Plot-only changes rebuild
plots while reusing predictions. A newly selected recovery threshold still
invalidates that model's predictions and reports.
If only a compact cache exists, the first `--save-prepped` run repeats inference
once to produce the full files; subsequent matching runs reuse them.

Reports go under `metrics/` for normal runs and `metrics/synthetic/` for synthetic
runs. Each has `prediction_metrics_overview.{txt,csv}` for pooled results and
`prediction_metrics_per_input.{txt,csv}` for per-input comparisons. Synthetic
reports group results by gradient frequency.

Append options from `create_prediction_metrics.sh` to either command, for example
`./metrics.sh normal --model ecg3 --model ecg3_recovery`. Use `--dry-run` to
print the expanded command, or `--help` for the shortcut options.

## Authorship

This work was authored in 2026 by **Furkan Selim Cetin** and developed at the
**German Cancer Research Center (DKFZ)**.

## Disclaimer

Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>

This software is provided "as is", without any warranty or condition. To the
extent permitted by law, the copyright holder will not be liable for damages
arising from the software, its use, or its nature. See the [LICENSE](LICENSE)
file for the complete PolyForm Noncommercial License 1.0.0 terms.
