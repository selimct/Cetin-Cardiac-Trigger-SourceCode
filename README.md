<!--
# Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Developed at the German Cancer Research Center (DKFZ).
-->

# A Real-Time Cardiac Triggering System for Ultra-High-Field MRI using Machine Learning

[![Build and distribution check](https://github.com/selimct/Cetin-Cardiac-Trigger-SourceCode/actions/workflows/build.yml/badge.svg)](https://github.com/selimct/Cetin-Cardiac-Trigger-SourceCode/actions/workflows/build.yml)
[![Trigger GUI](https://raw.githubusercontent.com/selimct/DKFZ-MRI-Cardiac-Trigger-GUI/refs/heads/main/packaging/github-badge.svg)](https://github.com/selimct/DKFZ-MRI-Cardiac-Trigger-GUI)
[![ParquetPlot](https://raw.githubusercontent.com/selimct/ParquetPlot/refs/heads/main/resources/parquetplot-gitlab-badge.svg)](https://github.com/selimct/ParquetPlot)

This project presents a machine-learning-based hybrid ECG-POX R-peak detector for real-time cardiac triggering in ultra-high-field MRI. It was developed by Furkan Selim Cetin as part of their master's thesis at DKFZ Heidelberg.

The repository includes the research workflow and the real-time triggering system, from recorded physiological signals to a trained model and its use during live acquisition.

This public distribution includes setup, data preparation, model training,
evaluation, deployment and live operation. Annotated recordings are included
under `data/live/`, with delayed 100 Hz models under `finished_models/base/`,
`finished_models/slope/` and `finished_models/ecg-only/`. Training and metric
outputs are generated locally. Development test suites,
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
- [Prediction metrics and synthetic benchmarks](docs/metrics.md)
- [Jetson deployment and qualification](docs/jetson-model-deployment.md)
- [Recovery models](docs/recovery-head.md)
- [Default 100 Hz previous-frame predictor](docs/100hz-delayed-target.md)

## Prediction metrics

Run `./metrics.sh normal` or `./metrics.sh synthetic` for the preferred
six-recording benchmark using non-recovery 100 Hz models. Add
`--include-50hz` for both rates, or name a 50 Hz model explicitly. See the [metrics guide](docs/metrics.md) for model
selection, concurrent workers, incremental additions, saved predictions,
cache reuse, synthetic settings, and report definitions.

## Authorship

This work was authored in 2026 by **Furkan Selim Cetin** and developed at the
**German Cancer Research Center (DKFZ)**.

## Disclaimer

Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>

This software is provided "as is", without any warranty or condition. To the
extent permitted by law, the copyright holder will not be liable for damages
arising from the software, its use, or its nature. See the [LICENSE](LICENSE)
file for the complete PolyForm Noncommercial License 1.0.0 terms.
