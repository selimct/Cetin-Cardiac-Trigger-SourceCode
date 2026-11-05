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

## Authorship

This work was authored in 2026 by **Furkan Selim Cetin** and developed at the
**German Cancer Research Center (DKFZ)**.

## Disclaimer

Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>

This software is provided "as is", without any warranty or condition. To the
extent permitted by law, the copyright holder will not be liable for damages
arising from the software, its use, or its nature. See the [LICENSE](LICENSE)
file for the complete PolyForm Noncommercial License 1.0.0 terms.
