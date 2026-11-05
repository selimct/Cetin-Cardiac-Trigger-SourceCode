<!--
# Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Developed at the German Cancer Research Center (DKFZ).
-->

# Cardiac triggering: using the system

This project detects ECG R-peaks for real-time cardiac triggering in
ultra-high-field MRI. It includes tools to prepare recordings, train and evaluate
models, and run a trained model on a Jetson Orin Nano.

During live operation, the Jetson receives ECG and pulse samples over UART,
processes them with a TensorRT model, and produces GPIO trigger pulses. The live
runtime is C++; Python is used for training, model preparation, and analysis.
Use the [Trigger GUI](https://github.com/selimct/DKFZ-MRI-Cardiac-Trigger-GUI)
on the desktop to deploy models and operate the Jetson over SSH.

- **First installation or wiring:** follow [Hardware setup](hardware.md).
- **Using an existing model:** continue below.
- **Building your own model:** follow [Training](training.md).
- **Operating without the GUI:** see [Terminal alternatives](terminal.md).

Examples use `my_model_cv`, produced by
the [LOPO training workflow](training.md) with base name `my_model`. Replace the
model name and source directory together when using your own model.

## Project layout

```text
live_stack/
├── setup.sh                 Environment setup and initial native build
├── prepare_model.sh         Prepare a trained model for the Jetson
├── run/                     Scripts for preparation, training, and evaluation
├── src/dkfz/
│   ├── core/                Network, datasets, and signal-processing utilities
│   ├── live/                Training and prediction workflows
│   └── native/              Python interface to shared C++ processing
├── native/                  C++ acquisition, preprocessing, and live runtime
├── config/                  Runtime and acquisition settings
├── deploy/                  Service template and active model slot
├── data/                    Recordings and prepared datasets
│   └── live/                Annotated dataset repository, grouped by patient
├── models/                  Training outputs, configurations, and predictions
├── notebooks/               Exploration and analysis
├── tests/                   Python tests; C++ tests are under native/tests/
├── build_native/            Generated native executables
├── docs_rework/             User guides and terminal alternatives
└── docs/                    Earlier documentation and thesis references
```

The main distinction is between a **source model directory**, such as
`models/my_model_cv/nested_lopo/`, and the **active deployment** in `deploy/model/`:

| Location | Purpose |
| --- | --- |
| `models/my_model_cv/nested_lopo/best.pth` | Final PyTorch checkpoint from LOPO training. |
| `models/my_model_cv/nested_lopo/train_config.json` | Matching architecture, preprocessing, and threshold settings. |
| `deploy/model/model.engine` | TensorRT engine selected for live use. |
| `deploy/model/model.conf` | Generated configuration identifying that engine and its model settings. |
| `config/native_uart_live.conf` | UART, GPIO, and runtime settings for this installation. |

Keep a checkpoint and its matching training configuration together. The active
engine and model configuration are generated during deployment.

## Connect with the Trigger GUI

Install the desktop application from the
[GUI repository](https://github.com/selimct/DKFZ-MRI-Cardiac-Trigger-GUI), whose
README covers installation and SSH setup. The desktop needs the GUI; the Jetson
needs this project's checkout and the configured Python/native environment.

In the GUI's **Connection** dialog, enter the Jetson host, user, port, and SSH
identity. Set **Remote project root** to the Jetson checkout, for example
`/home/orin/work/live_stack`. Select **Check** to test the connection. On a first
connection, use the option to trust and save a new host key after verifying the
host. Service actions also require remote sudo access; the dialog accepts a
temporary sudo password when needed.

Checks are manual. After changing connection or model settings, select
**Check availability** again; changing a field alone sends no request.

## Prepare and deploy a model

Do this when installing a model for the first time or replacing the active one.
For a model that is already deployed, go straight to [operation](#start-stop-and-check-the-system).

Complete [Jetson setup](hardware.md#jetson-setup) first. Copy the final
`best.pth` and matching `train_config.json` from the workstation to
`models/my_model_cv/nested_lopo/` in the Jetson checkout. Fold checkpoints and
evaluation outputs can stay on the workstation. Use the same project revision
on both machines.

1. On **Installed service**, select **Stop**. Also stop any direct inference.
2. Open **Model deployment / direct**. Set **Model name** to `my_model_cv`,
   **Model source directory** to `models/my_model_cv/nested_lopo`, and
   **Variant** to `best`.
3. Set **Jetson preparation** to **Complete: PTH → ONNX → TensorRT**, then select
   **Check** and **Prepare + activate model**.
4. Wait for export, comparison checks, engine building, and activation to finish
   successfully. Select **Check** again, then **Validate config + engine**.

This builds the engine on the target Jetson and activates the matching files in
`deploy/model/`. The full Jetson Python environment must include the export and
training dependencies, including compatible PyTorch. Resolve any reported error
before starting the new model.

Models kept directly under `finished_models/` also appear through **Refresh
models** and the **Finished models** selector on **Installed service**. For
those selections, **Re-export weights + activate** runs the complete pipeline.
Selecting a model alone does not activate it.

The [terminal reference](terminal.md#prepare-and-deploy-a-model) includes the
alternative workflow that exports ONNX on the workstation first.

## Start, stop, and check the system

After the service has been [installed](hardware.md#install-the-service), open
**Installed service** and select **Check availability**. Complete the
[hardware checks](hardware.md#verify-the-installation), then select the hardware
confirmation checkbox and **Start**. Use **Status** to check the service and
**Follow log** to view live diagnostics. Use **Stop** to stop triggering.

Status should show the service running. In the log, check that decoded
sample and inference counts advance; UART traffic alone does not confirm valid
signal input. Trigger counts should advance when events are detected. Physical
pulse timing is checked separately during hardware setup.

## Make changes during operation

### Adjust the detection threshold

A lower threshold generally admits weaker peaks; a higher one generally rejects
more of them. Check both missed detections and extra triggers after a change.

Open **Live thresholds** and select **Refresh deployed thresholds** to load the
active model's settings. Enter a value between 0 and 1 and select **Apply
custom**, or choose **Use Weighted DER** or **Use Historical FP-first** when
those calibrated presets are available. These explicit choices are saved as
manual selections. Select **Restore automatic** to return to the model default.

Select **Refresh deployed thresholds** after a change. Check **Request status**
and **Confirmed detector policy**; a saved request is not proof that it is
active. During an input pause, application can wait for usable input. Threshold
changes preserve signal history and need no service restart or engine rebuild.

The **Early emission** checkbox controls when a detected peak produces its
trigger: strong peaks can emit earlier on the falling edge, while weaker peaks
still wait for the selected detection threshold. It changes trigger timing;
check the confirmed policy and timing after changing it.

### Reset or replace the model

On **Installed service**, use **Restart** between patients or after lead changes
to clear accumulated signal and detector state. Recheck availability and confirm
the hardware configuration if the GUI requests it.

The runtime also pauses and resets on invalid or disconnected input. It resumes
after stable input returns and a fresh model window has filled. An interruption
that does not match those checks still requires an explicit restart.

| Change | Action |
| --- | --- |
| Threshold or early emission | Use **Live thresholds**, then refresh and confirm application. |
| UART, GPIO, or other runtime configuration | Edit `config/native_uart_live.conf`, then use **Restart**. |
| Model weights or preprocessing | Stop, prepare and activate the matching model, check, then start. |
| Service unit | After editing it, use **Reload unit**, check availability, then restart. |

## Use a model on a recording

For prediction without connected hardware, use the workstation environment and
a raw-rate parquet recording compatible with the model:

```bash
.venv/bin/python run/offline_predict.py data/new_patient/session.parquet \
    --modelname my_model_cv --model-dir models/my_model_cv/nested_lopo
```

Predictions and trigger events are written under the source model directory's
`prediction/` folder. If the recording has reviewed `target` annotations,
evaluation reports are written under its `metrics/` folder too. Use a patient
excluded from training when measuring performance on new patients.

## If something goes wrong

| Symptom | First check |
| --- | --- |
| Service fails to start | Use **Check availability** and **Follow log** for missing artifacts, invalid settings, or permissions. |
| No decoded samples | Check wiring, UART selection, pinmux, and competing serial processes. |
| Inference is paused | Check patient/lead connections and signal validity; recovery needs fresh input. |
| Events appear but no physical pulse | Check GPIO mapping, line ownership, polarity, and the receiving device. |
| Threshold change has no effect | Refresh **Live thresholds** and inspect request status and the confirmed policy. |

