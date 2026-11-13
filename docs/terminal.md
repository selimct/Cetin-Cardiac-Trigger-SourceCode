<!--
# Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>

For a transferred delayed 100 Hz model, use the [isolated Jetson preparation
workflow](jetson-model-deployment.md) with `--prepare-only` and real recordings
before considering activation. Legacy 50 Hz models remain supported.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Developed at the German Cancer Research Center (DKFZ).
-->

# Terminal alternatives

The optional [100 Hz previous-frame model](100hz-delayed-target.md) uses these
same preparation commands with a separately trained checkpoint/config pair.
Its guide covers raw-time comparison and sustained latency telemetry.

Use the [Trigger GUI workflow](README.md) for deployment and everyday operation.
This file collects the corresponding terminal procedures for troubleshooting
or installations operated without the GUI. Training and host provisioning stay
in their respective guides because the GUI does not perform those tasks.

Run commands from the repository root on the indicated machine. Examples use
`my_model_cv` and `models/my_model_cv/nested_lopo/`, as produced by
[LOPO training](training.md). Stop any GUI-managed direct inference before
running a standalone receiver or runtime.

## Prepare and deploy a model

GUI equivalent: [Prepare and deploy a model](README.md#prepare-and-deploy-a-model).
For the complete pipeline on a configured Jetson, with the checkpoint and
matching training configuration already present:

```bash
sudo systemctl stop dkfz-native-uart-live.service
./prepare_model.sh my_model_cv \
    --model-dir models/my_model_cv/nested_lopo
```

This exports ONNX, runs the comparison checks, builds TensorRT, and activates
`deploy/model/`. Continue only after successful completion.

Alternatively, export ONNX on the workstation first using its
[project environment](training.md#workstation-setup):

```bash
./prepare_model.sh my_model_cv \
    --model-dir models/my_model_cv/nested_lopo \
    --workstation-only
```

Copy `best.pth`, `train_config.json`, and the exported ONNX file to the same
relative model directory on the Jetson, then run there:

```bash
sudo systemctl stop dkfz-native-uart-live.service
./prepare_model.sh my_model_cv \
    --model-dir models/my_model_cv/nested_lopo \
    --jetson-only
```

Both Jetson paths use the Python environment configured through `pythonbase`.
The checkpoint comparison requires compatible PyTorch and the project's
training dependencies. Build the engine on the target Jetson.

Validate configuration and engine loading without opening UART or GPIO:

```bash
build_native/inference/native_uart_live_runtime \
    --config config/native_uart_live.conf \
    --validate-only --check-engine
```

Start the service only after validation and the
[hardware checks](hardware.md#verify-the-installation) pass. Always keep the
selected model name and exact source directory consistent.

## Change startup recovery on the same model

Stop the service and any direct runtime before either preparation command. For a
published two-output model whose artifacts are already present, current-only is:

```bash
sudo systemctl stop dkfz-native-uart-live.service
./prepare_model.sh ecg3_recovery --model-dir finished_models/ecg3/recovery \
    --jetson-only --no-recovery
```

To return to the same model's saved recovery policy, stop again and run the same
command without `--no-recovery`. A null calibrated threshold or
`recovery_enabled=false` still leaves recovery disabled. Both modes preserve the
two-output architecture and source configuration. Do not change `recovery_head`.
Validate with the command above before restarting the service. This is the same
workflow as **Startup recovery → Deploy and start** in the GUI; it does not
require live switching or a recovery installation. Preparation retains its
existing Python/parity requirements. Use the complete preparation path if the
ONNX artifact is not present.

## Validate models on the Jetson before rollout

Use the [isolated Jetson preparation workflow](jetson-model-deployment.md)
with `--prepare-only` and representative real recordings. It exports the model,
builds a target-specific TensorRT engine and checks numerical and native-stream
parity without replacing the active deployment. Repeat for every model and
startup recovery policy you intend to use. Inspect `validation_report.json`;
recordings without emitted events do not establish positive trigger coverage.

For sustained service-profile and fixed-weight accuracy checks on a delayed
100 Hz primary model, follow [model qualification](model-qualification.md).
Physical UART/GPIO timing requires the separate hardware checks. The public
distribution does not bundle the development regression suites.

## Link the service unit

Review and qualify the [worker scheduling profile](service-latency.md) before
installing the current template. Linking/reloading is a separate deployment
step performed by the operator after qualification.

GUI equivalent: **Installed service → Install / repair unit link**.
Complete the [host provisioning](hardware.md#install-the-service) first.
Adjust the checkout path below to match the installation:

```bash
sudo systemctl stop dkfz-native-uart-live.service
sudo ln -sfn \
    /home/orin/work/live_stack/deploy/dkfz-native-uart-live.service \
    /etc/systemd/system/dkfz-native-uart-live.service
sudo systemctl daemon-reload
```

Changing the unit requires `daemon-reload`. Model, executable, and runtime
configuration changes require a service restart.

## Start, stop, reset, and inspect

GUI equivalents: **Start**, **Stop**, **Restart**, **Status**, and **Follow log**
on **Installed service**. After installation and hardware verification:

```bash
sudo systemctl start dkfz-native-uart-live.service
sudo systemctl status dkfz-native-uart-live.service
sudo journalctl --namespace=dkfz-live \
    --unit=dkfz-native-uart-live.service --follow
```

Stop triggering or clear accumulated state with the appropriate command:

```bash
sudo systemctl stop dkfz-native-uart-live.service
sudo systemctl restart dkfz-native-uart-live.service
```

Use an explicit restart between patients or after lead changes. An input pause
alone is a healthy waiting state; it does not require restarting the service.

## Change the live threshold

GUI equivalent: [Detector controls](README.md#adjust-the-detection-threshold),
which also provides calibrated presets, automatic restoration, and readback.
For direct file control, change only `manual_threshold` in
`deploy/model/model.conf` and save by atomic replacement:

```ini
manual_threshold=0.246
```

Values must be between 0 and 1. Valid changes apply at the next feature frame
without a restart or engine rebuild. During an input pause, application waits
for frames to resume. Confirm the active value in `runtime_status` or a
`threshold_changed` journal entry.

Clear the manual choice with:

```ini
manual_threshold=
```

The optional [`run/set_live_threshold.py`](../run/set_live_threshold.py) utility
controls a separate fallback file and early-emission policy. A saved manual
value takes priority over its numeric threshold. If that utility was used,
request its `--reset` before clearing the manual field to restore the model
default. Removing the override file alone does not reset a running detector.
The helper's script documentation covers its arguments and interpreter setup.
Saved manual values and valid override files persist across service restarts.

## Rebuild the runtime

GUI equivalent: **Model deployment / direct → Build / rebuild runtime**.
For an already configured checkout, stop inference and rebuild incrementally:

```bash
sudo systemctl stop dkfz-native-uart-live.service
bash run/rebuild_native.sh --target native_uart_live_runtime
```

Validate the deployment again, then start the service to use the rebuilt binary.

## Capture UART without a model

For capture with a deployed model, use the
[GUI capture workflow](hardware.md#record-input-for-troubleshooting). The
standalone receiver below also works without a deployed model. On the Jetson,
stop service and direct inference so only the receiver owns the serial port:

```bash
sudo systemctl stop dkfz-native-uart-live.service
mkdir -p data/uart
build_native/uart/uart_receiver_stream \
    --config config/uart_receiver.conf \
    --capture-byte-file data/uart/session.bin > /dev/null
```

Stop the receiver with Ctrl+C. The supplied configuration caps capture at 5 GB;
check its UART settings match the installation. Convert the recording on the
workstation with:

```bash
.venv/bin/python run/io_uart_bytes_to_parquet.py data/uart/session.bin
```

The result is `data/uart/session.parquet`. Resume the service when ready.
