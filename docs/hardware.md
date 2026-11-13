<!--
# Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Developed at the German Cancer Research Center (DKFZ).
-->

# Hardware setup

Use this guide for a new Jetson installation or a change to the physical
connections. For everyday operation, see the [main guide](README.md).

## Connections

The production path is:

```text
ECG/pulse source → UART input → Jetson model inference → GPIO trigger → receiver
```

You need a Jetson Orin Nano developer kit with JetPack, the project's compatible
physiological UART source, a compatible trigger receiver, and a shared ground.
The decoder expects this source's protocol; it cannot read an arbitrary serial
ECG device.

| J12 physical pin | Use | Default Linux mapping |
| --- | --- | --- |
| 10 | UART receive: ECG and pulse input | `/dev/ttyTHS1` |
| 31 | Trigger output | `tegra234-gpio`, line offset 106 |
| 29 | Optional diagnostic edge capture | `tegra234-gpio`, line offset 105 |
| GND, e.g. pin 6 | Common signal ground | — |

These are **physical header pin numbers**. Linux device names, GPIO offsets, and
line ownership must be verified on the installed JetPack image. The carrier-board
specification, available from [NVIDIA's Jetson Download Center](https://developer.nvidia.com/embedded/downloads),
provides the connector reference.

The header uses 3.3 V logic. Connect a common ground and use a compatible
high-impedance trigger input. Pins 29 and 31 pass through weak level translators;
avoid contention or direct connection to higher-voltage or differential signals.
Verify output polarity and pulse timing with an oscilloscope or logic analyzer
before connecting the final receiver.

## Jetson setup

Run commands from the repository root on the Jetson. Use JetPack's CUDA and
TensorRT packages. Model preparation also needs a compatible Python environment
with PyTorch and the project training dependencies for its comparison checks.
Live inference itself has no Python dependency.

The setup and model-preparation helpers find the Jetson environment through a
`pythonbase` alias in `~/.bash_aliases`. For the current installation, it is:

```bash
alias pythonbase='source /home/orin/envs/jet/bin/activate'
```

Adjust that path for another installation, then run:

```bash
bash setup.sh --profile jetson
```

Setup creates or reuses the selected environment, installs training/ONNX
dependencies, and builds the native runtime with TensorRT required. It
preserves JetPack's system packages. Ensure a JetPack-compatible PyTorch
is available in that environment before preparing a model. Development
test suites are maintained upstream.

Review [`config/native_uart_live.conf`](../config/native_uart_live.conf).
Confirm the UART device, trigger GPIO, polarity, and pulse timing for your
installation. The supplied UART settings are 38,600 baud, 8E1, with 400 Hz ECG
samples. Check that UART pinmux is enabled, permissions allow access, and a
serial console is not using the port. `gpioinfo` shows GPIO line ownership.

After later C++ changes, stop inference and use **Build / rebuild runtime** on
the GUI's **Model deployment / direct** tab. Check availability afterward, then
restart the service to use the rebuilt executable. The
[terminal equivalent](terminal.md#rebuild-the-runtime) is kept separately.

## Install the service

The source template now includes a [bounded worker scheduling
profile](service-latency.md). Rebuild the runtime and qualify that profile before
installing it. Installing and starting the service are separate operator actions.

The service template runs as `dkfz-live` and reads the checkout through a
read-only mount at `/opt/dkfz-live`. Before installing it:

- Create the `dkfz-live` system user and group if they do not exist.
- Ensure the template's supplementary groups exist and grant the required
  serial, GPIO, and GPU access: `dialout`, `gpio`, `video`, and `render`.
- Set `BindReadOnlyPaths` in
  [`deploy/dkfz-native-uart-live.service`](../deploy/dkfz-native-uart-live.service)
  to the actual checkout location. The supplied path is
  `/home/orin/work/live_stack`.

Provision the state directory and journal policy once on the Jetson. The policy
path below uses the supplied checkout location; adjust it if necessary:

```bash
sudo install -d -o dkfz-live -g dkfz-live -m 0750 /var/lib/dkfz-live
sudo install -m 0644 \
    /home/orin/work/live_stack/deploy/journald@dkfz-live.conf \
    /etc/systemd/journald@dkfz-live.conf
```

In the [connected GUI](README.md#connect-with-the-trigger-gui), open **Installed
service**, select **Check availability**, then **Install / repair unit link**
while inference is stopped. Select **Check availability** again afterward.
This links the unit and reloads systemd; it does not start the service or enable
it on boot. The GUI also checks the account, groups, and state directory, which
must already be provisioned.

Diagnostics use the `dkfz-live` journal namespace with bounded retention. After
deployment and hardware verification, optional automatic startup on boot is a
one-time host setting:

```bash
sudo systemctl enable dkfz-native-uart-live.service
```

## Verify the installation

1. Confirm wiring, shared ground, UART/GPIO mappings, permissions, and trigger
   polarity/timing settings.
2. [Prepare and deploy a model](README.md#prepare-and-deploy-a-model), including
   **Validate config + engine** in the GUI.
3. Keep the final receiver disconnected while validating output with test
   equipment. Use the GUI's [service and log controls](README.md#start-stop-and-check-the-system).
4. Confirm decoded sample and inference counts advance, then verify actual GPIO
   pulses and their polarity/timing. Stop the service before changing wiring.

## Record input for troubleshooting

A raw UART capture lets you inspect the same input later. In the GUI:

1. Use **Stop** on **Installed service** so the serial port is available.
2. Open **Model deployment / direct** with the deployed model selected. Set
   **Trigger output** to `none` and **Raw UART capture** to `output/session.bin`.
3. Select **Check**, **Validate config + engine**, then **Start direct inference**.
4. When enough data has been collected, use **Stop direct inference** and clear
   the capture field. Return to **Installed service** to resume normal operation.

The capture contains the original bytes; the supplied runtime configuration
caps capture at 5 GB. This diagnostic workflow needs a valid deployed model.
For acquisition without a model, use the
[standalone terminal receiver](terminal.md#capture-uart-without-a-model).

Copy the capture to the workstation and convert it with the
[workstation environment](training.md#workstation-setup):

```bash
.venv/bin/python run/io_uart_bytes_to_parquet.py output/session.bin
```

The result is `output/session.parquet`, with centered raw-rate ECG and pulse
columns. It can be inspected or used for offline prediction. Training also
requires reviewed annotations. Pin-29 edge capture is a separate diagnostic
tool; its configuration and script help cover that less common workflow.

UART samples arrive in groups of three long packets followed by one short
packet. Pulse and respiration values carried in a group become active on the
first ECG sample of the following group and stay active until their next
update. `puls_updated` marks that first sample. Native UART/GPIO readout and
offline UART/GPIO conversion use this same boundary rule. Converted parquets
record `auxiliary_publication=following-group` in their metadata; regenerate
older UART-derived parquets from their captures to apply the corrected timing.

