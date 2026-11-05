# Active model slot

This directory contains the one model selected for the systemd service:

- `model.engine` is the TensorRT engine built on, or verified for, this Jetson.
- `model.conf` is the readable, fingerprinted native deployment configuration
  referencing `model.engine`.

`model.conf` is generated from the training configuration; it is not a
byte-for-byte copy of `train_config.json`. It records the model name, variant,
source model directory, preprocessing contract, feature order, threshold,
tensor interface, and engine fingerprint needed by the native runtime.

## Quick guide

Run from the repository root. Replace the example name and directory together:

```bash
sudo systemctl stop dkfz-native-uart-live.service

./prepare_model.sh target_r \
    --model-dir models_noCV/target_r \
    --jetson-only

sudo systemctl restart dkfz-native-uart-live.service
```

The Jetson phase builds the engine in temporary staging, checks PyTorch versus
TensorRT when the checkpoint is available, validates the generated configuration
and engine, and only then replaces this slot. To activate directly from an ONNX
model without the coordinator, use:

```bash
sudo systemctl stop dkfz-native-uart-live.service
bash run/deploy_model.sh MODEL_NAME --model-dir SOURCE_MODEL_DIR
sudo systemctl restart dkfz-native-uart-live.service
```

Inspect and validate the active model without opening UART or GPIO:

```bash
grep -E '^(model_name|model_variant|source_model_dir|engine_)' \
    deploy/model/model.conf

build_native/inference/native_uart_live_runtime \
    --config config/native_uart_live.conf \
    --validate-only \
    --check-engine
```

Both generated files are intentionally ignored by Git. Stop the service before
replacing them; model changes require a service restart but not
`systemctl daemon-reload`. Always select alternate source roots with
`--model-dir`: a same-named config, checkpoint, ONNX file, or existing engine
from another model directory is rejected.

## Disclaimer

Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>

This software is provided "as is", without any warranty or condition. To the
extent permitted by law, the copyright holder will not be liable for damages
arising from the software, its use, or its nature. See the [LICENSE](LICENSE)
file for the complete PolyForm Noncommercial License 1.0.0 terms.
