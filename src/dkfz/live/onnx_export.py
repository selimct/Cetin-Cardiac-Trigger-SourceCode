# Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Developed at the German Cancer Research Center (DKFZ).

"""ONNX export helpers for the live ECG LSTM model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import warnings

import torch

from dkfz.core.model import ECG_RNN
from dkfz.live.model_variants import default_config_path, require_config_modelname


DEFAULT_ONNX_OPSET = 18


class LogitOnly(torch.nn.Module):
    """Wrap ECG_RNN so deployment exports one logit tensor, not LSTM state."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.model(x)
        return logits


def load_train_config(
    modelname: str,
    config: str | Path | None = None,
    model_variant: str | None = None,
    model_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Load a training config and verify its declared model identity."""
    config_path = Path(config) if config is not None else default_config_path(
        modelname,
        model_variant,
        model_dir,
    )
    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    require_config_modelname(
        cfg,
        modelname=modelname,
        config_path=config_path,
    )
    return cfg


def build_model_from_config(
    cfg: dict[str, Any],
    checkpoint: str | Path,
    map_location: str | torch.device = "cpu",
) -> ECG_RNN:
    """Construct ECG_RNN from config and load a checkpoint state dict."""
    input_dim = int(cfg.get("input_dim") or len(cfg["feature_cols"]))
    hidden_dim = int(cfg["hidden_dim"])
    num_layers = int(cfg["num_layers"])
    model = ECG_RNN(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    )
    state_dict = torch.load(checkpoint, map_location=map_location)
    model.load_state_dict(state_dict)
    return model.eval()


def export_model_onnx(
    model: torch.nn.Module,
    output: str | Path,
    window_size: int,
    input_dim: int,
    opset: int = DEFAULT_ONNX_OPSET,
    batch_size: int = 1,
    dynamic_batch: bool = False,
) -> Path:
    """Export a logit-only live LSTM to ONNX."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    wrapped = LogitOnly(model).eval()
    dummy = torch.randn(
        int(batch_size),
        int(window_size),
        int(input_dim),
        dtype=torch.float32,
    )

    dynamic_shapes = None
    if dynamic_batch:
        dynamic_shapes = {
            "x": {
                0: torch.export.Dim("batch", min=1),
            }
        }

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=(
                r"The tensor attributes self\.model\.lstm\._flat_weights\[0\].*"
                r"were assigned during export\."
            ),
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r"`isinstance\(treespec, LeafSpec\)` is deprecated.*",
            category=FutureWarning,
        )
        torch.onnx.export(
            wrapped,
            dummy,
            output,
            export_params=True,
            input_names=["x"],
            output_names=["logit"],
            opset_version=int(opset),
            dynamo=True,
            external_data=False,
            do_constant_folding=True,
            dynamic_shapes=dynamic_shapes,
        )
    return output
