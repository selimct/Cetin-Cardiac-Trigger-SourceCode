# Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Developed at the German Cancer Research Center (DKFZ).

"""PyTorch backend for the live ECG streaming predictor."""

from __future__ import annotations

import numpy as np
import torch

from dkfz.live.feature_framing import LiveFeatureFramer
from dkfz.live.streaming_predictor import LiveStreamingPredictor


def default_torch_device() -> torch.device:
    """Return CUDA when available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LiveLSTMPredictor(LiveStreamingPredictor):
    """Live predictor backed by a PyTorch ECG_RNN model."""

    def __init__(
        self,
        model: torch.nn.Module,
        window_size: int,
        threshold: float,
        holdoff_frames: int,
        device: str | torch.device | None = None,
        model_output_is_probability: bool = False,
        framer: LiveFeatureFramer | None = None,
        **framer_kwargs,
    ):
        self.device = default_torch_device() if device is None else torch.device(device)
        self.model = model.to(self.device).eval()
        self.model_output_is_probability = bool(model_output_is_probability)
        framer_kwargs.setdefault(
            "prediction_delay_frames", int(getattr(model, "prediction_delay_frames", 0))
        )
        framer_kwargs.setdefault(
            "recovery_head", bool(getattr(model, "recovery_head", False))
        )
        super().__init__(
            window_size=window_size,
            threshold=threshold,
            holdoff_frames=holdoff_frames,
            framer=framer,
            **framer_kwargs,
        )

    def _extract_probability(self, model_output) -> float:
        """Convert raw model output into a scalar probability."""
        if isinstance(model_output, (tuple, list)):
            model_output = model_output[0]
        if not torch.is_tensor(model_output):
            raise TypeError("Model output must be a tensor or tuple/list of tensors.")

        y = model_output.reshape(-1)
        expected = 2 if self.recovery_head else 1
        if y.numel() != expected:
            raise ValueError(
                f"Expected {expected} model outputs, got {y.numel()} values."
            )

        probabilities = y if self.model_output_is_probability else torch.sigmoid(y)
        self.last_recovery_probability = (
            float(probabilities[1].item()) if self.recovery_head else None
        )
        return float(probabilities[0].item())

    def _run_model_on_current_window(self) -> float:
        window = np.stack(self._feature_window, axis=0).astype(np.float32)
        x = torch.from_numpy(window).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            output = self.model(x)
        return self._extract_probability(output)
