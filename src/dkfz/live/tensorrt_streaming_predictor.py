# Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Developed at the German Cancer Research Center (DKFZ).

"""TensorRT backend for the live ECG streaming predictor."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from dkfz.live.feature_framing import LiveFeatureFramer
from dkfz.live.io import require_raw_signal_columns
from dkfz.live.streaming_predictor import LiveStreamingPredictor

if TYPE_CHECKING:
    from dkfz.live.tensorrt_engine import TensorRTLogitEngine


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = np.exp(-x)
        return float(1.0 / (1.0 + z))
    z = np.exp(x)
    return float(z / (1.0 + z))


def _sigmoid_array(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid for a vector of logits."""
    x = np.asarray(x, dtype=np.float32)
    out = np.empty_like(x, dtype=np.float32)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


class LiveTensorRTPredictor(LiveStreamingPredictor):
    """Live predictor backed by a TensorRT logit engine."""

    def __init__(
        self,
        engine: TensorRTLogitEngine,
        window_size: int,
        threshold: float,
        holdoff_frames: int,
        model_output_is_probability: bool = False,
        framer: LiveFeatureFramer | None = None,
        **framer_kwargs,
    ):
        self.engine = engine
        self.model_output_is_probability = bool(model_output_is_probability)
        super().__init__(
            window_size=window_size,
            threshold=threshold,
            holdoff_frames=holdoff_frames,
            framer=framer,
            **framer_kwargs,
        )

    def _run_model_on_current_window(self) -> float:
        window = np.stack(self._feature_window, axis=0).astype(np.float32)
        if self.recovery_head:
            values = np.asarray(self.engine.infer_logits(window)).reshape(-1)
            if values.size != 2:
                raise ValueError("A recovery model must output two logits")
            probs = (
                values if self.model_output_is_probability else _sigmoid_array(values)
            )
            self.last_recovery_probability = float(probs[1])
            return float(probs[0])
        logit = self.engine.infer_logit(window)
        if self.model_output_is_probability:
            return float(logit)
        return _sigmoid(logit)

    def _infer_probabilities_batched(
        self,
        features: np.ndarray,
        batch_size: int,
    ) -> np.ndarray:
        """Run TensorRT over all sliding windows in feature order."""
        if int(batch_size) < 1:
            raise ValueError("batch_size must be positive")
        features = np.asarray(features, dtype=np.float32)
        if features.ndim != 2:
            raise ValueError(f"features must be 2D, got shape {features.shape}.")
        if len(features) < self.window_size:
            return np.asarray([], dtype=np.float32)

        num_windows = len(features) - self.window_size + 1
        windows = np.lib.stride_tricks.as_strided(
            features,
            shape=(num_windows, self.window_size, features.shape[1]),
            strides=(features.strides[0], features.strides[0], features.strides[1]),
            writeable=False,
        )

        logits = np.empty(
            (num_windows, 2) if self.recovery_head else num_windows, dtype=np.float32
        )
        for start in range(0, num_windows, int(batch_size)):
            end = min(start + int(batch_size), num_windows)
            batch = np.ascontiguousarray(windows[start:end], dtype=np.float32)
            output = np.asarray(self.engine.infer_logits(batch), dtype=np.float32)
            expected = (end - start, 2) if self.recovery_head else (end - start,)
            if output.shape != expected:
                raise ValueError(
                    f"Engine output shape {output.shape} does not match expected {expected} "
                    f"for recovery_head={self.recovery_head}"
                )
            logits[start:end] = output

        if self.model_output_is_probability:
            return logits
        return _sigmoid_array(logits)

    def run_dataframe_batched(
        self,
        df: pd.DataFrame,
        *,
        batch_size: int,
        ecg1_col: str = "ecg1_raw",
        ecg2_col: str = "ecg2_raw",
        puls_col: str | None = "puls_raw",
        time_col: str | None = "time_ms",
        time_unit: str = "ms",
        idx_col: str | None = None,
        sid_col: str = "sid",
        raw_idx_col: str = "raw_idx",
        ecg3_col: str | None = None,
    ):
        """Run a full DataFrame with batched TensorRT inference.

        This keeps causal feature framing and trigger semantics identical to the
        streaming path, but amortizes TensorRT launch/copy overhead across many
        offline windows.
        """
        if int(batch_size) < 1:
            raise ValueError("batch_size must be positive")
        if self.prediction_delay_frames:
            return self.run_dataframe(
                df,
                ecg1_col=ecg1_col,
                ecg2_col=ecg2_col,
                ecg3_col=ecg3_col,
                puls_col=puls_col,
                time_col=time_col,
                time_unit=time_unit,
                idx_col=idx_col,
                sid_col=sid_col,
                raw_idx_col=raw_idx_col,
            )
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        ecg1_col, ecg2_col, ecg3_col, puls_col = require_raw_signal_columns(
            df,
            ecg1_col=ecg1_col,
            ecg2_col=ecg2_col,
            ecg3_col=ecg3_col,
            puls_col=puls_col,
            include_puls=puls_col is not None,
        )

        grouped = (
            df.groupby(sid_col, sort=False) if sid_col in df.columns else [(0, df)]
        )
        all_events = []
        logs: list[pd.DataFrame] = []

        for sid, group in grouped:
            self.reset()

            times_ms = None
            if time_col is not None and time_col in group.columns:
                times_ms = group[time_col].to_numpy(dtype=float)
                if time_unit == "s":
                    times_ms = times_ms * 1000.0
                elif time_unit != "ms":
                    raise ValueError("time_unit must be 'ms' or 's'.")

            puls = None
            if puls_col is not None:
                if puls_col not in group.columns:
                    raise KeyError(f"Missing puls column: {puls_col}")
                puls = group[puls_col].to_numpy(dtype=float)

            source_idx_col = idx_col or raw_idx_col
            if source_idx_col not in group.columns and "idx" in group.columns:
                source_idx_col = "idx"
            sample_idxs = (
                group[source_idx_col].to_numpy(dtype=int)
                if source_idx_col in group.columns
                else np.arange(len(group), dtype=int)
            )

            frames = []
            ecg1 = group[ecg1_col].to_numpy(dtype=float)
            ecg2 = group[ecg2_col].to_numpy(dtype=float)
            ecg3 = None if ecg3_col is None else group[ecg3_col].to_numpy(dtype=float)
            for i in range(len(group)):
                frame = self.framer.process_sample(
                    ecg1=float(ecg1[i]),
                    ecg2=float(ecg2[i]),
                    ecg3=None if ecg3 is None else float(ecg3[i]),
                    puls=None if puls is None else float(puls[i]),
                    sample_idx_400hz=int(sample_idxs[i]),
                    time_ms=None if times_ms is None else float(times_ms[i]),
                )
                if frame is not None:
                    frames.append(frame)

            if not frames:
                continue

            self.last_frame = frames[-1]
            features = np.stack([frame.features for frame in frames], axis=0).astype(
                np.float32,
                copy=False,
            )
            probabilities = self._infer_probabilities_batched(features, batch_size)
            rows = []

            for prob_idx, values in enumerate(probabilities):
                probability = float(values[0]) if self.recovery_head else float(values)
                recovery_probability = float(values[1]) if self.recovery_head else None
                frame = frames[prob_idx + self.window_size - 1]
                event = self.trigger_logic.update(
                    probability=float(probability),
                    current_frame_idx_50hz=frame.frame_idx_50hz,
                    current_sample_idx_400hz=frame.end_sample_idx_400hz,
                    current_time_ms=frame.end_time_ms,
                    sid=sid,
                    recovery_probability=recovery_probability,
                )
                if event is not None:
                    all_events.append(event)

                row = {
                    "sid": sid,
                    "frame_idx": frame.frame_idx_50hz,
                    "raw_idx": frame.start_sample_idx_400hz,
                    "_raw_frame_end_idx": frame.end_sample_idx_400hz,
                    "time_ms": frame.end_time_ms,
                    "prob": float(probability),
                    "trigger_emitted_now": event is not None,
                    "feature_dim": int(len(frame.features)),
                }
                row.update(
                    {
                        col: float(val)
                        for col, val in zip(self.framer.feature_cols, frame.features)
                    }
                )
                if self.recovery_head:
                    row["recovery_prob"] = recovery_probability
                    row["recovery_emitted_now"] = (
                        self.trigger_logic.last_event_was_recovery
                    )
                rows.append(row)

            final_event = self.flush(sid=sid)
            if final_event is not None:
                all_events.append(final_event)
            logs.append(pd.DataFrame(rows))

        if not logs:
            return all_events, pd.DataFrame()
        return all_events, pd.concat(logs, ignore_index=True)
