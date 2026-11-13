# Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Developed at the German Cancer Research Center (DKFZ).

"""Backend-neutral streaming predictor for live ECG inference."""

from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd

from dkfz.live.feature_framing import LiveFeatureFrame, LiveFeatureFramer
from dkfz.live.io import require_raw_signal_columns
from dkfz.live.triggers import StreamingHoldoffLocalMaxTrigger, TriggerEvent


class LiveStreamingPredictor:
    """
    Shared live ECG predictor pipeline.

    This class owns the backend-independent parts of live inference: causal
    feature framing, sliding model windows, trigger logic, and log/event
    adapters. Subclasses only need to implement `_run_model_on_current_window`.
    """

    def __init__(
        self,
        *,
        window_size: int,
        threshold: float,
        holdoff_frames: int,
        recovery_head: bool = False,
        recovery_threshold: float | None = None,
        prediction_delay_frames: int = 0,
        framer: LiveFeatureFramer | None = None,
        **framer_kwargs,
    ):
        self.prediction_delay_frames = int(prediction_delay_frames)
        if prediction_delay_frames not in (0, 1) or isinstance(
            prediction_delay_frames, bool
        ):
            raise ValueError("prediction_delay_frames must be 0 or 1")
        if prediction_delay_frames and (recovery_head or int(window_size) < 2):
            raise ValueError(
                "A delayed primary needs window_size >= 2 and no recovery head"
            )
        self.window_size = int(window_size)
        self.threshold = float(threshold)
        self.holdoff_frames = int(holdoff_frames)
        self.recovery_head = bool(recovery_head)
        if self.recovery_head and self.window_size < 2:
            raise ValueError("recovery_head requires window_size >= 2")
        self.recovery_threshold = recovery_threshold
        if recovery_threshold is not None and not self.recovery_head:
            raise ValueError("recovery_threshold requires recovery_head=True")
        self.framer = framer or LiveFeatureFramer(**framer_kwargs)
        if self.prediction_delay_frames and (
            self.framer.raw_hz != 400 or self.framer.frame_size_raw != 4
        ):
            raise ValueError("A delayed primary requires 400 Hz and four-sample frames")
        self.trigger_logic = StreamingHoldoffLocalMaxTrigger(
            threshold=self.threshold,
            holdoff_frames=self.holdoff_frames,
            recovery_threshold=recovery_threshold,
        )
        self.reset()

    @staticmethod
    def holdoff_frames_from_ms(holdoff_ms: float, frame_hz: float = 50.0) -> int:
        """Convert a holdoff duration in milliseconds to frame count."""
        return int(np.ceil(float(holdoff_ms) * float(frame_hz) / 1000.0))

    def reset(self) -> None:
        """Reset framer, trigger logic, and sliding model window."""
        self.framer.reset()
        self.trigger_logic.reset()
        self._feature_window: deque[np.ndarray] = deque(maxlen=self.window_size)
        self.last_probability: float | None = None
        self.last_recovery_probability: float | None = None
        self.last_probability_updated = False
        self.last_frame: LiveFeatureFrame | None = None
        self.last_input_frame: LiveFeatureFrame | None = None
        self._target_frames = deque(maxlen=self.prediction_delay_frames + 1)
        self._previous_raw_idx = None
        self._previous_sid = None
        self._next_raw_idx = 0
        self._last_raw_time = None

    def _run_model_on_current_window(self) -> float:
        """Return the probability for the current feature window."""
        raise NotImplementedError

    def process_sample(
        self,
        ecg1: float,
        ecg2: float,
        puls: float | None = None,
        sample_idx_400hz: int | None = None,
        time_ms: float | None = None,
        sid: int | str = 0,
        ecg3: float | None = None,
    ) -> TriggerEvent | None:
        """Process one raw sample through framing, inference, and triggering."""
        if self.prediction_delay_frames:
            if self._previous_sid is not None and sid != self._previous_sid:
                self.reset()
            sample_idx_400hz = (
                self._next_raw_idx
                if sample_idx_400hz is None
                else int(sample_idx_400hz)
            )
            if self._previous_sid is not None and (
                sid != self._previous_sid
                or sample_idx_400hz != self._previous_raw_idx + 1
            ):
                self.reset()
            self._previous_raw_idx = sample_idx_400hz
            self._previous_sid = sid
            self._next_raw_idx = sample_idx_400hz + 1
            self._last_raw_time = (
                time_ms
                if time_ms is not None
                else sample_idx_400hz / self.framer.raw_hz * 1000
            )
        self.last_probability_updated = False
        frame = self.framer.process_sample(
            ecg1=ecg1,
            ecg2=ecg2,
            ecg3=ecg3,
            puls=puls,
            sample_idx_400hz=sample_idx_400hz,
            time_ms=time_ms,
        )
        if frame is None:
            return None

        self.last_input_frame = frame
        self._target_frames.append(frame)
        self._feature_window.append(frame.features)
        self.last_frame = self._target_frames[0]
        if len(self._feature_window) < self.window_size:
            self.last_probability = None
            return None

        probability = self._run_model_on_current_window()
        self.last_probability = probability
        self.last_probability_updated = True
        return self.trigger_logic.update(
            probability=probability,
            current_frame_idx_50hz=frame.frame_idx_50hz,
            current_sample_idx_400hz=frame.end_sample_idx_400hz,
            current_time_ms=frame.end_time_ms,
            sid=sid,
            recovery_probability=self.last_recovery_probability,
            **(
                dict(
                    target_frame_idx=self.last_frame.frame_idx_50hz,
                    target_sample_idx=self.last_frame.end_sample_idx_400hz,
                    target_time_ms=self.last_frame.end_time_ms,
                )
                if self.prediction_delay_frames
                else {}
            ),
        )

    def flush(self, sid: int | str = 0) -> TriggerEvent | None:
        """Emit any still-open candidate when the input stream ends."""
        if self.last_frame is None:
            return None
        return self.trigger_logic.flush(
            current_frame_idx_50hz=(
                self.last_input_frame or self.last_frame
            ).frame_idx_50hz,
            current_sample_idx_400hz=(
                self._previous_raw_idx
                if self.prediction_delay_frames
                else self.last_frame.end_sample_idx_400hz
            ),
            current_time_ms=(
                self._last_raw_time
                if self.prediction_delay_frames
                else self.last_frame.end_time_ms
            ),
            sid=sid,
        )

    def run_arrays(
        self,
        ecg1: np.ndarray,
        ecg2: np.ndarray,
        puls: np.ndarray | None = None,
        times_ms: np.ndarray | None = None,
        sample_idxs: np.ndarray | None = None,
        sid: int | str = 0,
        ecg3: np.ndarray | None = None,
    ) -> tuple[list[TriggerEvent], pd.DataFrame]:
        """Run a full array through the live predictor sample by sample."""
        self.reset()
        return self.process_chunk_arrays(
            ecg1=ecg1,
            ecg2=ecg2,
            ecg3=ecg3,
            puls=puls,
            times_ms=times_ms,
            sample_idxs=sample_idxs,
            sid=sid,
            flush=True,
        )

    def process_chunk_arrays(
        self,
        ecg1: np.ndarray,
        ecg2: np.ndarray,
        puls: np.ndarray | None = None,
        times_ms: np.ndarray | None = None,
        sample_idxs: np.ndarray | None = None,
        sid: int | str = 0,
        flush: bool = False,
        ecg3: np.ndarray | None = None,
    ) -> tuple[list[TriggerEvent], pd.DataFrame]:
        """Process one chunk without resetting predictor state."""
        ecg1 = np.asarray(ecg1)
        ecg2 = np.asarray(ecg2)
        if ecg1.shape != ecg2.shape:
            raise ValueError(f"ecg1/ecg2 shapes differ: {ecg1.shape}, {ecg2.shape}.")

        if ecg3 is not None:
            ecg3 = np.asarray(ecg3)
            if ecg3.shape != ecg1.shape:
                raise ValueError(f"ecg3 shape differs from ECG shape: {ecg3.shape}.")

        if puls is not None:
            puls = np.asarray(puls)
            if puls.shape != ecg1.shape:
                raise ValueError(f"puls shape differs from ECG shape: {puls.shape}.")

        if times_ms is not None:
            times_ms = np.asarray(times_ms)
            if times_ms.shape != ecg1.shape:
                raise ValueError("times_ms must have the same shape as ecg1.")

        if sample_idxs is not None:
            sample_idxs = np.asarray(sample_idxs)
            if sample_idxs.shape != ecg1.shape:
                raise ValueError("sample_idxs must have the same shape as ecg1.")

        events: list[TriggerEvent] = []
        logs: list[dict] = []

        for i in range(len(ecg1)):
            time_i_ms = None if times_ms is None else float(times_ms[i])
            puls_i = None if puls is None else float(puls[i])
            sample_idx = (
                (None if self.prediction_delay_frames else i)
                if sample_idxs is None
                else int(sample_idxs[i])
            )
            event = self.process_sample(
                ecg1=float(ecg1[i]),
                ecg2=float(ecg2[i]),
                ecg3=None if ecg3 is None else float(ecg3[i]),
                puls=puls_i,
                sample_idx_400hz=sample_idx,
                time_ms=time_i_ms,
                sid=sid,
            )

            if (
                self.last_probability_updated
                and self.last_probability is not None
                and self.last_frame is not None
            ):
                row = {
                    "sid": sid,
                    "frame_idx": self.last_frame.frame_idx_50hz,
                    "raw_idx": self.last_frame.start_sample_idx_400hz,
                    "_raw_frame_end_idx": self.last_frame.end_sample_idx_400hz,
                    "time_ms": self.last_frame.end_time_ms,
                    "prob": self.last_probability,
                    "trigger_emitted_now": event is not None,
                    "feature_dim": int(len(self.last_frame.features)),
                }
                if self.prediction_delay_frames:
                    row.update(
                        prediction_delay_frames=self.prediction_delay_frames,
                        observed_frame_idx=self.last_input_frame.frame_idx_50hz,
                        observed_raw_idx=self.last_input_frame.start_sample_idx_400hz,
                        observed_raw_frame_end_idx=self.last_input_frame.end_sample_idx_400hz,
                        observed_time_ms=self.last_input_frame.end_time_ms,
                        raw_frame_end_idx=self.last_frame.end_sample_idx_400hz,
                        selected_peak_raw_idx=(
                            None if event is None else event.peak_sample_idx_400hz
                        ),
                        emitted_on_raw_idx=(
                            None if event is None else event.emitted_on_sample_idx_400hz
                        ),
                        emitted_on_time_ms=(
                            None if event is None else event.emitted_on_time_ms
                        ),
                    )
                row.update(
                    {
                        col: float(val)
                        for col, val in zip(
                            self.framer.feature_cols,
                            self.last_frame.features,
                        )
                    }
                )
                if self.recovery_head:
                    row["recovery_prob"] = self.last_recovery_probability
                    row["recovery_emitted_now"] = (
                        self.trigger_logic.last_event_was_recovery
                    )
                logs.append(row)

            if event is not None:
                events.append(event)

        final_event = self.flush(sid=sid) if flush else None
        if final_event is not None:
            events.append(final_event)

        return events, pd.DataFrame(logs)

    def run_dataframe(
        self,
        df: pd.DataFrame,
        ecg1_col: str = "ecg1_raw",
        ecg2_col: str = "ecg2_raw",
        puls_col: str | None = "puls_raw",
        time_col: str | None = "time_ms",
        time_unit: str = "ms",
        idx_col: str | None = None,
        sid_col: str = "sid",
        raw_idx_col: str = "raw_idx",
        ecg3_col: str | None = None,
    ) -> tuple[list[TriggerEvent], pd.DataFrame]:
        """Run a DataFrame through the live predictor."""
        ecg1_col, ecg2_col, ecg3_col, puls_col = require_raw_signal_columns(
            df,
            ecg1_col=ecg1_col,
            ecg2_col=ecg2_col,
            ecg3_col=ecg3_col,
            puls_col=puls_col,
            include_puls=puls_col is not None,
        )

        if sid_col in df.columns:
            grouped = df.groupby(sid_col, sort=False)
        else:
            grouped = [(0, df)]

        all_events: list[TriggerEvent] = []
        logs: list[pd.DataFrame] = []

        for sid, group in grouped:
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

            sample_idxs = None
            source_idx_col = idx_col or raw_idx_col
            if source_idx_col not in group.columns and "idx" in group.columns:
                source_idx_col = "idx"
            if source_idx_col in group.columns:
                sample_idxs = group[source_idx_col].to_numpy(dtype=int)

            events, log_df = self.run_arrays(
                ecg1=group[ecg1_col].to_numpy(dtype=float),
                ecg2=group[ecg2_col].to_numpy(dtype=float),
                ecg3=(
                    None if ecg3_col is None else group[ecg3_col].to_numpy(dtype=float)
                ),
                puls=puls,
                times_ms=times_ms,
                sample_idxs=sample_idxs,
                sid=sid,
            )
            all_events.extend(events)
            logs.append(log_df)

        if not logs:
            return all_events, pd.DataFrame()
        return all_events, pd.concat(logs, ignore_index=True)
