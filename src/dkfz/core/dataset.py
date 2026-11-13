# Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Developed at the German Cancer Research Center (DKFZ).

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


def _frame_index_col(df, preferred="frame_idx"):
    if preferred in df.columns:
        return preferred
    if preferred == "frame_idx" and "idx" in df.columns:
        return "idx"
    return preferred


def add_live_sequence_id(df, sid_col="sid", frame_idx_col="frame_idx", idx_col=None):
    """Add or normalize the live sequence id.

    Live frame tables use `frame_idx` as the 50 Hz frame index within a
    sequence. Older tables with `idx` are accepted as a fallback. The resulting
    `sid` is a zero-based integer sequence id.
    """
    frame_idx_col = _frame_index_col(df, idx_col or frame_idx_col)
    if frame_idx_col not in df.columns:
        if sid_col in df.columns:
            df[sid_col] = pd.factorize(df[sid_col], sort=False)[0]
        else:
            df[sid_col] = 0
        return df

    diffs = df[frame_idx_col].diff().fillna(1)
    is_new_sequence = diffs <= 0
    if sid_col in df.columns:
        is_new_sequence |= df[sid_col].ne(df[sid_col].shift()).fillna(False)

    if len(df) > 0:
        is_new_sequence.iloc[0] = False
    df[sid_col] = is_new_sequence.cumsum()
    return df


class RRPhaseTracker:
    """Stateful causal RR-phase feature for online/live inference.

    The first R peak initializes the beat clock. The second R peak initializes
    the RR estimate. From then on, each call returns
    ``(frame_idx - last_rpeak_frame_idx) / rr_hat`` clipped to ``clip_max``.
    """

    def __init__(self, alpha=0.2, clip_max=2.0):
        if not 0.0 < float(alpha) <= 1.0:
            raise ValueError("alpha must be in (0, 1].")
        if clip_max is not None and float(clip_max) <= 0.0:
            raise ValueError("clip_max must be positive or None.")

        self.alpha = float(alpha)
        self.clip_max = None if clip_max is None else float(clip_max)
        self.reset()

    def reset(self):
        """Clear the stored R-peak state and RR estimate."""
        self.last_r_idx = None
        self.rr_hat = None

    def step(self, frame_idx, is_rpeak):
        """Update the tracker with one frame and return the current RR phase."""
        idx_now = float(frame_idx)

        if bool(is_rpeak):
            if self.last_r_idx is not None:
                rr = idx_now - self.last_r_idx
                if rr <= 0.0:
                    raise ValueError(
                        "R-peak frame indices must increase within a stream."
                    )
                if self.rr_hat is None:
                    self.rr_hat = rr
                else:
                    self.rr_hat = (1.0 - self.alpha) * self.rr_hat + self.alpha * rr
            self.last_r_idx = idx_now

        if self.last_r_idx is None or self.rr_hat is None:
            return np.nan

        phi = (idx_now - self.last_r_idx) / self.rr_hat
        if self.clip_max is not None:
            phi = min(phi, self.clip_max)
        return float(phi)


def add_rr_phase(
    df,
    idx="frame_idx",
    rpeak_col="target",
    out_col="rr_phase",
    alpha=0.2,
    clip_max=2.0,
    mode: str | None = None,
    sid_col="sid",
):
    """
    Add a causal RR-phase feature column to a dataframe.

    This is a batch adapter over :class:`RRPhaseTracker`, so precomputed training
    features follow the same state-machine logic that a realtime application can
    use frame by frame.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe, sorted in time order.
    idx : str
        Name of the time column.
    rpeak_col : str
        Name of the column marking detected R peaks (bool, 0/1).
    out_col : str
        Name of the output feature column.
    alpha : float
        EWMA update factor for RR_hat.
    clip_max : float or None
        Optional upper clipping value for the feature.
    mode : str or None
        Deprecated compatibility argument. All modes now use the exact online
        state machine.
    sid_col : str or None
        Optional sequence id column. When present, RR state resets at sequence
        boundaries.

    Returns
    -------
    pandas.DataFrame
        Copy of df with a new column `out_col`.
    """
    out = df.copy()
    idx = _frame_index_col(out, idx)

    tracker = RRPhaseTracker(alpha=alpha, clip_max=clip_max)
    feature = np.full(len(out), np.nan, dtype=float)
    use_sid = sid_col is not None and sid_col in out.columns
    idxs = out[idx].to_numpy()
    rpeaks = out[rpeak_col].astype(bool).to_numpy()
    sids = out[sid_col].to_numpy() if use_sid else None

    for i in range(len(out)):
        if use_sid and i > 0 and sids[i] != sids[i - 1]:
            tracker.reset()
        feature[i] = tracker.step(idxs[i], rpeaks[i])

    out[out_col] = feature
    return out


class SlidingWindowDataset(Dataset):
    def __init__(
        self,
        df,
        feature_cols,
        target_col,
        window_size=25,
        recovery_head=False,
        prediction_delay_frames=0,
    ):
        """Sliding Window Dataset for Torch.

        Args:
            df (Pandas): Dataframe
            feature_cols (list(str)): Features to train NN.
            target_col (str): Target feature.
            window_size (int, optional): Sliding Window Size. Defaults to 25.
        """
        if prediction_delay_frames not in (0, 1) or isinstance(
            prediction_delay_frames, bool
        ):
            raise ValueError("prediction_delay_frames must be 0 or 1")
        if prediction_delay_frames and recovery_head:
            raise ValueError("A delayed primary cannot be combined with recovery_head")
        self.prediction_delay_frames = int(prediction_delay_frames)
        if int(window_size) < (2 if recovery_head or prediction_delay_frames else 1):
            raise ValueError(
                "window_size must be at least 2 with recovery_head, otherwise 1"
            )
        self.recovery_head = bool(recovery_head)
        df = df.copy()
        if prediction_delay_frames and "frame_idx" in df:
            changed = df["frame_idx"].diff().ne(1)
            if "sid" in df:
                changed |= df.sid.ne(df.sid.shift())
            df["sid"] = changed.cumsum()
        df = add_live_sequence_id(df)
        self.X = torch.tensor(df[feature_cols].values, dtype=torch.float32)
        self.y = torch.tensor(df[target_col].values, dtype=torch.float32)
        self.seq_id = torch.tensor(df["sid"].values, dtype=torch.long)
        self.seq = df["sid"].to_numpy()
        # Python slices exclude the stop index, so i:i+25 covers frames i..i+24.
        # The target is the last frame inside the window to match live timing.
        self.window_size = window_size
        self.data = df

        # self.n_samples = len(df) - window_size + 1

        self.feature_cols = feature_cols
        self.target_col = target_col

        self.starts = []
        n = len(df)
        w = window_size
        for i in range(n - w + 1):
            # Window covers [i .. i+w-1], and target is the last window frame.
            if self.seq[i] == self.seq[i + w - 1]:
                self.starts.append(i)

    def __len__(self):
        # number of valid windows len(self.starts)-1
        return len(self.starts)  # - 1

    def __getitem__(self, idx):
        """
        Returns:
            x : (window_size, num_features)
            y : scalar (or (1,))
        """
        i = self.starts[idx]
        x = self.X[i : i + self.window_size]
        y = self.y[i + self.window_size - 1 - self.prediction_delay_frames]
        if self.recovery_head:
            y = torch.stack((y, self.y[i + self.window_size - 2]))
        seq_id = self.seq_id[i]  # : i + self.window_size + 1]
        return (x, y, seq_id)

    def __getdataframe__(self):
        return self.data
