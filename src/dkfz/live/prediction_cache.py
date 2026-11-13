# Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Developed at the German Cancer Research Center (DKFZ).

"""Small report caches independent of optional full prediction parquets."""

from pathlib import Path

import pandas as pd


def read_prediction_frame(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    return pd.read_feather(path) if path.suffix == ".arrow" else pd.read_parquet(path)


def cache_prediction_frame(
    source: Path, destination: Path, target_col="target"
) -> None:
    """Retain exact scores, annotations and timing, omitting model feature windows."""
    frame = read_prediction_frame(source)
    columns = {
        "prediction_delay_frames",
        "raw_frame_end_idx",
        "observed_frame_idx",
        "observed_raw_idx",
        "observed_raw_frame_end_idx",
        "observed_time_ms",
        "selected_peak_raw_idx",
        "emitted_on_raw_idx",
        "emitted_on_time_ms",
        "sid",
        "frame_idx",
        "frame_idx_50hz",
        "idx",
        "raw_idx",
        "time_ms",
        "prob",
        "recovery_prob",
        "pred",
        target_col,
        "siemens_tr",
        "puls_peak",
        "trigger_emitted_now",
        "recovery_emitted_now",
    }
    cached = frame.loc[:, [name for name in frame.columns if name in columns]]
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".arrow.tmp")
    try:
        cached.reset_index(drop=True).to_feather(temporary, compression="zstd")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
