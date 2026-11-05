# Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Developed at the German Cancer Research Center (DKFZ).

"""Shared duration and footer formatting for pooled metrics overviews."""

from __future__ import annotations

import math
from typing import Any, Iterable


def duration_minutes(timepoints: int, config: dict[str, Any]) -> float:
    """Convert evaluated frames using the configured rate (default: 50 Hz)."""
    raw_hz = float(config.get("raw_hz", 400.0))
    frame_size = float(config.get("frame_size_raw", 8))
    if not all(math.isfinite(value) and value > 0 for value in (raw_hz, frame_size)):
        raise ValueError("raw_hz and frame_size_raw must be positive and finite.")
    return timepoints * frame_size / raw_hz / 60.0


def format_totals(rows: Iterable[dict[str, Any]], *, scope: str) -> str:
    """Format totals once per system, without counting shared data across models."""
    lines = [f"Totals over {scope} (per system)"]
    for row in rows:
        minutes = row["duration_minutes"]
        duration = (
            "unavailable (missing fold metrics)"
            if minutes is None
            else f"{minutes:.2f}"
        )
        lines.extend(
            (
                "",
                str(row["system"]),
                f"Total annotated R peaks: {row['num_target_events']}",
                f"Total predicted R peaks: {row['num_predicted_events']}",
                f"Total minutes of data used: {duration}",
            )
        )
    return "\n".join(lines)
