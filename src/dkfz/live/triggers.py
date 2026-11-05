# Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Developed at the German Cancer Research Center (DKFZ).

"""Stable Python API for native live-trigger behavior and trigger data tools."""

from __future__ import annotations

from dkfz.native.triggers import (
    SiemensTriggerCleanStats,
    StreamingHoldoffLocalMaxTrigger,
    TriggerEvent,
    clean_siemens_trigger_pairs,
)

TRIGGER_BACKEND = "native"


def trigger_backend_name() -> str:
    """Return the sole supported streaming-trigger backend name."""
    return TRIGGER_BACKEND


__all__ = [
    "SiemensTriggerCleanStats",
    "StreamingHoldoffLocalMaxTrigger",
    "TRIGGER_BACKEND",
    "TriggerEvent",
    "clean_siemens_trigger_pairs",
    "trigger_backend_name",
]
