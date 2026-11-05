# Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Developed at the German Cancer Research Center (DKFZ).

"""Python adapters for compiled DKFZ backends.

The public pipeline API remains under :mod:`dkfz.live`.  This package is the
explicit boundary between Python orchestration and reusable native code.
"""

from __future__ import annotations


def is_available() -> bool:
    """Return whether the compiled live core can be imported."""
    try:
        from dkfz.native import _live_core  # noqa: F401
    except ImportError:
        return False
    return True


__all__ = ["is_available"]
