#!/usr/bin/env bash
# Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Developed at the German Cancer Research Center (DKFZ).

set -euo pipefail

# Reports automatically include recovery-head settings and calibration metadata
# from each run's config, alongside the held-out outer-fold detection metrics.
# All CV methods, including nested_recovery_lopo, are included by default;
# --cv-method selects a particular workflow.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${CREATE_TRAIN_METRICS_PYTHON:-}"

if [[ -z "$PYTHON_BIN" ]]; then
    if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
        PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
    elif [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
        PYTHON_BIN="$VIRTUAL_ENV/bin/python"
    elif command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="$(command -v python3)"
    elif command -v python >/dev/null 2>&1; then
        PYTHON_BIN="$(command -v python)"
    else
        printf 'create_train_metrics.sh: no Python interpreter found\n' >&2
        exit 2
    fi
elif [[ "$PYTHON_BIN" != */* ]]; then
    if ! PYTHON_BIN="$(command -v "$PYTHON_BIN")"; then
        printf 'create_train_metrics.sh: Python interpreter not found: %s\n' \
            "${CREATE_TRAIN_METRICS_PYTHON}" >&2
        exit 2
    fi
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
    printf 'create_train_metrics.sh: Python interpreter is not executable: %s\n' \
        "$PYTHON_BIN" >&2
    exit 2
fi

exec "$PYTHON_BIN" run/create_train_metrics.py "$@"
