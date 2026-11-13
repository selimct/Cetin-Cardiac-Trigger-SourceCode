#!/usr/bin/env bash
# Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Developed at the German Cancer Research Center (DKFZ).

# Recompute thresholds from validation scores without changing trained weights.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$project_root"

case "${1:-}" in
    recovery|calibrate-recovery)
        shift
        exec "$project_root/.venv/bin/python" run/calibrate_recovery.py "$@"
        ;;
    --help|-h)
        cat <<'USAGE'
Usage:
  ./update_model_thresholds.sh [roots ...] [options]
  ./update_model_thresholds.sh recovery --config CONFIG [--validation-log LOG] [options]

Single-output models: backfill named FP/DER thresholds from validation data.
Two-output models: calibrate recovery while preserving the primary threshold.
Mixed roots are handled automatically. Recovery calibration reads saved training
or same-epoch inner-validation logs. No benchmark/outer-test fallback is used.
Use annotated validation logs containing prob, recovery_prob, target, and sid.
Never use the prediction-metrics test recordings for calibration.

Recovery options:
  --config PATH             Two-output training config.
  --validation-log PATH     Validation prediction parquet; repeatable. Optional
                            when saved validation artifacts can be discovered.
  --max-fp-increase VALUE   Relative FP allowance; defaults to config or 0.10.
  --rebuild-cv-validation   For older recovery CV runs, replay saved inner
                            checkpoints at the deployment epoch without retraining.
  --output PATH             Threshold-search CSV; defaults beside config.
  --apply                   Write the selected recovery policy to config.
Without --apply, both commands only write their audit reports.

Mixed-collection options (including --recovery-max-fp-increase):
USAGE
        exec "$project_root/.venv/bin/python" run/post_backfill_thresholds.py --help
        ;;
esac
exec "$project_root/.venv/bin/python" run/post_backfill_thresholds.py "$@"
