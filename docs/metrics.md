# Prediction metrics

Run the preferred six-recording benchmark from the project root:

```bash
./metrics.sh normal
./metrics.sh synthetic
```

Both commands evaluate published non-recovery 100 Hz models in `finished_models/`,
using the `delayed_100hz/` variants by default. They use
p02 head, p14 heart/head, and p16 head/heart/pelvis. `./metrics.sh` defaults to
the normal run.

## Model selection

50 Hz models require explicit selection: use `--include-50hz` to include
them in automatic comparisons, or name a particular model with `--model` or
`--add-model`. For example, `./metrics.sh normal --include-50hz` compares
both rates. The `normal` shortcut mode means original (non-synthetic) input;
it now selects 100 Hz models by default.

Recovery versions (including frozen and LOPO recovery) are excluded by default.
Add `--recovery-models` to include them:

```bash
./metrics.sh normal --recovery-models
./metrics.sh synthetic --recovery-models
```

The option also works with `create_prediction_metrics.sh`. It filters model
selection; `--recovery-models --no-recovery` includes recovery models but disables
their recovery events. Selecting or adding a recovery model by name also requires
`--recovery-models`. The older `--exclude-recovery-models` flag remains accepted.

For explicit recordings or a custom output root, use the general wrapper:

```bash
./create_prediction_metrics.sh --output-dir metrics/custom \
  --model ecg3 finished_models data/true/p02/p02_head.parquet \
  data/true/p14/p14_heart.parquet
```

Both shortcut modes use `--only-missing`. Full prediction/event parquets are saved by
default under `metrics/prepped/`; small score/annotation caches under
`metrics/.cache/` also allow reports and plots to be rebuilt without repeating
inference. Synthetic runs do not save modified source recordings by default.

## Concurrent recordings

Use `--jobs N` to evaluate up to N recordings concurrently. Each worker starts
a separate prediction process and uses CUDA when available, sharing the selected
GPU with the other workers. Start with two workers and compare elapsed time;
more workers need more VRAM and host memory and may not improve throughput.
`CUDA_VISIBLE_DEVICES` selects the GPU for all workers. The default is one worker.
Models and synthetic frequencies run sequentially within each recording worker;
pooled dashboards and comparison tables are assembled once after workers finish.

```bash
./metrics.sh normal --jobs 2
./metrics.sh synthetic --jobs 2
```

## Add models incrementally

To add one published model to a completed benchmark, use `--add-model` instead
of `--model`. Repeat it to add several models. Only the selected models are
evaluated; the comparison tables retain the other models and the existing Siemens
baseline. Adding an existing model name replaces its rows without duplicating it.

```bash
./metrics.sh normal --add-model slope_ecg3_delayed_100hz --jobs 2
```

Use the same input recordings and their size/mtime signatures, variant,
prediction overrides, synthetic frequency set, and
`--include-outer-position-errors` setting as the original benchmark.
Benchmark settings and summary counts are saved under `.cache/` for subsequent
additions. Existing normal CSV tables can be imported automatically with the
default prediction settings. Older synthetic tables require a full run once to
record the noise settings, preventing a mixture of old and new noise defaults.
`--model` retains its existing behavior: it writes a comparison containing only
the selected models. These options also work with `create_prediction_metrics.sh`.

## Saved predictions and cache reuse

Full prediction/event parquets are kept by default (`--save-prepped`). Normal
outputs go under `metrics/prepped/<model>/`; synthetic outputs go under
`metrics/prepped/synthetic/main_<frequency>hz/<model>/`. Matching saved files
are reused on subsequent runs. Changing input data, model weights, or prediction
settings invalidates the cache; changing only this saving flag does not.
Prediction caches track runtime settings and inference/native code separately
from training settings and calibration audit metadata. Plot-only changes rebuild
plots while reusing predictions. A newly selected recovery threshold still
invalidates that model's predictions and reports. Legacy unmarked single-output
prediction caches are accepted for clean recordings; recovery models require a
matching key. Failed prediction or plot refreshes are retried on subsequent runs.

If only a compact cache exists, requesting full outputs repeats inference once
to produce the full files; subsequent matching runs reuse them.
Use `--no-save-prepped` to make full files temporary while retaining compact caches.
All `prepped/` directories are ignored by Git, including those under custom
output roots; regenerate predictions locally after a fresh checkout.
With `--no-recovery`, saved paths gain a `current_only/` component, for example
`metrics/prepped/current_only/<model>/`.

## Reports and metric policy

Reports go under `metrics/` for normal runs and `metrics/synthetic/` for synthetic
runs. Each has `prediction_metrics_overview.{txt,csv}` for pooled results and
`prediction_metrics_per_input.{txt,csv}` for per-input comparisons. Synthetic
reports group results by gradient frequency.
With `--no-recovery`, reports go under `metrics/current_only/`, including its
`synthetic/` subdirectory for synthetic runs.

Individual prediction reports and dashboards retain outer position errors.
Comparison tables count each outer match as one FP plus one FN by default;
use `--include-outer-position-errors` to retain that tier in the tables.
These differences are intentional. The
[50/100 Hz event policy](100hz-delayed-target.md#tp-pe-and-ope-definitions-option-2)
defines the model-specific frame radii. Use the
[raw-time comparison](100hz-delayed-target.md#compare-saved-models-on-raw-time)
for cross-rate comparisons over common valid recording intervals.

## Synthetic gradient benchmarks

The sweep uses main frequencies 190, 90, 80, 40, 20, and 10 Hz. Add
`--exclude-low-frequencies` to omit 20 and 10 Hz. Use `--save-synthetic` to
also save modified inputs under `data/synthetic/`; raw recordings remain intact.

Synthetic metrics default to **0.6 noise amplitude**, scaling every
added component together. To restore full amplitude, use a separate report
directory and an explicit override:

```bash
./metrics.sh synthetic --output-dir metrics/noise_full -- --synthetic-noise-multiplier 1
```

To reproduce the previous `0.5` strength, use
`-- --synthetic-noise-multiplier 0.5` with its own output directory. Separate
roots keep comparisons clear; incremental additions reject incompatible noise
settings rather than mixing their results.

Both `metrics.sh synthetic` and `create_prediction_metrics.sh --synthetic-gradient`
use `0.6` unless overridden. Standalone `offline_predict.py` and the notebook
helper retain their multiplier of `1`. Levels such as `0.25`, `0.5`, `0.6`, `0.75`, and `1`
provide an amplitude sweep without changing raw ECGs,
annotations, or trained thresholds. This is an amplitude control, not automatic
SNR normalization. Scaled standalone prediction filenames and saved synthetic
inputs include `noise_x<multiplier>` to keep levels separate. Metrics wrappers
use the explicitly selected output directory.

Synthetic benchmarks couple the gradient peak phases across ECG1 and ECG2 by
default, with ECG2 at `-30` degrees relative to ECG1. This approximates the
measured head-artifact phase relationship while preserving each lead's complete
noise power spectrum. ECG3 remains ECG2 minus ECG1; its noise therefore changes
with the lead relationship. Noise outside the configured peak bands keeps its
independent phase. The phase setting is an empirical benchmark setting, not a
universal calibration for every position or frequency.

Override the offset with `-- --synthetic-lead-phase-deg -25`, or reproduce the
historical independent draws with `-- --synthetic-independent-leads`. Use a
separate output directory when comparing modes. Standalone predictions and
saved synthetic inputs include the phase setting in their paths. The notebook
helper retains independent leads by default; pass `lead_phase_degrees=-30` to
`apply_synthetic_gradient` to reproduce the coupled benchmark.
For the original full-strength independent benchmark, combine
`--synthetic-independent-leads` with `--synthetic-noise-multiplier 1` after `--`.

Append options from `create_prediction_metrics.sh` to either command, for example
`./metrics.sh normal --recovery-models --model ecg3 --model ecg3_recovery`. Use `--dry-run` to
print the expanded command, or `--help` for the shortcut options.

Training-result overviews (`./create_train_metrics.sh models`) use the same
100 Hz default and `--include-50hz` / `--recovery-models` opt-ins.
An explicit `--model` selects a named 50 Hz training run too.
