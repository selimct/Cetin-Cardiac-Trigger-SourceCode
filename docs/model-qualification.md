# Qualify a finished delayed 100 Hz model on this Jetson

Run the single entrypoint from the project root as the ordinary user:

```bash
bash run/qualify_model.sh \
  --model-dir finished_models/ecg3/delayed_100hz \
  --parity-input data/live/p05/p05_out.parquet \
  --test-input data/live/p12/p12_out.parquet \
  --uart-recording data/pmu/raw/uart/p13/p13_out.bin \
  --ecg3-source ecg2_minus_ecg1
```

Replace the example model/recording paths with your selected finished model,
validation recording(s), and genuinely held-out annotated test recording(s).
Repeat `--parity-input` and `--test-input` for more coverage. The baseline defaults
to `finished_models/ecg3`; select another with `--baseline-dir`. Best/tuned source
variants have separate `--variant` and `--baseline-variant` options. The selected
config supplies the model identity automatically.

The source model must be the separately trained four-sample delayed model, with
`best.pth` plus matching `train_config.json` (or the corresponding tuned pair).
Finish transferring it before running the script. CUDA checkpoint tensors are
loaded on CPU for export/reference inference; training need not be restarted.
The model must use the confirmed ECG2 − ECG1 lead definition.

First inspect the plan by adding `--dry-run`. It reads model metadata but performs
no GPU work, authentication, service operation or output-directory creation.
The default output is a new `artifact/qualification-<timestamp>-<id>/`; an explicit
`--output-dir` must also be a new directory under this project's `artifact/`.
Run `bash run/qualify_model.sh --help` for all options.

## What runs

1. **Implementation and engine parity.** Rebuild the existing native build only
   as needed. Development regression suites are maintained upstream and
   are not run by this distribution; the report records them as
   `development_regressions.status=not_bundled`. Copy
   the exact model into a new isolated bundle, export ONNX and build TensorRT on
   this Jetson. Compare Torch/ONNX/TRT on causal recording-derived feature windows,
   then Torch/native probabilities, event positions and emission coordinates over
   the parity recordings. An additional real-signal fixture exercises an index
   gap, sequence change, independent warmups and partial EOF. FP16 that changes
   numerical/event results is rejected; automatic precision retries strict FP32.
   Positive trigger coverage is required. No synthetic trained model is used.

2. **Fixed-weight accuracy comparison.** Run the finished 50 Hz baseline and new
   candidate on the same held-out inputs and common valid interval after warmup.
   Temporary input copies derive ECG3 without unsigned subtraction wraparound;
   original signals/annotations remain untouched. Weights and saved thresholds
   are unchanged. The controlled comparison uses a common physical holdoff based
   on the candidate's saved duration, rounded up to a representable 20 ms grid.
   If rounding would change the candidate, a separate saved-policy evaluation
   supplies its acceptance metrics. No source config is edited. Native early
   emission policy is taken from the production runtime config. Reports include
   recovered/new misses, added/removed false positives, timing error, sample-axis
   emission latency, boundary positions and premature-trigger/holdoff cases.
   Schema-4 frame categories (100 Hz 0–1/2–3/4–5; 50 Hz 0/1/2) remain separate
   from the shared ±20 ms raw-time diagnostic.

3. **Complete service scheduling/sandbox checks.** By default run two paced
   612-second tests, then a 60-second live UART check with **GPIO disabled**.
   Each uses a uniquely named transient `dkfz-qualification-...` service,
   production User/Group/supplementary groups, CPU/IO weights, nice priority,
   timer slack, bounded RT limits, watchdog and filesystem protection. Actual
   manager properties, process identity and verified RR/20 worker states are
   checked; requested flags alone are insufficient. The checkout is read-only
   in the service. A read-only Python-environment bind supports the test driver.
   The driver gives readiness/watchdog ownership to the native child explicitly.

The temporary unit uses [systemd-run](https://www.freedesktop.org/software/systemd/man/249/systemd-run.html) and substitutes private writable `/run` state, diagnostic JSONL,
no automatic restart and a private PTY for paced input. The live check uses the
UART device from `config/native_uart_live.conf`; connect valid live input before
running it. Raw capture and external threshold polling are disabled in private
config copies. The production detector must already be idle; the script refuses
active or unverified detector/process activity and never stops that detector.
It does not install/reload/start the production service, activate a deployment,
change power/clocks, retrain, change annotations, or tune thresholds. Your
physical GPIO timing measurement must be performed separately.

Timing reports include p50/p95/p99/max processing and host-decode-to-completion,
CPU/GPU stages, backlog, deadlines, sample counts, valid-input/reset/loss counters,
positive output coverage, temperatures and unchanged power mode. Native deadline
limits start at host decode; source-pacer delays are reported separately. This
is not a proof of a universal worst-case bound. Power watts may be unavailable.

## Authentication and duration

The script always uses `/home/orin/envs/jet/bin/python`. Do **not** run the entire
script under sudo. It prompts for sudo authentication for the private service
jobs and cleanup of their exact, generated `/run` directories; another prompt
may occur when a long phase outlasts the credential cache. No password is stored
or passed as a command argument. Export/build and offline comparison run as you;
service tests run as the production service account.

Allow at least 21–22 minutes for the default timing checks, plus engine building,
full-recording CPU accuracy inference. `--timing-runs 1` runs one
sustained test; `--seconds` cannot be below 600. `--live-uart-seconds 0` explicitly
skips the physical UART access/liveness check and leaves qualification for review.
`--accuracy-limit-seconds N` limits the evaluated prefix and records that scope;
use full held-out recordings for the final assessment. Keep the machine under
representative background activity; do not run unrelated GPU benchmarks.

## Acceptance and results

The default timing budget is zero native missed deadlines and at most eight
samples of completion backlog. Every test must consume the expected sample and
evaluation counts without invalid input/reset/loss (a bounded initial synchronization prefix is
allowed only for live UART), retain the required worker
scheduling and exercise positive triggers. Override the timing budgets only if
that is an explicitly accepted operating requirement.

Accuracy limits are **not invented by the tool**. Supply your agreed bounds:

```text
--min-sensitivity FRACTION --max-fp-per-minute RATE
--max-emission-latency-ms MILLISECONDS   # optional additional bound
```

These are acceptance checks on fixed results, not threshold selection. Without
both primary accuracy bounds, all checks still run but the result is
`review_required`. Held-out status is declared by the caller; the tool does not
prove the provenance of the training/test split.

Read `qualification_report.md` for the summary and `qualification_report.json`
for complete hashes, acceptance states and per-stage evidence. `accuracy/`
contains the paired raw/frame results and event/probability logs; `prepared/`
contains the Jetson bundle; `service-*` contains each timing/context/thermal run.
Original model, recording, service-config, deployed-pair, relevant source-code
and built native-binary hashes are checked again on exit. Code edits during a
run invalidate its result. The report records the commit, working-tree state
and target software versions. A failure retains its evidence and prevents a passing report.

Exit statuses are **0 passed**, **1 failed/error**, **2 CLI usage error**, and
**3 checks completed but review is required**. A dry run exits 0. Nothing is
activated after success. New weights or changed runtime/service settings need
new qualification rather than reuse of an old report.
