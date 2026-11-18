# Jetson deployment and qualification

Run all commands from the project root, `/home/orin/work/live_stack`, using
`/home/orin/envs/jet/bin/python`. Preparation needs an idle detector and no
concurrent training or GPU validation. The commands refuse an active or
unverified detector service; they never stop it. No physical UART/GPIO is opened
by model preparation, and no power or clock setting is changed.

After [receiving a delayed 100 Hz model](#receive-the-finished-model), use
[qualification](#qualify-a-finished-model) for parity, held-out accuracy, and
service-context timing against a 50 Hz baseline. For standard 50 Hz models or
individual diagnostic steps, use [manual bundle preparation](#build-an-isolated-bundle)
and [processing measurements](#measure-sustained-processing).
Review the [service profile](#worker-scheduling-profile)
before installation, then [activate the validated model](#activate-the-validated-model)
after accepting accuracy, timing, and hardware results.

## Receive the finished model

Transfer the selected checkpoint and its **matching, unedited** configuration
into a new model directory. For the delayed family, the publication location is
`finished_models/<family>/delayed_100hz/`, with a distinct `modelname` in the
configuration. The normal best variant needs `best.pth` and `train_config.json`;
the tuned variant needs `best_tuned.pth` and `tuning/tuned_config.json`.
Retain training/validation reports alongside them for the accuracy review.
Finish the transfer before preparing. Existing source files are never replaced
by the preparation tool.

Do not reuse the original 50 Hz weights or a workstation TensorRT engine.
Preparation re-exports ONNX and builds the engine on this Jetson. It preserves
saved weights, thresholds, physical normalization horizons and holdoff overrides.
A delayed model must declare `prediction_mode=delayed_100hz`,
`prediction_delay_frames=1`, `raw_hz=400`, `frame_size_raw=4`, and preprocessing
contract 4. A standard one-second window contains 100 frames. The newest
recurrent state predicts the previous frame.

The UART transmits two ECG leads; this runtime derives ECG3 as **ECG2 − ECG1**
after source offset removal. The incoming model's training data must use the
same channel definition. `--ecg3-source ecg2_minus_ecg1` explicitly confirms this
and derives the third channel in temporary parity inputs. A saved recording's
independently measured `ecg3_raw` is not assumed equivalent. Source recordings
and annotations remain unchanged.

## Qualify a finished model

For a delayed 100 Hz candidate and its 50 Hz baseline, run the combined
qualification entrypoint from the project root as the ordinary user:

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

### What runs

1. **Implementation and engine parity.** Rebuild the existing native build only
   as needed. Development regression suites are maintained upstream and
   are not bundled in this public distribution; the report records
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
change power/clocks, retrain, change annotations, or tune thresholds. Measure
physical GPIO timing separately.

Timing reports include p50/p95/p99/max processing and host-decode-to-completion,
CPU/GPU stages, backlog, deadlines, sample counts, valid-input/reset/loss counters,
positive output coverage, temperatures and unchanged power mode. Native deadline
limits start at host decode; source-pacer delays are reported separately. This
is not a proof of a universal worst-case bound. Power watts may be unavailable.

### Authentication and duration

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

### Acceptance and results

The default timing budget is zero native missed deadlines and at most eight
samples of completion backlog. Every test must consume the expected sample and
evaluation counts without invalid input/reset/loss (a bounded initial
synchronization prefix is allowed only for live UART), retain the required
worker scheduling and exercise positive triggers. Override the timing budgets only if
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
and target software versions. A failure retains its evidence and prevents a
passing report.

Exit statuses are **0 passed**, **1 failed/error**, **2 CLI usage error**, and
**3 checks completed but review is required**. A dry run exits 0. Nothing is
activated after success. New weights or changed runtime/service settings need
new qualification rather than reuse of an old report.

## Build an isolated bundle

Example, after substituting the actual model identity and source directory:

```bash
MODEL_NAME=ecg3_delayed_100hz_full
MODEL_DIR=finished_models/ecg3/delayed_100hz
BUNDLE="artifact/${MODEL_NAME}-jetson-$(date +%Y%m%d-%H%M%S)"
bash prepare_model.sh "$MODEL_NAME" --model-dir "$MODEL_DIR" \
    --jetson-only --prepare-only --output-dir "$BUNDLE" \
    --validation-input data/live/p12/p12_out.parquet \
    --validation-input data/live/p13/p13_out.parquet \
    --max-stream-samples 0 --ecg3-source ecg2_minus_ecg1 \
    --python /home/orin/envs/jet/bin/python
```

Choose representative raw parity recordings with valid input and beats. They
need not be held out, since this check does not tune anything or measure
accuracy. Repeat `--validation-input` for more coverage. The default limit is
4,000 raw samples per recording; `--max-stream-samples 0` processes all of them.
`--prepped-file PATH` optionally supplies real saved windows for numerical
Torch/ONNX/TRT checks; otherwise those numerical checks use seeded random
windows, with real causal stream validation performed separately. ONNX Runtime
is optional: the CPU ONNX reference evaluator is used when it is unavailable.
`mplhep` is not required for this preparation workflow.

The output directory must be new and outside `deploy/`. It contains:

- `model/train_config.json` and `model/best.pth`: exact source copies, including
  when the selected source variant was tuned.
- A newly exported ONNX file and Jetson-built engine candidates.
- `model/model.engine` and fingerprinted `model/model.conf`: selected pair.
- Numerical-check logs, Torch/native stream logs and `validation_report.json`.

Default `--precision auto` tries FP16, checks logits and then requires matching
real-stream probabilities, candidate/event positions and emission coordinates.
A precision or event mismatch triggers a separate strict FP32 build. Failed
candidate evidence is retained. A numerical pass alone does not establish
trigger parity near a threshold. The numerical inference timing check is also
required; it is **inference-only**. A performance failure is reported rather
than treated as a precision fix.

`validation_passed=true` means these preparation checks passed. It does not
certify accuracy, sustained latency or physical output timing. Reports record
those limits. A recording with no emitted events provides probability coverage
but no positive trigger coverage; inspect the reported event counts. Choose
additional fixtures when needed. Re-run preparation for any changed weights,
configuration, recovery policy or target device. The deploy command verifies
config/checkpoint/engine hashes and requires this report for delayed models.

`--prepare-only` never activates anything. Omitting it retains the existing
prepare-and-activate workflow; delayed models then go through these same gates.
Use `--dry-run` to inspect the plan without creating the bundle.

## Measure sustained processing

After numerical/event validation, while the Jetson is idle, use safely paced
recorded UART groups through a private pseudo-terminal:

```bash
/home/orin/envs/jet/bin/python run/jetson_paced_uart_check.py \
    --manifest "$BUNDLE/model/model.conf" \
    --recording data/pmu/raw/uart/p13/p13_out.bin \
    --seconds 612 --output-dir "${BUNDLE}-paced"
```

This produces 400 raw samples/s (100 four-sample groups/s), hence 100 delayed
evaluations/s after normal warmup. Standard eight-sample models produce 50/s.
The tool repeats recorded groups if necessary; this is a processing load test,
not an accuracy test. It cannot measure physical UART transport or GPIO edges.
GPIO remains disabled. Use `--no-early-emission` only when intentionally testing
that existing trigger policy; the default matches the runtime's default.

Check `pacer.json`, `runtime.log`, `tegrastats.log`, `power-mode.txt` and
`latency-summary.json`. Require the expected sample/evaluation counts, no
invalid input/reset/loss, bounded backlog and acceptable deadline counts. The
summary includes p50/p95/p99/max processing, host-decode-to-completion delay,
queue delay, CPU thread time, CUDA upload/compute/download and missed deadlines.
Processing includes preprocessing for every raw sample, window assembly,
transfers, inference and trigger processing. Startup preprocessing is accumulated
into the first evaluation. Output-queue/GPIO completion is measured separately.

The frame-end anchor is **host UART decode time**, not the physical acquisition
edge. Stage CPU time excludes waiting/descheduling; CUDA event time can include
interleaved GPU work. Use these together to localize stalls. They do not identify
an OS thread, interrupt or driver operation responsible for a historical stall.
For an isolated diagnostic comparison, `DKFZ_DISABLE_CUDA_GRAPH=1` forces normal
enqueue without editing the model or deployment. GPU stage timing is enabled
with `--emit-probabilities`; it adds measurement overhead.

Production health diagnostics also retain `missed_frame_deadlines` and
`max_frame_end_to_completion_ns` when probability output is disabled. A direct
shell benchmark does not reproduce the
[service scheduling profile](#worker-scheduling-profile); use qualification to
measure the service context.

The installed power/clock policy is only queried. A Linux/CUDA system without
additional scheduling guarantees does not acquire a hard worst-case bound from
one successful run. For physical GPIO latency, measure input and output edges
with suitable hardware before enabling physical triggering.

### Live UART processing

For a separately validated/prepared model, with the service/direct runtime
stopped, measure live UART processing without GPIO using:

```bash
mkdir -p tmp/jetson-latency
build_native/inference/native_uart_live_runtime \
    --config config/native_uart_live.conf --manifest "$BUNDLE/model/model.conf" \
    --trigger-output jsonl --emit-probabilities > tmp/jetson-latency/runtime.jsonl

/home/orin/envs/jet/bin/python run/summarize_inference_latency.py \
    tmp/jetson-latency/runtime.jsonl
```

The report includes p50/p95/p99/max processing across all four samples,
preprocessing, transfers, inference and trigger processing; decoder-frame-end
to completion latency; backlog; and missed 10 ms deadlines. Startup is included.
Frame-end latency uses the host `decoded_monotonic_ns` anchor, not the sample
clock reconstructed from the number of decoded samples. Logs without that
anchor from older binaries need a new measurement; unpaced replay can place
reconstructed sample times in the future and previously reported zero delay.
GPIO runs additionally log host decision-to-assertion timing. Verify physical
input/output edges independently. Unpaced byte replay cannot validate sustained
100 Hz operation.

## Worker scheduling profile

`deploy/dkfz-native-uart-live.service` supplies a candidate profile that must
be qualified on the target before installation:

| Setting | Purpose |
| --- | --- |
| `--realtime-priority 20 --require-realtime` | Acquisition, inference and output workers request and verify `SCHED_RR` priority 20. |
| `CPUSchedulingPolicy=other` | Main/CUDA initialization and supervision retain normal scheduling. |
| `LimitRTPRIO=20` | Permits the bounded worker priority without granting `CAP_SYS_NICE`. |
| `LimitRTTIME=200ms` | Limits continuous real-time CPU use by a worker that stops blocking normally. |
| `CPUWeight=1000` | Gives normal-priority runtime/helper work more weight under CPU contention; no CPU quota is added. |
| `TimerSlackNSec=1000` | Reduces timer coalescing for the service. |

Existing `Nice=-5`, watchdogs, GPIO configuration, source calibration and model
settings are retained. No power/clock policy, global scheduler knob or CPU
isolation/pinning is changed. No whole-model memory lock is requested.

Only the three application workers explicitly promote themselves after CUDA
initialization. `worker_scheduling` diagnostics report requested/actual policy
and priority, named threads, whether real-time scheduling was applied, and any
failure. Required startup waits for all three scheduling checks before sending
systemd readiness. Missing process limits or an unavailable worker scheduler
fail with configuration status 2, preventing a restart loop under the unit's
existing `RestartPreventExitStatus=2`.

Standalone runs preserve the inherited scheduler by default:
`realtime_priority=0`, `require_realtime=false`. This preserves existing 50 Hz
and isolated replay behavior. An optional priority request logs degradation and
continues if the bounded prerequisites or permissions are missing. A required
request fails instead. Priorities outside 0–20, required priority zero, and
unbounded/zero required RT CPU limits are rejected. Read-only `--validate-only`
checks prerequisites; it does not change scheduling or open UART/GPIO.

`LimitRTTIME` bounds continuous RT CPU use, **not** the maximum completion delay
of a frame. RT scheduling does not eliminate interrupts, kernel/driver waits,
priority inversion or device stalls. Qualify the installed kernel and driver
stack before relying on its timing. See the
[Linux scheduling reference](https://man7.org/linux/man-pages/man7/sched.7.html)
and [systemd execution settings](https://www.freedesktop.org/software/systemd/man/249/systemd.exec.html).
Measure this profile with the [qualification workflow](#qualify-a-finished-model)
before installing the service template.

## Activate the validated model

Evaluate each model on held-out recordings using saved validation-selected
thresholds. Review accuracy under metric schema 4 and compare raw annotations
on a common time interval separately; see the
[100 Hz metric policy](100hz-delayed-target.md#tp-pe-and-ope-definitions-option-2).

For old training runs, recomputing current validation metrics from saved scores
is distinct from selecting a different best epoch/checkpoint. An older run that
saved only its former best/last weights cannot reconstruct discarded epochs.
Do not relabel historical schema-3 counts or overwrite weights/configuration as
part of deployment. Fresh training already uses the current metric policy.

After the model's accuracy, sustained processing and hardware checks are
accepted, the existing activation command can install the validated pair while
the service is already idle:

```bash
bash run/deploy_model.sh "$MODEL_NAME" --model-dir "$BUNDLE/model" \
    --config "$BUNDLE/model/train_config.json" \
    --checkpoint "$BUNDLE/model/best.pth" --engine "$BUNDLE/model/model.engine" \
    --validated-report "$BUNDLE/validation_report.json" \
    --python /home/orin/envs/jet/bin/python
```

This command activates files; run it only when activation is intended. It does
not start the service. Starting physical input/output remains a separate action.

## Older 50 Hz models

Current-frame 50 Hz models remain supported when explicitly selected.
They retain their saved frame rate and target contract; new training defaults
to delayed 100 Hz. Both Python runtime configuration
and the native manifest reader accept saved preprocessing contract 3 for the
existing 400 Hz/eight-sample model with default quality histories (10/200
frames). The original version marker is preserved; it is not rewritten to 4.
Contract 3 is rejected for delayed/four-sample or incompatible quality-history
settings. New contract-4 models continue to work. Old preprocessing **training
caches** remain rejected and must be regenerated from raw input for fresh work.
Existing 50 Hz preparation/deployment commands and explicit holdoff/recovery
policies remain available. Rebuild an old engine on the target if its device or
software provenance is uncertain, even if deserialization succeeds.
