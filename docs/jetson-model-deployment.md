# Prepare a 50 Hz or delayed 100 Hz model on the Jetson

Run all commands from the project root, `/home/orin/work/live_stack`, using
`/home/orin/envs/jet/bin/python`. Preparation needs an idle detector and no
concurrent training or GPU validation. The commands refuse an active or
unverified detector service; they never stop it. No physical UART/GPIO is opened
by model preparation, and no power or clock setting is changed.

For the combined parity, fixed-threshold accuracy and service-profile checks,
use the [single qualification script](model-qualification.md). It includes
paced and GPIO-disabled live UART checks without production activation.

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
The optimized native engine uses persistent pinned host buffers, startup warmup,
a nonblocking CUDA stream and CUDA graph launch when capture succeeds. Engines
that cannot be captured retain normal enqueue, following
[NVIDIA's capture/fallback guidance](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/performance/optimization.html#cuda-graphs).
For an isolated diagnostic comparison, `DKFZ_DISABLE_CUDA_GRAPH=1` forces normal
enqueue without editing the model or deployment. GPU stage timing is enabled with
`--emit-probabilities`; it adds measurement overhead.

Production health diagnostics also retain `missed_frame_deadlines` and
`max_frame_end_to_completion_ns` when probability output is disabled. The installed service's original `Nice=-5`/CPU weight policy differs from a
direct shell benchmark's normal scheduling. The repository now also stages a
[bounded worker scheduling profile](service-latency.md), which needs privileged
paced qualification before installation. Neither is a hard real-time guarantee.

The installed power/clock policy is only queried. A Linux/CUDA system without
additional scheduling guarantees does not acquire a hard worst-case bound from
one successful run. For physical GPIO latency, measure input and output edges
with suitable hardware before enabling physical triggering.

## Accuracy and activation

Every transferred model needs its own validation/test review, without
threshold tuning on held-out recordings. Comparisons between models
trained on different data do not isolate the effect of framing.
Use metric schema 4: absolute 100 Hz offsets 0–1 TP, 2–3 PE, 4–5 oPE; standard
50 Hz remains 0/1/2. Compare raw annotations on a common time interval separately.

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

Current-frame behavior remains the default. Both Python runtime configuration
and the native manifest reader accept saved preprocessing contract 3 for the
existing 400 Hz/eight-sample model with default quality histories (10/200
frames). The original version marker is preserved; it is not rewritten to 4.
Contract 3 is rejected for delayed/four-sample or incompatible quality-history
settings. New contract-4 models continue to work. Old preprocessing **training
caches** remain rejected and must be regenerated from raw input for fresh work.
Existing 50 Hz preparation/deployment commands and explicit holdoff/recovery
policies remain available. Rebuild an old engine on the target if its device or
software provenance is uncertain, even if deserialization succeeds.
