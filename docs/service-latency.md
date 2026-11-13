# Jetson service scheduling

The supplied service template uses a bounded worker scheduling profile. Qualify
the model and profile on the target installation before installing the unit.
Timing evidence from another model or machine does not establish its behavior
on your system.

| Setting | Purpose |
| --- | --- |
| `--realtime-priority 20 --require-realtime` | Acquisition, inference and output workers request and verify `SCHED_RR` priority 20. |
| `CPUSchedulingPolicy=other` | Initialization and supervision retain normal scheduling. |
| `LimitRTPRIO=20` | Permits bounded worker priority without granting `CAP_SYS_NICE`. |
| `LimitRTTIME=200ms` | Limits continuous real-time CPU use by a worker that stops blocking normally. |
| `CPUWeight=1000` | Gives normal-priority runtime/helper work more weight under CPU contention. |
| `TimerSlackNSec=1000` | Reduces timer coalescing for the service. |

The template also uses `Nice=-5`, readiness notifications and watchdogs. Only
the application workers promote themselves after CUDA initialization.
`worker_scheduling` diagnostics report requested and actual policy and priority,
thread names, and scheduling failures. Startup requires all three scheduling
checks before notifying systemd readiness. Missing prerequisites fail with
configuration status 2, preventing a restart loop.

Standalone runs retain the inherited scheduler by default:
`realtime_priority=0`, `require_realtime=false`. An optional priority request may
continue with degraded scheduling; a required request fails. Read-only
`--validate-only` checks prerequisites without changing scheduling or opening
UART/GPIO.

## Qualification and installation

Follow [model qualification](model-qualification.md) for delayed-model parity,
fixed-weight accuracy comparisons, paced processing, and service-account/sandbox
checks. The production detector must be idle. The qualification tool does not
install, reload or start the production service.

Check actual worker policies, sample/evaluation counts, invalid-input/reset/loss
counters, backlog, missed deadlines, temperatures and power conditions. Keep
weights, thresholds, holdoff and recordings fixed for comparisons. Physical
UART/GPIO latency needs separate hardware measurements.

`LimitRTTIME` bounds continuous real-time CPU use; it does not bound frame
completion delay. Scheduling does not eliminate interrupts, driver waits or
device stalls. One successful run does not establish a universal worst-case
bound. See the [Linux scheduling reference](https://man7.org/linux/man-pages/man7/sched.7.html)
and [systemd execution settings](https://www.freedesktop.org/software/systemd/man/249/systemd.exec.html).

After qualification, follow [service installation](hardware.md#install-the-service).
Build the matching native runtime before installing the template's command-line
flags. Link/reload and startup are separate operator actions. Existing units
without the real-time flags retain their previous scheduling behavior.

For transferred checkpoints, preserve the matching configuration and build a
fresh TensorRT engine using [Jetson preparation](jetson-model-deployment.md).
The stack supports current-frame 50 Hz and separately trained delayed 100 Hz
models. Saved contract-3 configurations are accepted only for compatible
400 Hz/eight-sample current-frame models with default quality histories;
delayed models require contract 4.
