/*
 * Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
 * SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
 * Developed at the German Cancer Research Center (DKFZ).
 */

#include "dkfz_live/integrated_runtime.hpp"

#include "dkfz_live/bounded_queue.hpp"
#include "dkfz_live/diagnostics.hpp"
#include "dkfz_live/live_inference_pipeline.hpp"
#include "dkfz_live/systemd_notify.hpp"
#include "dkfz_live/threshold_override.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <iostream>
#include <map>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

namespace dkfz::live
{
namespace
{

class FailureState
{
  public:
    bool set(
        IntegratedExitCode exit_code,
        std::string code,
        std::string component,
        std::string message)
    {
        int expected = 0;
        if (!exit_code_.compare_exchange_strong(
                expected,
                static_cast<int>(exit_code),
                std::memory_order_acq_rel))
        {
            return false;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        code_ = std::move(code);
        component_ = std::move(component);
        message_ = std::move(message);
        return true;
    }

    bool failed() const noexcept
    {
        return exit_code_.load(std::memory_order_acquire) != 0;
    }

    IntegratedExitCode exit_code() const noexcept
    {
        return static_cast<IntegratedExitCode>(
            exit_code_.load(std::memory_order_acquire));
    }

  private:
    std::atomic<int> exit_code_{0};
    mutable std::mutex mutex_;
    std::string code_;
    std::string component_;
    std::string message_;
};

RawSample to_raw_sample(
    const dkfz::uart::DecodedSample &decoded,
    const dkfz::uart::UartSignalOffsets &offsets)
{
    const auto calibrated = dkfz::uart::calibrate_signals(decoded, offsets);
    RawSample sample;
    sample.ecg1 = calibrated.ecg1;
    sample.ecg2 = calibrated.ecg2;
    sample.ecg3 = calibrated.ecg3;
    sample.puls = calibrated.puls;
    sample.sample_idx = static_cast<std::int64_t>(decoded.raw_index);
    sample.time_ms = decoded.time_ms;
    return sample;
}

} // namespace

IntegratedLiveRuntime::IntegratedLiveRuntime(
    DeploymentManifest manifest,
    std::unique_ptr<LogitEngine> engine,
    IntegratedRuntimeConfig config,
    std::ostream &event_output,
    std::ostream &diagnostic_output)
    : manifest_(std::move(manifest)),
      engine_(std::move(engine)),
      config_(std::move(config)),
      event_output_(event_output),
      diagnostic_output_(diagnostic_output)
{
    validate_deployment_manifest(manifest_);
    if (!engine_)
    {
        throw std::invalid_argument("Integrated runtime requires an engine");
    }
    if (config_.sample_rate_hz == 0 || config_.queue_capacity == 0)
    {
        throw std::invalid_argument(
            "sample_rate_hz and queue_capacity must be positive");
    }
    if (std::abs(
            manifest_.framer.raw_hz -
            static_cast<double>(config_.sample_rate_hz)) > 1e-9)
    {
        throw std::invalid_argument(
            "UART sample_rate_hz does not match the deployment manifest raw_hz");
    }
    if (manifest_.framer.ecg1_offset != 0.0 ||
        manifest_.framer.ecg2_offset != 0.0 ||
        manifest_.framer.ecg3_offset != 0.0 ||
        manifest_.framer.puls_offset != 0.0)
    {
        throw std::invalid_argument(
            "Integrated UART input is already source-calibrated; deployment "
            "manifest signal offsets must be zero");
    }
    validate_runtime_output_config(
        config_.output,
        !config_.source.replay_path.empty());
    if (config_.emit_probabilities &&
        !runtime_output_has_jsonl(config_.output.mode))
    {
        throw std::invalid_argument(
            "emit_probabilities requires jsonl or gpio+jsonl output mode");
    }
    if (runtime_output_has_gpio(config_.output.mode) &&
        config_.watchdog.enabled &&
        config_.watchdog.output_timeout_ns > 0 &&
        (config_.watchdog.output_timeout_ns <=
             config_.output.gpio.pulse_width_ns ||
         config_.watchdog.output_timeout_ns -
                 config_.output.gpio.pulse_width_ns <=
             config_.output.gpio.idle_width_ns))
    {
        throw std::invalid_argument(
            "output timeout must exceed the GPIO pulse plus idle width");
    }
}

IntegratedRuntimeSummary IntegratedLiveRuntime::run(
    std::atomic<bool> &stop_requested)
{
    DiagnosticLogger logger(diagnostic_output_);
    SystemdNotifier notifier;
    const bool check_puls = manifest_.framer.include_puls ||
                            manifest_.framer.signal_quality_mode ==
                                SignalQualityMode::ecg_puls;
    RuntimeHealth health;
    health.started_ns = dkfz::uart::monotonic_clock_ns();
    BoundedQueue<dkfz::uart::DecodedSample> queue(config_.queue_capacity);
    BoundedQueue<RuntimeOutputRecord> output_queue(
        config_.output.queue_capacity);
    std::atomic<std::uint64_t> input_reset_generation{0};
    std::atomic<std::uint32_t> input_reset_reasons{0};
    std::atomic<bool> input_paused{false};
    std::mutex input_state_mutex;
    FailureState failure;
    std::unique_ptr<RuntimeOutputSink> output_sink;
    try
    {
        output_sink = make_runtime_output_sink(
            config_.output, event_output_);
    }
    catch (const std::exception &error)
    {
        logger.emit(
            DiagnosticSeverity::fatal,
            "trigger_output_initialization_failed",
            "trigger_output",
            error.what(),
            {{"output_mode", runtime_output_mode_name(config_.output.mode)}});
        notifier.stopping("Trigger output initialization failed");
        IntegratedRuntimeSummary summary;
        summary.exit_code = IntegratedExitCode::output;
        return summary;
    }
    auto pipeline = std::make_shared<LiveInferencePipeline>(
        manifest_, std::move(engine_), config_.early_emission);
    ThresholdOverrideReader threshold_reader(config_.threshold_override_path, manifest_);
    auto reported_policy = pipeline->active_policy();
    auto requested_policy = reported_policy;
    auto manual_threshold = manifest_.manual_threshold;
    std::optional<TriggerPolicy> file_policy;
    std::string last_manifest_stamp;
    std::string last_manifest_error;
    auto report_threshold = [&]()
    {
        const auto active = pipeline->active_policy();
        if (active.threshold != reported_policy.threshold ||
            active.early_emission != reported_policy.early_emission)
        {
            // Diagnostics can block on a mutex/journald: keep them on the
            // supervisor, after the inference thread has applied the change.
            logger.emit(DiagnosticSeverity::info, "threshold_changed", "inference",
                        "Supervisor observed an applied live threshold update",
                        {{"previous_threshold", reported_policy.threshold},
                         {"threshold", active.threshold},
                         {"early_emission", active.early_emission},
                         {"emission_threshold", active.early_emission
                             ? std::max(active.threshold, manifest_.emission_reference_threshold())
                             : active.threshold}});
            reported_policy = active;
        }
    };
    auto poll_threshold = [&]()
    {
        const auto previous_manual_threshold = manual_threshold;
        if (!manifest_.source_path.empty())
        {
            try
            {
                const auto stamp = std::to_string(std::filesystem::last_write_time(
                    manifest_.source_path).time_since_epoch().count()) + ":" +
                    std::to_string(std::filesystem::file_size(manifest_.source_path));
                if (stamp != last_manifest_stamp)
                {
                    last_manifest_stamp = stamp;
                    manual_threshold = reload_manual_threshold(manifest_);
                    last_manifest_error.clear();
                }
            }
            catch (const std::exception &error)
            {
                const auto stamp = "error:" + std::string(error.what());
                if (last_manifest_error != stamp)
                {
                    logger.emit(DiagnosticSeverity::warning, "manual_threshold_rejected",
                                "inference", error.what());
                    last_manifest_error = stamp;
                }
            }
        }
        const auto update = threshold_reader.poll();
        if (!update.error.empty())
        {
            logger.emit(DiagnosticSeverity::warning, "threshold_override_rejected",
                        "inference", update.error);
        }
        if (update.threshold.has_value())
        {
            file_policy = TriggerPolicy{*update.threshold,
                update.early_emission.value_or(config_.early_emission)};
        }
        auto desired = file_policy.value_or(TriggerPolicy{manifest_.threshold, config_.early_emission});
        desired.threshold = manual_threshold.value_or(desired.threshold);
        if (manual_threshold != previous_manual_threshold)
        {
            // A saved manual choice is meaningful even when its number equals
            // the automatic threshold and no detector update is necessary.
            logger.emit(DiagnosticSeverity::info, "manual_threshold_configured",
                        "inference", "Observed saved manual-threshold configuration",
                        {{"manual_override_configured", manual_threshold.has_value()},
                         {"configured_threshold", desired.threshold}});
        }
        if (desired.threshold != requested_policy.threshold ||
            desired.early_emission != requested_policy.early_emission)
        {
            pipeline->request_threshold(desired.threshold, desired.early_emission);
            requested_policy = desired;
            logger.emit(DiagnosticSeverity::info, "threshold_requested", "inference",
                        "Threshold update queued for the next completed frame",
                        {{"threshold", desired.threshold},
                         {"early_emission", desired.early_emission},
                         {"manual_override", manual_threshold.has_value()}});
        }
    };
    poll_threshold();

    auto fail = [&](IntegratedExitCode exit_code,
                    const std::string &code,
                    const std::string &component,
                    const std::string &message)
    {
        if (failure.set(exit_code, code, component, message))
        {
            logger.emit(
                DiagnosticSeverity::fatal,
                code,
                component,
                message,
                {
                    {"queue_depth", static_cast<std::uint64_t>(queue.size())},
                    {"output_queue_depth",
                     static_cast<std::uint64_t>(output_queue.size())},
                    {"samples_decoded", health.samples_decoded.load()},
                    {"samples_consumed", health.samples_consumed.load()},
                    {"gpio_pulses", health.gpio_pulses.load()},
                });
        }
        stop_requested.store(true, std::memory_order_relaxed);
        queue.close();
        output_queue.close();
    };

    auto pause_input = [&](std::uint32_t reasons,
                           const std::string &source_condition,
                           const std::string &message)
    {
        std::size_t discarded_samples = 0;
        std::size_t discarded_outputs = 0;
        {
            std::lock_guard<std::mutex> lock(input_state_mutex);
            if (input_paused.load(std::memory_order_relaxed))
            {
                input_reset_reasons.fetch_or(
                    reasons, std::memory_order_relaxed);
                return false;
            }
            input_paused.store(true, std::memory_order_release);
            input_reset_reasons.store(reasons, std::memory_order_relaxed);
            discarded_samples = queue.clear();
            discarded_outputs = output_queue.clear();
            health.queue_depth.store(0, std::memory_order_relaxed);
            health.output_queue_depth.store(0, std::memory_order_relaxed);
            health.first_inference_expected_ns.store(
                0, std::memory_order_relaxed);
            health.last_inference_completed_ns.store(
                0, std::memory_order_relaxed);
            health.first_output_expected_ns.store(
                0, std::memory_order_relaxed);
            input_reset_generation.fetch_add(1, std::memory_order_release);
        }
        logger.emit(
            DiagnosticSeverity::warning,
            "input_paused",
            "input_validity",
            message,
            {
                {"source_condition", source_condition},
                {"puls_sentinel", (reasons & kInvalidPulsSentinel) != 0},
                {"puls_baseline", (reasons & kInvalidPulsBaseline) != 0},
                {"ecg_baseline", (reasons & kInvalidEcgBaseline) != 0},
                {"frozen_signal", (reasons & kInvalidFrozenSignal) != 0},
                {"uart_silence", (reasons & kInvalidUartSilence) != 0},
                {"decoder_stall", (reasons & kInvalidDecoderStall) != 0},
                {"discarded_samples",
                 static_cast<std::uint64_t>(discarded_samples)},
                {"discarded_outputs",
                 static_cast<std::uint64_t>(discarded_outputs)},
            });
        return true;
    };

    auto resume_input = [&]()
    {
        {
            std::lock_guard<std::mutex> lock(input_state_mutex);
            if (!input_paused.load(std::memory_order_relaxed))
            {
                return false;
            }
            input_paused.store(false, std::memory_order_release);
        }
        logger.emit(
            DiagnosticSeverity::info,
            "input_signal_recovered",
            "input_validity",
            "Input is stable again; inference will rebuild from an empty state");
        return true;
    };

    logger.emit(
        DiagnosticSeverity::info,
        "runtime_starting",
        "runtime",
        "Starting integrated UART and TensorRT runtime",
        {
            {"model", manifest_.model_name},
            {"threshold", pipeline->active_threshold()},
            {"automatic_threshold", manifest_.threshold},
            {"manual_override_configured", manual_threshold.has_value()},
            {"early_emission", config_.early_emission},
            {"threshold_override_path", config_.threshold_override_path.string()},
            {"variant", manifest_.model_variant},
            {"queue_capacity", static_cast<std::uint64_t>(config_.queue_capacity)},
            {"replay", !config_.source.replay_path.empty()},
            {"uart_ecg1_offset", config_.uart_offsets.ecg1},
            {"uart_ecg2_offset", config_.uart_offsets.ecg2},
            {"uart_puls_offset", config_.uart_offsets.puls},
            {"ecg3_derived", manifest_.framer.include_ecg3},
            {"output_mode", runtime_output_mode_name(config_.output.mode)},
            {"output_queue_capacity",
             static_cast<std::uint64_t>(config_.output.queue_capacity)},
            {"status_interval_ns", config_.status_interval_ns},
            {"paused_status_interval_ns",
             config_.paused_status_interval_ns},
            {"gpio_active_high", config_.output.gpio.active_high},
            {"gpio_pulse_width_ns", config_.output.gpio.pulse_width_ns},
            {"gpio_idle_width_ns", config_.output.gpio.idle_width_ns},
            {"input_reset_enabled", config_.input_reset.enabled},
            {"puls_input_checks_enabled", config_.input_reset.enabled && check_puls},
            {"puls_baseline_reset_ms",
             config_.input_reset.puls_baseline_reset_ms},
            {"ecg_baseline_reset_ms",
             config_.input_reset.ecg_baseline_reset_ms},
            {"stuck_signal_reset_ms",
             config_.input_reset.stuck_signal_reset_ms},
            {"input_recovery_ms", config_.input_reset.recovery_ms},
            {"watchdog_enabled", config_.watchdog.enabled},
        });

    const auto expected_samples =
        manifest_.framer.frame_size_raw * manifest_.window_size;
    const auto output_enabled =
        config_.output.mode != RuntimeOutputMode::none;

    auto enqueue_output = [&](RuntimeOutputRecord record)
    {
        if (!output_enabled)
        {
            return true;
        }
        const auto now = dkfz::uart::monotonic_clock_ns();
        std::uint64_t unset = 0;
        static_cast<void>(
            health.first_output_expected_ns.compare_exchange_strong(
                unset, now, std::memory_order_relaxed));

        bool pushed = false;
        if (!config_.source.replay_path.empty())
        {
            while (!stop_requested.load(std::memory_order_relaxed) &&
                   !output_queue.closed())
            {
                if (output_queue.wait_push(
                        record, std::chrono::milliseconds(50)))
                {
                    pushed = true;
                    break;
                }
            }
        }
        else
        {
            pushed = output_queue.try_push(std::move(record));
        }
        if (!pushed)
        {
            if (!stop_requested.load(std::memory_order_relaxed) &&
                !output_queue.closed())
            {
                fail(
                    IntegratedExitCode::output,
                    "trigger_output_queue_overflow",
                    "trigger_output",
                    "Trigger output did not keep up with live inference");
            }
            return false;
        }
        const auto depth = output_queue.size();
        health.output_queue_depth.store(
            static_cast<std::uint64_t>(depth),
            std::memory_order_relaxed);
        update_atomic_max(
            health.output_queue_high_water,
            static_cast<std::uint64_t>(output_queue.high_water_mark()));
        return true;
    };

    std::thread output_thread(
        [&, owned_sink = std::move(output_sink)]() mutable
        {
            try
            {
                RuntimeOutputRecord record;
                while (true)
                {
                    const auto pop = output_queue.wait_pop(
                        record, std::chrono::milliseconds(100));
                    if (pop == QueuePopResult::timeout)
                    {
                        continue;
                    }
                    if (pop == QueuePopResult::closed)
                    {
                        break;
                    }
                    health.output_queue_depth.store(
                        static_cast<std::uint64_t>(output_queue.size()),
                        std::memory_order_relaxed);
                    const auto started = dkfz::uart::monotonic_clock_ns();
                    health.output_active_since_ns.store(
                        started, std::memory_order_release);
                    try
                    {
                        owned_sink->write(record);
                    }
                    catch (...)
                    {
                        health.output_active_since_ns.store(
                            0, std::memory_order_release);
                        throw;
                    }
                    const auto completed = dkfz::uart::monotonic_clock_ns();
                    health.output_active_since_ns.store(
                        0, std::memory_order_release);
                    health.last_output_completed_ns.store(
                        completed, std::memory_order_release);
                    health.output_records.fetch_add(
                        1, std::memory_order_relaxed);
                    if (record.kind == RuntimeOutputRecordKind::trigger &&
                        runtime_output_has_gpio(config_.output.mode))
                    {
                        health.gpio_pulses.fetch_add(
                            1, std::memory_order_relaxed);
                    }
                }
                owned_sink->close();
            }
            catch (const std::exception &error)
            {
                health.output_active_since_ns.store(
                    0, std::memory_order_release);
                fail(
                    IntegratedExitCode::output,
                    "trigger_output_failed",
                    "trigger_output",
                    error.what());
            }
            health.output_done.store(true, std::memory_order_release);
        });

    std::thread acquisition_thread(
        [&]
        {
            try
            {
                std::uint64_t reported_malformed = 0;
                std::uint64_t reported_discarded = 0;
                std::uint64_t accepted_since_reset = 0;
                std::uint64_t observed_reset_generation = 0;
                InputValidityTracker input_validity(
                    config_.input_reset, config_.sample_rate_hz, check_puls);
                dkfz::uart::UartByteDecoder decoder(
                    config_.sample_rate_hz,
                    [&](const dkfz::uart::DecodedSample &sample)
                    {
                        if (stop_requested.load(std::memory_order_relaxed))
                        {
                            return false;
                        }
                        const auto now = dkfz::uart::monotonic_clock_ns();
                        health.last_sample_ns.store(now, std::memory_order_relaxed);
                        health.samples_decoded.fetch_add(
                            1, std::memory_order_relaxed);

                        const auto reset_generation =
                            input_reset_generation.load(
                                std::memory_order_acquire);
                        if (reset_generation != observed_reset_generation)
                        {
                            input_validity.require_recovery();
                            observed_reset_generation = reset_generation;
                            accepted_since_reset = 0;
                        }
                        const auto validity = input_validity.observe(sample);
                        if (!validity.accept_sample)
                        {
                            health.invalid_samples.fetch_add(
                                1, std::memory_order_relaxed);
                            if (validity.reset_required)
                            {
                                static_cast<void>(pause_input(
                                    validity.reasons,
                                    "decoded_signal_invalid",
                                    "Input became invalid; inference is paused until stable signal returns"));
                            }
                            return config_.max_samples == 0 ||
                                   sample.raw_index + 1 < config_.max_samples;
                        }
                        if (validity.recovered)
                        {
                            static_cast<void>(resume_input());
                        }
                        if (input_paused.load(std::memory_order_acquire))
                        {
                            health.invalid_samples.fetch_add(
                                1, std::memory_order_relaxed);
                            return config_.max_samples == 0 ||
                                   sample.raw_index + 1 < config_.max_samples;
                        }

                        ++accepted_since_reset;
                        if (accepted_since_reset >= expected_samples)
                        {
                            std::uint64_t unset = 0;
                            static_cast<void>(
                                health.first_inference_expected_ns
                                    .compare_exchange_strong(
                                        unset,
                                        now,
                                        std::memory_order_relaxed));
                        }

                        bool pushed = false;
                        if (!config_.source.replay_path.empty())
                        {
                            while (!stop_requested.load(std::memory_order_relaxed) &&
                                   !queue.closed())
                            {
                                if (queue.wait_push(
                                        sample, std::chrono::milliseconds(50)))
                                {
                                    pushed = true;
                                    break;
                                }
                            }
                        }
                        else
                        {
                            pushed = queue.try_push(sample);
                        }

                        if (!pushed)
                        {
                            if (!stop_requested.load(std::memory_order_relaxed) &&
                                !queue.closed())
                            {
                                fail(
                                    IntegratedExitCode::queue_overflow,
                                    "sample_queue_overflow",
                                    "sample_queue",
                                    "Inference did not keep up with live UART input");
                            }
                            return false;
                        }
                        const auto depth = queue.size();
                        health.queue_depth.store(
                            static_cast<std::uint64_t>(depth),
                            std::memory_order_relaxed);
                        update_atomic_max(
                            health.queue_high_water,
                            static_cast<std::uint64_t>(queue.high_water_mark()));
                        return config_.max_samples == 0 ||
                               sample.raw_index + 1 < config_.max_samples;
                    });

                const auto source_stats = dkfz::uart::run_byte_source(
                    config_.source,
                    [&](const std::uint8_t *bytes, std::size_t size)
                    {
                        const auto now = dkfz::uart::monotonic_clock_ns();
                        health.last_byte_ns.store(now, std::memory_order_relaxed);
                        health.bytes_read.fetch_add(
                            size, std::memory_order_relaxed);
                        decoder.process(bytes, size);

                        const auto &stats = decoder.stats();
                        health.malformed_groups.store(
                            stats.malformed_groups, std::memory_order_relaxed);
                        health.bytes_discarded.store(
                            stats.bytes_discarded, std::memory_order_relaxed);
                        if (stats.malformed_groups > reported_malformed ||
                            stats.bytes_discarded > reported_discarded)
                        {
                            logger.emit(
                                DiagnosticSeverity::warning,
                                "uart_resynchronized",
                                "uart_decoder",
                                "UART decoder discarded bytes while resynchronizing",
                                {
                                    {"malformed_groups", stats.malformed_groups},
                                    {"bytes_discarded", stats.bytes_discarded},
                                });
                            reported_malformed = stats.malformed_groups;
                            reported_discarded = stats.bytes_discarded;
                        }
                        return !decoder.stopped() &&
                               !stop_requested.load(std::memory_order_relaxed);
                    },
                    stop_requested);
                health.bytes_read.store(
                    source_stats.bytes_read, std::memory_order_relaxed);
                if (source_stats.capture_limit_reached)
                {
                    logger.emit(
                        DiagnosticSeverity::warning,
                        "capture_limit_reached",
                        "uart_capture",
                        "Raw UART capture reached its configured byte limit",
                        {{"capture_bytes", source_stats.capture_bytes}});
                }
            }
            catch (const std::exception &error)
            {
                fail(
                    IntegratedExitCode::acquisition,
                    "uart_acquisition_failed",
                    "uart",
                    error.what());
            }
            health.acquisition_done.store(true, std::memory_order_release);
            queue.close();
        });

    std::thread inference_thread(
        [&, owned_pipeline = pipeline]() mutable
        {
            try
            {
                dkfz::uart::DecodedSample decoded;
                std::uint64_t applied_reset_generation = 0;
                auto apply_pending_reset = [&]()
                {
                    const auto generation = input_reset_generation.load(
                        std::memory_order_acquire);
                    if (generation == applied_reset_generation)
                    {
                        return false;
                    }
                    owned_pipeline->reset();
                    applied_reset_generation = generation;
                    const auto reasons = input_reset_reasons.load(
                        std::memory_order_relaxed);
                    health.input_state_resets.fetch_add(
                        1, std::memory_order_relaxed);
                    logger.emit(
                        DiagnosticSeverity::info,
                        "input_state_reset",
                        "inference",
                        "Cleared normalization, framing, inference-window, and trigger state",
                        {
                            {"reset_generation", generation},
                            {"puls_sentinel",
                             (reasons & kInvalidPulsSentinel) != 0},
                            {"puls_baseline",
                             (reasons & kInvalidPulsBaseline) != 0},
                            {"ecg_baseline",
                             (reasons & kInvalidEcgBaseline) != 0},
                            {"frozen_signal",
                             (reasons & kInvalidFrozenSignal) != 0},
                            {"uart_silence",
                             (reasons & kInvalidUartSilence) != 0},
                            {"decoder_stall",
                             (reasons & kInvalidDecoderStall) != 0},
                        });
                    return true;
                };
                while (!stop_requested.load(std::memory_order_relaxed) ||
                       queue.size() > 0)
                {
                    static_cast<void>(apply_pending_reset());
                    const auto pop = queue.wait_pop(
                        decoded, std::chrono::milliseconds(100));
                    if (pop == QueuePopResult::timeout)
                    {
                        continue;
                    }
                    if (pop == QueuePopResult::closed)
                    {
                        break;
                    }
                    if (apply_pending_reset())
                    {
                        continue;
                    }
                    health.queue_depth.store(
                        static_cast<std::uint64_t>(queue.size()),
                        std::memory_order_relaxed);
                    const auto started = dkfz::uart::monotonic_clock_ns();
                    health.last_sample_consumed_ns.store(
                        started, std::memory_order_relaxed);
                    health.inference_active_since_ns.store(
                        started, std::memory_order_release);
                    std::optional<InferenceResult> result;
                    try
                    {
                        result = owned_pipeline->process_sample(
                            to_raw_sample(decoded, config_.uart_offsets),
                            config_.sid);
                    }
                    catch (...)
                    {
                        health.inference_active_since_ns.store(
                            0, std::memory_order_release);
                        throw;
                    }
                    const auto completed = dkfz::uart::monotonic_clock_ns();
                    health.inference_active_since_ns.store(
                        0, std::memory_order_release);
                    health.samples_consumed.fetch_add(
                        1, std::memory_order_relaxed);
                    if (apply_pending_reset())
                    {
                        continue;
                    }
                    if (!result.has_value())
                    {
                        continue;
                    }

                    const auto duration = completed - started;
                    health.last_inference_duration_ns.store(
                        duration, std::memory_order_relaxed);
                    update_atomic_max(health.max_inference_duration_ns, duration);
                    health.last_inference_completed_ns.store(
                        completed, std::memory_order_release);
                    health.frames_inferred.fetch_add(
                        1, std::memory_order_relaxed);
                    if (config_.emit_probabilities)
                    {
                        ProbabilityOutputRecord probability;
                        probability.sid = config_.sid;
                        probability.frame_idx = result->frame.frame_idx;
                        probability.start_sample_idx =
                            result->frame.start_sample_idx;
                        probability.end_sample_idx =
                            result->frame.end_sample_idx;
                        probability.time_ms = result->frame.end_time_ms;
                        probability.model_output = result->model_output;
                        probability.probability = result->probability;
                        probability.trigger_emitted_now =
                            result->trigger.has_value();
                        if (!enqueue_output(
                                RuntimeOutputRecord::from_probability(
                                    std::move(probability))))
                        {
                            break;
                        }
                    }
                    if (result->trigger.has_value())
                    {
                        health.trigger_events.fetch_add(
                            1, std::memory_order_relaxed);
                        if (!enqueue_output(
                                RuntimeOutputRecord::from_trigger(
                                    *result->trigger)))
                        {
                            break;
                        }
                    }
                }

                if (!failure.failed())
                {
                    const auto final_event = owned_pipeline->flush(config_.sid);
                    if (final_event.has_value())
                    {
                        health.trigger_events.fetch_add(
                            1, std::memory_order_relaxed);
                        static_cast<void>(enqueue_output(
                            RuntimeOutputRecord::from_trigger(*final_event)));
                    }
                }
            }
            catch (const std::exception &error)
            {
                health.inference_active_since_ns.store(
                    0, std::memory_order_release);
                fail(
                    IntegratedExitCode::inference,
                    "inference_failed",
                    "inference",
                    error.what());
            }
            output_queue.close();
            health.inference_done.store(true, std::memory_order_release);
        });

    notifier.ready("Integrated DKFZ UART/TensorRT runtime is running");
    auto next_status = health.started_ns + config_.status_interval_ns;
    auto next_threshold_poll = health.started_ns + 250'000'000ULL;
    bool status_input_paused = false;
    auto next_systemd_watchdog = health.started_ns;
    const auto systemd_interval = notifier.watchdog_interval();
    if (systemd_interval.has_value())
    {
        next_systemd_watchdog += static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                *systemd_interval)
                .count());
    }

    while (!failure.failed())
    {
        report_threshold();
        if (health.acquisition_done.load(std::memory_order_acquire) &&
            health.inference_done.load(std::memory_order_acquire) &&
            health.output_done.load(std::memory_order_acquire))
        {
            break;
        }
        if (stop_requested.load(std::memory_order_relaxed))
        {
            queue.close();
        }

        const auto now = dkfz::uart::monotonic_clock_ns();
        if (now >= next_threshold_poll)
        {
            poll_threshold();
            next_threshold_poll = now + 250'000'000ULL;
        }
        const auto input_watchdog_failure = evaluate_input_watchdog(
            health, config_.watchdog, now);
        if (input_watchdog_failure.has_value() && config_.input_reset.enabled)
        {
            const auto reason = input_watchdog_failure->component == "uart"
                ? kInvalidUartSilence
                : kInvalidDecoderStall;
            static_cast<void>(pause_input(
                reason,
                input_watchdog_failure->code,
                input_watchdog_failure->message +
                    "; inference is paused until stable signal returns"));
        }

        auto internal_watchdog = config_.watchdog;
        if (config_.input_reset.enabled)
        {
            internal_watchdog.monitor_uart = false;
        }
        const auto watchdog_failure = evaluate_watchdog(
            health, internal_watchdog, now);
        if (watchdog_failure.has_value())
        {
            fail(
                IntegratedExitCode::watchdog,
                watchdog_failure->code,
                watchdog_failure->component,
                watchdog_failure->message);
            notifier.stopping("Internal watchdog failed");
            if (config_.hard_exit_on_watchdog)
            {
                std::_Exit(static_cast<int>(IntegratedExitCode::watchdog));
            }
            break;
        }

        const auto current_input_paused =
            input_paused.load(std::memory_order_acquire);
        if (current_input_paused != status_input_paused)
        {
            status_input_paused = current_input_paused;
            next_status = now;
        }
        const auto status_interval_ns = status_input_paused
            ? config_.paused_status_interval_ns
            : config_.status_interval_ns;
        if (status_interval_ns > 0 && now >= next_status)
        {
            const auto bytes_read = health.bytes_read.load();
            const auto samples_decoded = health.samples_decoded.load();
            const auto frames_inferred = health.frames_inferred.load();
            const auto queue_depth = health.queue_depth.load();
            const auto policy = pipeline->active_policy();
            logger.emit(
                DiagnosticSeverity::info,
                "runtime_status",
                "runtime",
                "Integrated runtime health status",
                {
                    {"bytes_read", bytes_read},
                    {"samples_decoded", samples_decoded},
                    {"samples_consumed", health.samples_consumed.load()},
                    {"frames_inferred", frames_inferred},
                    {"threshold", policy.threshold},
                    {"manual_override_configured", manual_threshold.has_value()},
                    {"configured_threshold", requested_policy.threshold},
                    {"early_emission", policy.early_emission},
                    {"emission_threshold", policy.early_emission
                        ? std::max(policy.threshold, manifest_.emission_reference_threshold())
                        : policy.threshold},
                    {"trigger_events", health.trigger_events.load()},
                    {"invalid_samples", health.invalid_samples.load()},
                    {"input_state_resets",
                     health.input_state_resets.load()},
                    {"input_paused", current_input_paused},
                    {"queue_depth", queue_depth},
                    {"queue_high_water", health.queue_high_water.load()},
                    {"output_mode",
                     runtime_output_mode_name(config_.output.mode)},
                    {"output_records", health.output_records.load()},
                    {"gpio_pulses", health.gpio_pulses.load()},
                    {"output_queue_depth",
                     health.output_queue_depth.load()},
                    {"output_queue_high_water",
                     health.output_queue_high_water.load()},
                    {"malformed_groups", health.malformed_groups.load()},
                    {"bytes_discarded", health.bytes_discarded.load()},
                    {"last_inference_duration_ns",
                     health.last_inference_duration_ns.load()},
                    {"max_inference_duration_ns",
                     health.max_inference_duration_ns.load()},
                });
            notifier.status(
                "input=" + std::string(
                    current_input_paused ? "paused" : "running") +
                " bytes=" + std::to_string(bytes_read) +
                " samples=" + std::to_string(samples_decoded) +
                " frames=" + std::to_string(frames_inferred) +
                " queue=" + std::to_string(queue_depth) +
                " pulses=" +
                std::to_string(health.gpio_pulses.load()));
            next_status = now + status_interval_ns;
        }
        if (systemd_interval.has_value() && now >= next_systemd_watchdog)
        {
            notifier.watchdog();
            next_systemd_watchdog = now + static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    *systemd_interval)
                    .count());
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(25));
    }

    stop_requested.store(true, std::memory_order_relaxed);
    queue.close();
    acquisition_thread.join();
    inference_thread.join();
    output_queue.close();
    output_thread.join();
    report_threshold();

    IntegratedRuntimeSummary summary;
    summary.exit_code = failure.failed()
        ? failure.exit_code()
        : IntegratedExitCode::success;
    summary.bytes_read = health.bytes_read.load();
    summary.samples_decoded = health.samples_decoded.load();
    summary.samples_consumed = health.samples_consumed.load();
    summary.frames_inferred = health.frames_inferred.load();
    summary.trigger_events = health.trigger_events.load();
    summary.malformed_groups = health.malformed_groups.load();
    summary.bytes_discarded = health.bytes_discarded.load();
    summary.max_inference_duration_ns =
        health.max_inference_duration_ns.load();
    summary.queue_high_water = static_cast<std::size_t>(
        health.queue_high_water.load());
    summary.output_records = health.output_records.load();
    summary.gpio_pulses = health.gpio_pulses.load();
    summary.invalid_samples = health.invalid_samples.load();
    summary.input_state_resets = health.input_state_resets.load();
    summary.output_queue_high_water = static_cast<std::size_t>(
        health.output_queue_high_water.load());

    logger.emit(
        summary.exit_code == IntegratedExitCode::success
            ? DiagnosticSeverity::info
            : DiagnosticSeverity::error,
        "runtime_finished",
        "runtime",
        summary.exit_code == IntegratedExitCode::success
            ? "Integrated runtime stopped cleanly"
            : "Integrated runtime stopped after a fatal error",
        {
            {"exit_code", static_cast<std::int64_t>(summary.exit_code)},
            {"bytes_read", summary.bytes_read},
            {"samples_decoded", summary.samples_decoded},
            {"samples_consumed", summary.samples_consumed},
            {"frames_inferred", summary.frames_inferred},
            {"trigger_events", summary.trigger_events},
            {"output_mode", runtime_output_mode_name(config_.output.mode)},
            {"output_records", summary.output_records},
            {"gpio_pulses", summary.gpio_pulses},
            {"invalid_samples", summary.invalid_samples},
            {"input_state_resets", summary.input_state_resets},
            {"output_queue_high_water",
             static_cast<std::uint64_t>(summary.output_queue_high_water)},
            {"queue_high_water",
             static_cast<std::uint64_t>(summary.queue_high_water)},
            {"max_inference_duration_ns", summary.max_inference_duration_ns},
        });
    notifier.stopping(
        summary.exit_code == IntegratedExitCode::success
            ? "Integrated runtime stopped cleanly"
            : "Integrated runtime failed");
    return summary;
}

} // namespace dkfz::live
