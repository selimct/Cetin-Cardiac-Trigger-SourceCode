/*
 * Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
 * SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
 * Developed at the German Cancer Research Center (DKFZ).
 */

#include "dkfz_live/runtime_health.hpp"

namespace dkfz::live
{
namespace
{

bool exceeded(std::uint64_t now, std::uint64_t then, std::uint64_t timeout)
{
    return timeout > 0 && then > 0 && now >= then && now - then > timeout;
}

} // namespace

std::optional<WatchdogFailure> evaluate_input_watchdog(
    const RuntimeHealth &health,
    const WatchdogConfig &config,
    std::uint64_t now_ns)
{
    if (!config.enabled)
    {
        return std::nullopt;
    }
    const auto startup_expired =
        now_ns >= health.started_ns &&
        now_ns - health.started_ns > config.startup_grace_ns;

    if (config.monitor_uart &&
        !health.acquisition_done.load(std::memory_order_relaxed))
    {
        const auto last_byte = health.last_byte_ns.load(std::memory_order_relaxed);
        if (config.uart_timeout_ns > 0 && last_byte == 0 && startup_expired)
        {
            return WatchdogFailure{
                "uart_startup_timeout",
                "uart",
                "No UART bytes arrived before the startup deadline",
            };
        }
        if (exceeded(now_ns, last_byte, config.uart_timeout_ns))
        {
            return WatchdogFailure{
                "uart_silence_timeout",
                "uart",
                "UART byte stream stopped",
            };
        }

        const auto last_sample =
            health.last_sample_ns.load(std::memory_order_relaxed);
        if (config.sample_timeout_ns > 0 &&
            last_byte > 0 && last_sample == 0 && startup_expired)
        {
            return WatchdogFailure{
                "decoder_startup_timeout",
                "uart_decoder",
                "UART bytes arrived but no samples were decoded",
            };
        }
        if (last_byte > last_sample &&
            exceeded(now_ns, last_sample, config.sample_timeout_ns))
        {
            return WatchdogFailure{
                "decoder_sample_timeout",
                "uart_decoder",
                "UART bytes continue but sample decoding stopped",
            };
        }
    }

    return std::nullopt;
}

std::optional<WatchdogFailure> evaluate_watchdog(
    const RuntimeHealth &health,
    const WatchdogConfig &config,
    std::uint64_t now_ns)
{
    if (!config.enabled)
    {
        return std::nullopt;
    }
    const auto input_failure = evaluate_input_watchdog(health, config, now_ns);
    if (input_failure.has_value())
    {
        return input_failure;
    }

    const auto inference_active =
        health.inference_active_since_ns.load(std::memory_order_relaxed);
    if (exceeded(now_ns, inference_active, config.inference_timeout_ns))
    {
        return WatchdogFailure{
            "inference_timeout",
            "tensorrt",
            "An inference call exceeded its deadline",
        };
    }

    const auto inference_expected =
        health.first_inference_expected_ns.load(std::memory_order_relaxed);
    const auto inference_completed =
        health.last_inference_completed_ns.load(std::memory_order_relaxed);
    if (inference_expected > 0 && inference_completed == 0 &&
        exceeded(now_ns, inference_expected, config.inference_timeout_ns))
    {
        return WatchdogFailure{
            "inference_startup_timeout",
            "inference",
            "Enough samples arrived for inference, but none completed",
        };
    }

    const auto queue_depth = health.queue_depth.load(std::memory_order_relaxed);
    const auto last_consumed =
        health.last_sample_consumed_ns.load(std::memory_order_relaxed);
    if (queue_depth > 0 && last_consumed > 0 &&
        exceeded(now_ns, last_consumed, config.inference_timeout_ns))
    {
        return WatchdogFailure{
            "inference_consumer_stall",
            "inference",
            "The inference worker stopped consuming queued samples",
        };
    }

    const auto output_active =
        health.output_active_since_ns.load(std::memory_order_relaxed);
    if (exceeded(now_ns, output_active, config.output_timeout_ns))
    {
        return WatchdogFailure{
            "trigger_output_timeout",
            "trigger_output",
            "A trigger output operation exceeded its deadline",
        };
    }

    const auto output_expected =
        health.first_output_expected_ns.load(std::memory_order_relaxed);
    const auto output_completed =
        health.last_output_completed_ns.load(std::memory_order_relaxed);
    if (output_expected > 0 && output_completed == 0 &&
        exceeded(now_ns, output_expected, config.output_timeout_ns))
    {
        return WatchdogFailure{
            "trigger_output_startup_timeout",
            "trigger_output",
            "An output record was queued, but none completed",
        };
    }

    const auto output_depth =
        health.output_queue_depth.load(std::memory_order_relaxed);
    if (output_depth > 0 && output_completed > 0 &&
        exceeded(now_ns, output_completed, config.output_timeout_ns))
    {
        return WatchdogFailure{
            "trigger_output_consumer_stall",
            "trigger_output",
            "The output worker stopped consuming queued records",
        };
    }
    return std::nullopt;
}

void update_atomic_max(
    std::atomic<std::uint64_t> &destination,
    std::uint64_t value) noexcept
{
    auto current = destination.load(std::memory_order_relaxed);
    while (current < value &&
           !destination.compare_exchange_weak(
               current,
               value,
               std::memory_order_relaxed,
               std::memory_order_relaxed))
    {
    }
}

} // namespace dkfz::live
