/*
 * Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
 * SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
 * Developed at the German Cancer Research Center (DKFZ).
 */

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

namespace dkfz::live
{

struct RuntimeHealth
{
    std::uint64_t started_ns = 0;
    std::atomic<std::uint64_t> last_byte_ns{0};
    std::atomic<std::uint64_t> last_sample_ns{0};
    std::atomic<std::uint64_t> last_sample_consumed_ns{0};
    std::atomic<std::uint64_t> inference_active_since_ns{0};
    std::atomic<std::uint64_t> first_inference_expected_ns{0};
    std::atomic<std::uint64_t> last_inference_completed_ns{0};
    std::atomic<std::uint64_t> last_inference_duration_ns{0};
    std::atomic<std::uint64_t> max_inference_duration_ns{0};
    std::atomic<std::uint64_t> bytes_read{0};
    std::atomic<std::uint64_t> samples_decoded{0};
    std::atomic<std::uint64_t> samples_consumed{0};
    std::atomic<std::uint64_t> frames_inferred{0};
    std::atomic<std::uint64_t> trigger_events{0};
    std::atomic<std::uint64_t> invalid_samples{0};
    std::atomic<std::uint64_t> input_state_resets{0};
    std::atomic<std::uint64_t> malformed_groups{0};
    std::atomic<std::uint64_t> bytes_discarded{0};
    std::atomic<std::uint64_t> queue_depth{0};
    std::atomic<std::uint64_t> queue_high_water{0};
    std::atomic<std::uint64_t> first_output_expected_ns{0};
    std::atomic<std::uint64_t> output_active_since_ns{0};
    std::atomic<std::uint64_t> last_output_completed_ns{0};
    std::atomic<std::uint64_t> output_records{0};
    std::atomic<std::uint64_t> gpio_pulses{0};
    std::atomic<std::uint64_t> output_queue_depth{0};
    std::atomic<std::uint64_t> output_queue_high_water{0};
    std::atomic<bool> acquisition_done{false};
    std::atomic<bool> inference_done{false};
    std::atomic<bool> output_done{false};
};

struct WatchdogConfig
{
    bool enabled = true;
    bool monitor_uart = true;
    std::uint64_t startup_grace_ns = 5'000'000'000ULL;
    std::uint64_t uart_timeout_ns = 2'000'000'000ULL;
    std::uint64_t sample_timeout_ns = 2'000'000'000ULL;
    std::uint64_t inference_timeout_ns = 1'000'000'000ULL;
    std::uint64_t output_timeout_ns = 1'000'000'000ULL;
};

struct WatchdogFailure
{
    std::string code;
    std::string component;
    std::string message;
};

std::optional<WatchdogFailure> evaluate_watchdog(
    const RuntimeHealth &health,
    const WatchdogConfig &config,
    std::uint64_t now_ns);

std::optional<WatchdogFailure> evaluate_input_watchdog(
    const RuntimeHealth &health,
    const WatchdogConfig &config,
    std::uint64_t now_ns);

void update_atomic_max(
    std::atomic<std::uint64_t> &destination,
    std::uint64_t value) noexcept;

} // namespace dkfz::live
