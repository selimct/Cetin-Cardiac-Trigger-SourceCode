/*
 * Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
 * SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
 * Developed at the German Cancer Research Center (DKFZ).
 */

#pragma once

#include "dkfz_live/deployment_manifest.hpp"
#include "dkfz_live/inference_engine.hpp"
#include "dkfz_live/input_validity.hpp"
#include "dkfz_live/runtime_output.hpp"
#include "dkfz_live/runtime_health.hpp"
#include "dkfz_live/thread_scheduling.hpp"
#include "dkfz_uart/uart_stream.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iosfwd>
#include <memory>
#include <string>

namespace dkfz::live
{

enum class IntegratedExitCode : int
{
    success = 0,
    configuration = 2,
    acquisition = 3,
    queue_overflow = 4,
    inference = 5,
    watchdog = 6,
    output = 7,
};

struct IntegratedRuntimeConfig
{
    dkfz::uart::ByteSourceConfig source;
    dkfz::uart::UartSignalOffsets uart_offsets;
    std::uint32_t sample_rate_hz = 400;
    std::size_t queue_capacity = 2048;
    std::uint64_t max_samples = 0;
    std::uint64_t status_interval_ns = 5'000'000'000ULL;
    std::uint64_t paused_status_interval_ns = 60'000'000'000ULL;
    std::string sid = "0";
    std::filesystem::path threshold_override_path;
    bool early_emission = true;
    bool emit_probabilities = false;
    ThreadSchedulingConfig scheduling;
    InputResetConfig input_reset;
    RuntimeOutputConfig output;
    bool hard_exit_on_watchdog = true;
    WatchdogConfig watchdog;
};

struct IntegratedRuntimeSummary
{
    IntegratedExitCode exit_code = IntegratedExitCode::success;
    std::uint64_t bytes_read = 0;
    std::uint64_t samples_decoded = 0;
    std::uint64_t samples_consumed = 0;
    std::uint64_t frames_inferred = 0;
    std::uint64_t trigger_events = 0;
    std::uint64_t malformed_groups = 0;
    std::uint64_t bytes_discarded = 0;
    std::uint64_t max_inference_duration_ns = 0;
    std::uint64_t missed_frame_deadlines = 0;
    std::uint64_t max_frame_end_to_completion_ns = 0;
    std::size_t queue_high_water = 0;
    std::uint64_t output_records = 0;
    std::uint64_t gpio_pulses = 0;
    std::uint64_t invalid_samples = 0;
    std::uint64_t input_state_resets = 0;
    std::size_t output_queue_high_water = 0;
};

class IntegratedLiveRuntime
{
  public:
    IntegratedLiveRuntime(
        DeploymentManifest manifest,
        std::unique_ptr<LogitEngine> engine,
        IntegratedRuntimeConfig config,
        std::ostream &event_output,
        std::ostream &diagnostic_output);

    IntegratedRuntimeSummary run(std::atomic<bool> &stop_requested);

  private:
    DeploymentManifest manifest_;
    std::unique_ptr<LogitEngine> engine_;
    IntegratedRuntimeConfig config_;
    std::ostream &event_output_;
    std::ostream &diagnostic_output_;
};

} // namespace dkfz::live
