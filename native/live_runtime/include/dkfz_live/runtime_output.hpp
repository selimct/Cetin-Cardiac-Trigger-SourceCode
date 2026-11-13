/*
 * Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
 * SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
 * Developed at the German Cancer Research Center (DKFZ).
 */

#pragma once

#include "dkfz_live/live_core.hpp"

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <memory>
#include <optional>
#include <string>

namespace dkfz::live
{

enum class RuntimeOutputMode
{
    none,
    jsonl,
    gpio,
    gpio_jsonl,
};

RuntimeOutputMode parse_runtime_output_mode(const std::string &value);
std::string runtime_output_mode_name(RuntimeOutputMode mode);
bool runtime_output_has_jsonl(RuntimeOutputMode mode) noexcept;
bool runtime_output_has_gpio(RuntimeOutputMode mode) noexcept;

struct GpioTriggerOutputConfig
{
    // A /dev/gpiochipN path or a kernel GPIO chip label.
    std::string chip;
    std::optional<std::uint32_t> line_offset;
    bool active_high = true;
    std::uint64_t pulse_width_ns = 10'000'000ULL;
    std::uint64_t idle_width_ns = 10'000'000ULL;
    std::string consumer = "dkfz-live-trigger";
};

struct RuntimeOutputConfig
{
    RuntimeOutputMode mode = RuntimeOutputMode::none;
    std::size_t queue_capacity = 256;
    bool allow_gpio_during_replay = false;
    GpioTriggerOutputConfig gpio;
};

void validate_runtime_output_config(
    const RuntimeOutputConfig &config,
    bool replay);

struct ProbabilityOutputRecord
{
    std::string sid;
    std::int64_t frame_idx = 0;
    std::int64_t start_sample_idx = 0;
    std::int64_t end_sample_idx = 0;
    std::optional<double> time_ms;
    float model_output = 0.0F;
    std::optional<float> recovery_probability;
    bool recovery_emitted_now = false;
    float probability = 0.0F;
    bool trigger_emitted_now = false;
    std::uint64_t processing_duration_ns = 0;
    std::uint64_t preprocessing_ns = 0, window_assembly_ns = 0, inference_ns = 0, trigger_ns = 0;
    std::uint64_t cpu_thread_ns = 0, sample_queue_wait_ns = 0;
    std::uint64_t gpu_upload_ns = 0, gpu_compute_ns = 0, gpu_download_ns = 0;
    bool cuda_graph = false;
    std::uint64_t frame_end_to_completion_ns = 0;
    std::uint64_t completed_monotonic_ns = 0;
    std::uint64_t decoded_monotonic_ns = 0;
    std::uint64_t backlog_samples = 0;
    bool missed_deadline = false;
    std::optional<std::int64_t> observed_frame_idx;
    std::int64_t observed_start_sample_idx = 0;
    std::int64_t observed_end_sample_idx = 0;
    std::optional<double> observed_time_ms;
};

enum class RuntimeOutputRecordKind
{
    probability,
    trigger,
};

struct RuntimeOutputRecord
{
    RuntimeOutputRecordKind kind = RuntimeOutputRecordKind::probability;
    ProbabilityOutputRecord probability;
    TriggerEvent trigger;

    static RuntimeOutputRecord from_probability(ProbabilityOutputRecord value);
    static RuntimeOutputRecord from_trigger(TriggerEvent value);
};

class TriggerLine
{
  public:
    virtual ~TriggerLine() = default;
    virtual void set_active(bool active) = 0;
};

class RuntimeOutputSink
{
  public:
    virtual ~RuntimeOutputSink() = default;
    virtual void write(const RuntimeOutputRecord &record) = 0;
    virtual void close() = 0;
};

std::unique_ptr<TriggerLine> make_linux_gpio_trigger_line(
    const GpioTriggerOutputConfig &config);

std::unique_ptr<RuntimeOutputSink> make_runtime_output_sink(
    RuntimeOutputConfig config,
    std::ostream &jsonl_output,
    std::unique_ptr<TriggerLine> trigger_line = nullptr);

} // namespace dkfz::live
