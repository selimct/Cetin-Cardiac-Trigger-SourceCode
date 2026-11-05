/*
 * Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
 * SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
 * Developed at the German Cancer Research Center (DKFZ).
 */

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <functional>
#include <optional>
#include <vector>

namespace dkfz::uart
{

inline constexpr std::size_t kGroupBytes = 26;
inline constexpr std::size_t kSamplesPerGroup = 4;
inline constexpr std::size_t kInterPacketBytes = 3;

struct DecodedSample
{
    std::uint64_t raw_index = 0;
    double time_ms = 0.0;
    std::uint64_t monotonic_ns = 0;
    std::uint64_t realtime_ns = 0;
    std::uint16_t ecg1_raw = 0;
    std::uint16_t ecg2_raw = 0;
    std::uint16_t puls_raw = 0;
    std::uint8_t ext = 0;
    std::uint8_t pr_sign = 0;
    bool puls_updated = false;
    std::uint64_t source_packet = 0;
    std::size_t sample_in_packet = 0;
};

struct UartSignalOffsets
{
    double ecg1 = 2048.0;
    double ecg2 = 2048.0;
    double puls = 2048.0;
};

struct CalibratedSignals
{
    double ecg1 = 0.0;
    double ecg2 = 0.0;
    double ecg3 = 0.0;
    double puls = 0.0;
};

CalibratedSignals calibrate_signals(
    const DecodedSample &sample,
    const UartSignalOffsets &offsets = {});

struct DecoderStats
{
    std::uint64_t bytes_discarded = 0;
    std::uint64_t groups = 0;
    std::uint64_t malformed_groups = 0;
    std::uint64_t samples = 0;
    std::uint64_t pulse_queue_high_water = 0;
};

using SampleCallback = std::function<bool(const DecodedSample &)>;

class UartByteDecoder
{
  public:
    UartByteDecoder(
        std::uint32_t sample_rate_hz,
        SampleCallback callback,
        std::optional<std::uint64_t> monotonic_start_ns = std::nullopt,
        std::optional<std::uint64_t> realtime_start_ns = std::nullopt);

    void process(const std::uint8_t *data, std::size_t size);
    void process(const std::vector<std::uint8_t> &data);
    bool stopped() const noexcept { return stopped_; }
    const DecoderStats &stats() const noexcept { return stats_; }

  private:
    bool has_group_anchors(std::size_t start) const;
    void consume_prefix(std::size_t count);
    void discard_prefix(std::size_t count);
    void parse_available();
    void accept_front_group();
    void accept_group(const std::uint8_t *bytes);
    bool select_pulse(std::uint16_t fallback);
    void emit_sample(
        const std::uint8_t *bytes,
        std::size_t offset,
        std::uint64_t group_index,
        std::size_t sample_in_group,
        std::uint8_t pr_sign,
        std::uint16_t pulse_fallback);

    std::uint32_t sample_rate_hz_;
    SampleCallback callback_;
    std::uint64_t monotonic_start_ns_;
    std::uint64_t realtime_start_ns_;
    DecoderStats stats_;
    std::vector<std::uint8_t> buffer_;
    bool synchronized_ = false;
    bool ever_synchronized_ = false;
    std::size_t resync_discarded_ = 0;
    std::deque<std::uint16_t> pending_pulses_;
    std::optional<std::uint16_t> current_pulse_;
    std::uint32_t pulse_samples_remaining_ = 0;
    bool stopped_ = false;
};

struct ByteSourceConfig
{
    std::filesystem::path device = "/dev/ttyTHS1";
    std::uint32_t baud_rate = 38'600;
    std::filesystem::path capture_path;
    std::uint64_t max_capture_bytes = 5'000'000'000ULL;
    std::filesystem::path replay_path;
};

struct ByteSourceStats
{
    std::uint64_t bytes_read = 0;
    std::uint64_t capture_bytes = 0;
    bool capture_limit_reached = false;
};

using ByteChunkCallback =
    std::function<bool(const std::uint8_t *, std::size_t)>;
using SourceProgressCallback =
    std::function<void(const ByteSourceStats &)>;

ByteSourceStats run_byte_source(
    const ByteSourceConfig &config,
    const ByteChunkCallback &callback,
    std::atomic<bool> &stop_requested,
    const SourceProgressCallback &progress_callback = {});

std::uint64_t monotonic_clock_ns();
std::uint64_t realtime_clock_ns();

} // namespace dkfz::uart
