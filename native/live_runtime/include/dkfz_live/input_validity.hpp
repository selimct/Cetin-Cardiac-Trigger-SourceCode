/*
 * Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
 * SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
 * Developed at the German Cancer Research Center (DKFZ).
 */

#pragma once

#include "dkfz_uart/uart_stream.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>

namespace dkfz::live
{

inline constexpr std::uint32_t kInvalidPulsSentinel = 1U << 0U;
inline constexpr std::uint32_t kInvalidEcgBaseline = 1U << 1U;
inline constexpr std::uint32_t kInvalidFrozenSignal = 1U << 2U;
inline constexpr std::uint32_t kInvalidUartSilence = 1U << 3U;
inline constexpr std::uint32_t kInvalidDecoderStall = 1U << 4U;
inline constexpr std::uint32_t kInvalidPulsBaseline = 1U << 5U;

struct InputResetConfig
{
    bool enabled = true;
    std::uint64_t puls_baseline_reset_ms = 2000;
    std::uint64_t ecg_baseline_reset_ms = 2000;
    std::uint64_t stuck_signal_reset_ms = 2000;
    std::uint64_t recovery_ms = 500;
    std::uint16_t invalid_sensor_value = 4095;
    std::uint16_t puls_baseline_value = 2048;
    std::uint16_t ecg_baseline_value = 2048;
};

struct InputValidityDecision
{
    bool accept_sample = true;
    bool reset_required = false;
    bool recovered = false;
    std::uint32_t reasons = 0;
};

class InputValidityTracker
{
  public:
    InputValidityTracker(
        InputResetConfig config,
        std::uint32_t sample_rate_hz,
        bool check_puls = true);

    void require_recovery() noexcept;
    InputValidityDecision observe(const dkfz::uart::DecodedSample &sample);

  private:
    InputResetConfig config_;
    bool check_puls_ = true;
    std::size_t puls_baseline_samples_required_ = 0;
    std::size_t puls_baseline_samples_ = 0;
    std::size_t ecg_baseline_samples_required_ = 0;
    std::size_t ecg_baseline_samples_ = 0;
    std::size_t stuck_signal_samples_required_ = 0;
    std::size_t stuck_signal_samples_ = 0;
    std::size_t recovery_samples_required_ = 0;
    std::size_t recovery_samples_ = 0;
    std::size_t recovery_signal_changes_ = 0;
    std::optional<std::array<std::uint16_t, 3>> previous_signals_;
    bool invalid_ = false;
};

} // namespace dkfz::live
