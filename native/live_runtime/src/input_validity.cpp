/*
 * Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
 * SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
 * Developed at the German Cancer Research Center (DKFZ).
 */

#include "dkfz_live/input_validity.hpp"

#include <limits>
#include <stdexcept>
#include <string>

namespace dkfz::live
{
namespace
{

std::size_t duration_samples(
    std::uint64_t duration_ms,
    std::uint32_t sample_rate_hz,
    const char *description)
{
    if (duration_ms >
        std::numeric_limits<std::uint64_t>::max() / sample_rate_hz)
    {
        throw std::invalid_argument(
            std::string(description) + " duration is too large");
    }
    const auto sample_milliseconds = duration_ms * sample_rate_hz;
    const auto required =
        sample_milliseconds / 1000ULL +
        (sample_milliseconds % 1000ULL != 0 ? 1ULL : 0ULL);
    if (required > std::numeric_limits<std::size_t>::max())
    {
        throw std::invalid_argument(
            std::string(description) + " sample count is too large");
    }
    return static_cast<std::size_t>(required);
}

std::array<std::uint16_t, 3> signal_values(
    const dkfz::uart::DecodedSample &sample)
{
    return {
        sample.ecg1_raw,
        sample.ecg2_raw,
        sample.puls_raw,
    };
}

} // namespace

InputValidityTracker::InputValidityTracker(
    InputResetConfig config,
    std::uint32_t sample_rate_hz,
    bool check_puls)
    : config_(config), check_puls_(check_puls)
{
    if (sample_rate_hz == 0)
    {
        throw std::invalid_argument(
            "InputValidityTracker sample rate must be positive");
    }
    puls_baseline_samples_required_ = duration_samples(
        config_.puls_baseline_reset_ms, sample_rate_hz, "Pulse baseline reset");
    ecg_baseline_samples_required_ = duration_samples(
        config_.ecg_baseline_reset_ms, sample_rate_hz, "ECG baseline reset");
    stuck_signal_samples_required_ = duration_samples(
        config_.stuck_signal_reset_ms, sample_rate_hz, "Stuck signal reset");
    recovery_samples_required_ = duration_samples(
        config_.recovery_ms, sample_rate_hz, "Input recovery");
}

void InputValidityTracker::require_recovery() noexcept
{
    if (!config_.enabled)
    {
        return;
    }
    invalid_ = true;
    recovery_samples_ = 0;
    recovery_signal_changes_ = 0;
}

InputValidityDecision InputValidityTracker::observe(
    const dkfz::uart::DecodedSample &sample)
{
    if (!config_.enabled)
    {
        return {};
    }

    const auto signals = signal_values(sample);
    const bool signal_changed =
        !previous_signals_.has_value() || *previous_signals_ != signals;
    previous_signals_ = signals;
    if (signal_changed)
    {
        stuck_signal_samples_ = 1;
    }
    else if (stuck_signal_samples_ < std::numeric_limits<std::size_t>::max())
    {
        ++stuck_signal_samples_;
    }

    const bool at_puls_baseline =
        check_puls_ && sample.puls_raw == config_.puls_baseline_value;
    if (at_puls_baseline)
    {
        if (puls_baseline_samples_ < std::numeric_limits<std::size_t>::max())
        {
            ++puls_baseline_samples_;
        }
    }
    else
    {
        puls_baseline_samples_ = 0;
    }

    const bool at_ecg_baseline =
        sample.ecg1_raw == config_.ecg_baseline_value &&
        sample.ecg2_raw == config_.ecg_baseline_value;
    if (at_ecg_baseline)
    {
        if (ecg_baseline_samples_ < std::numeric_limits<std::size_t>::max())
        {
            ++ecg_baseline_samples_;
        }
    }
    else
    {
        ecg_baseline_samples_ = 0;
    }

    std::uint32_t reasons = 0;
    if (check_puls_ && sample.puls_raw == config_.invalid_sensor_value)
    {
        reasons |= kInvalidPulsSentinel;
    }
    if (puls_baseline_samples_required_ > 0 &&
        puls_baseline_samples_ >= puls_baseline_samples_required_)
    {
        reasons |= kInvalidPulsBaseline;
    }
    if (ecg_baseline_samples_required_ > 0 &&
        ecg_baseline_samples_ >= ecg_baseline_samples_required_)
    {
        reasons |= kInvalidEcgBaseline;
    }
    if (stuck_signal_samples_required_ > 0 &&
        stuck_signal_samples_ >= stuck_signal_samples_required_)
    {
        reasons |= kInvalidFrozenSignal;
    }

    const bool invalid_now = reasons != 0;
    InputValidityDecision result;
    result.recovered = !invalid_now && invalid_;
    result.reasons = reasons;
    if (invalid_now)
    {
        result.accept_sample = false;
        result.reset_required = !invalid_;
        invalid_ = true;
        recovery_samples_ = 0;
        recovery_signal_changes_ = 0;
        result.recovered = false;
        return result;
    }

    if (invalid_)
    {
        if (recovery_samples_ < std::numeric_limits<std::size_t>::max())
        {
            ++recovery_samples_;
        }
        if (signal_changed &&
            recovery_signal_changes_ < std::numeric_limits<std::size_t>::max())
        {
            ++recovery_signal_changes_;
        }
        const bool duration_met =
            recovery_samples_required_ == 0 ||
            recovery_samples_ >= recovery_samples_required_;
        const bool movement_met =
            recovery_samples_required_ == 0 ||
            stuck_signal_samples_required_ == 0 ||
            recovery_signal_changes_ >= 2;
        if (!duration_met || !movement_met)
        {
            result.accept_sample = false;
            result.recovered = false;
            return result;
        }
        invalid_ = false;
        result.recovered = true;
    }
    result.accept_sample = true;
    return result;
}

} // namespace dkfz::live
