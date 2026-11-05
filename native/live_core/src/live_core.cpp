/*
 * Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
 * SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
 * Developed at the German Cancer Research Center (DKFZ).
 */

#include "dkfz_live/live_core.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <tuple>

namespace dkfz::live
{
namespace
{

constexpr double kDefaultSignalQualityMinIqr = 0.05;

double range_of(const std::vector<double> &values)
{
    if (values.empty())
    {
        return 0.0;
    }
    const auto [minimum, maximum] = std::minmax_element(values.begin(), values.end());
    return *maximum - *minimum;
}

double median_of(const std::deque<double> &values)
{
    if (values.empty())
    {
        return 0.0;
    }
    std::vector<double> sorted(values.begin(), values.end());
    std::sort(sorted.begin(), sorted.end());
    const auto middle = sorted.size() / 2;
    if (sorted.size() % 2 != 0)
    {
        return sorted[middle];
    }
    return 0.5 * (sorted[middle - 1] + sorted[middle]);
}

void append_bounded(
    std::deque<double> &values,
    double value,
    std::size_t capacity)
{
    if (capacity == 0)
    {
        return;
    }
    if (values.size() == capacity)
    {
        values.pop_front();
    }
    values.push_back(value);
}

std::optional<double> group_chi2(
    const std::map<std::string, double> &components,
    const std::vector<std::string> &columns,
    const SignalQualityStats &stats,
    double eps)
{
    if (columns.empty())
    {
        return std::nullopt;
    }

    double sum = 0.0;
    for (const auto &column : columns)
    {
        const auto value_it = components.find(column);
        const auto median_it = stats.median.find(column);
        const auto iqr_it = stats.iqr.find(column);
        const auto min_iqr_it = stats.min_iqr.find(column);
        if (value_it == components.end() ||
            median_it == stats.median.end() ||
            iqr_it == stats.iqr.end())
        {
            return std::nullopt;
        }
        if (min_iqr_it == stats.min_iqr.end())
        {
            throw std::invalid_argument(
                "signal_quality_stats min_iqr is missing component '" +
                column + "'.");
        }

        const auto scale = std::max({iqr_it->second, min_iqr_it->second, eps});
        auto z = (value_it->second - median_it->second) / scale;
        if (stats.z_clip.has_value() && *stats.z_clip > 0.0)
        {
            z = std::clamp(z, -*stats.z_clip, *stats.z_clip);
        }
        sum += z * z;
    }
    return sum / static_cast<double>(columns.size());
}

double signal_quality_from_badness(double badness, const std::string &transform)
{
    badness = std::max(badness, 0.0);
    if (transform == "log1p_weighted_chi2_iqr")
    {
        return std::log1p(badness);
    }
    return std::exp(-0.5 * badness);
}

} // namespace

EcgSlopeMode parse_ecg_slope_mode(const std::string &value)
{
    if (value == "none")
    {
        return EcgSlopeMode::none;
    }
    if (value == "fast")
    {
        return EcgSlopeMode::fast;
    }
    if (value == "fast_slow")
    {
        return EcgSlopeMode::fast_slow;
    }
    if (value == "slow")
    {
        return EcgSlopeMode::slow;
    }
    throw std::invalid_argument(
        "ecg_slope_mode must be one of: none, fast, slow, fast_slow.");
}

SignalQualityMode parse_signal_quality_mode(const std::string &value)
{
    if (value == "none")
    {
        return SignalQualityMode::none;
    }
    if (value == "ecg")
    {
        return SignalQualityMode::ecg;
    }
    if (value == "ecg_puls")
    {
        return SignalQualityMode::ecg_puls;
    }
    throw std::invalid_argument(
        "signal_quality_mode must be one of: none, ecg, ecg_puls.");
}

std::string to_string(EcgSlopeMode value)
{
    switch (value)
    {
    case EcgSlopeMode::none:
        return "none";
    case EcgSlopeMode::fast:
        return "fast";
    case EcgSlopeMode::fast_slow:
        return "fast_slow";
    case EcgSlopeMode::slow:
        return "slow";
    }
    throw std::invalid_argument("Unknown ECG slope mode.");
}

std::string to_string(SignalQualityMode value)
{
    switch (value)
    {
    case SignalQualityMode::none:
        return "none";
    case SignalQualityMode::ecg:
        return "ecg";
    case SignalQualityMode::ecg_puls:
        return "ecg_puls";
    }
    throw std::invalid_argument("Unknown signal quality mode.");
}

const std::vector<std::string> &ecg_quality_component_names()
{
    static const std::vector<std::string> names{
        "_signal_quality_ecg_log_rough",
        "_signal_quality_ecg_log_curvature",
        "_signal_quality_ecg_flat",
        "_signal_quality_ecg_log_scale_jump",
    };
    return names;
}

const std::vector<std::string> &puls_quality_component_names()
{
    static const std::vector<std::string> names{
        "_signal_quality_puls_log_step",
        "_signal_quality_puls_log_curvature",
        "_signal_quality_puls_flat",
        "_signal_quality_puls_log_outlier",
    };
    return names;
}

std::vector<std::string> signal_quality_component_names(
    SignalQualityMode mode,
    bool include_puls)
{
    if (mode == SignalQualityMode::none)
    {
        return {};
    }
    auto result = ecg_quality_component_names();
    if (mode == SignalQualityMode::ecg_puls && include_puls)
    {
        const auto &pulse = puls_quality_component_names();
        result.insert(result.end(), pulse.begin(), pulse.end());
    }
    return result;
}

std::vector<std::string> live_feature_columns(
    std::size_t frame_size_raw,
    bool include_puls,
    const std::vector<std::string> &ecg_columns,
    EcgSlopeMode slope_mode,
    SignalQualityMode quality_mode)
{
    if (ecg_columns.size() < 2)
    {
        throw std::invalid_argument("At least ECG1 and ECG2 are required.");
    }

    std::vector<std::string> columns;
    columns.reserve(ecg_columns.size() * frame_size_raw + 6);
    for (const auto &ecg_column : ecg_columns)
    {
        for (std::size_t sample = 0; sample < frame_size_raw; ++sample)
        {
            columns.push_back(
                ecg_column + "_t" + std::to_string(sample));
        }
    }
    if (slope_mode == EcgSlopeMode::fast)
    {
        columns.insert(columns.end(), {"ecg1_slope_fast", "ecg2_slope_fast"});
    }
    else if (slope_mode == EcgSlopeMode::slow)
    {
        columns.insert(columns.end(), {"ecg1_slope_slow", "ecg2_slope_slow"});
    }
    else if (slope_mode == EcgSlopeMode::fast_slow)
    {
        columns.insert(
            columns.end(),
            {
                "ecg1_slope_fast",
                "ecg1_slope_slow",
                "ecg2_slope_fast",
                "ecg2_slope_slow",
            });
    }
    if (include_puls)
    {
        columns.push_back("puls");
    }
    if (quality_mode != SignalQualityMode::none)
    {
        columns.push_back(kSignalQualityFeature);
    }
    return columns;
}

double signal_quality_feature_from_components(
    const std::map<std::string, double> &components,
    const SignalQualityStats &stats,
    double eps)
{
    if (components.empty())
    {
        return 1.0;
    }

    if (!stats.present)
    {
        double sum = 0.0;
        std::size_t count = 0;
        for (const auto &[name, value] : components)
        {
            static_cast<void>(name);
            if (std::isfinite(value))
            {
                sum += value * value;
                ++count;
            }
        }
        const auto badness = count == 0 ? 0.0 : sum / static_cast<double>(count);
        return signal_quality_from_badness(badness, "");
    }

    if (stats.min_iqr.empty())
    {
        throw std::invalid_argument(
            "signal_quality_stats are missing 'min_iqr'.");
    }

    const auto &ecg_columns = stats.ecg_component_cols.empty()
        ? ecg_quality_component_names()
        : stats.ecg_component_cols;
    const auto ecg_badness = group_chi2(components, ecg_columns, stats, eps);
    if (!ecg_badness.has_value())
    {
        return 1.0;
    }

    const auto puls_badness = group_chi2(
        components,
        stats.puls_component_cols,
        stats,
        eps);
    auto badness = *ecg_badness;
    if (puls_badness.has_value())
    {
        const auto total = std::max(stats.ecg_weight + stats.puls_weight, eps);
        badness =
            (stats.ecg_weight * *ecg_badness +
             stats.puls_weight * *puls_badness) /
            total;
    }
    return signal_quality_from_badness(badness, stats.transform);
}

RollingPairNormQuantileNormalizer::RollingPairNormQuantileNormalizer(
    double q,
    std::size_t window_samples,
    double eps,
    double warmup_scale)
    : q_(q),
      window_samples_(window_samples),
      eps_(eps),
      warmup_scale_(warmup_scale),
      last_scale_(std::max(warmup_scale, eps))
{
    if (!(q_ > 0.0 && q_ <= 1.0))
    {
        throw std::invalid_argument("q must be in (0, 1].");
    }
    if (window_samples_ == 0)
    {
        throw std::invalid_argument("window_samples must be positive.");
    }
}

void RollingPairNormQuantileNormalizer::reset()
{
    norm_history_.clear();
    last_scale_ = std::max(warmup_scale_, eps_);
}

double RollingPairNormQuantileNormalizer::current_quantile() const
{
    if (norm_history_.empty())
    {
        return std::max(warmup_scale_, eps_);
    }

    std::vector<double> values(norm_history_.begin(), norm_history_.end());
    std::sort(values.begin(), values.end());
    const auto position = q_ * static_cast<double>(values.size() - 1);
    const auto lower = static_cast<std::size_t>(std::floor(position));
    const auto upper = static_cast<std::size_t>(std::ceil(position));
    const auto fraction = position - static_cast<double>(lower);
    const auto quantile = values[lower] + fraction * (values[upper] - values[lower]);
    return std::max(quantile, eps_);
}

std::array<std::optional<double>, 3>
RollingPairNormQuantileNormalizer::step_with_extra(
    double ecg1,
    double ecg2,
    std::optional<double> ecg3)
{
    const auto scale = current_quantile();
    last_scale_ = scale;
    const auto ecg1_out = ecg1 / scale;
    const auto ecg2_out = ecg2 / scale;
    const auto ecg3_out = ecg3.has_value()
        ? std::optional<double>(*ecg3 / scale)
        : std::nullopt;

    if (norm_history_.size() == window_samples_)
    {
        norm_history_.pop_front();
    }
    norm_history_.push_back(std::sqrt(ecg1 * ecg1 + ecg2 * ecg2));
    return {ecg1_out, ecg2_out, ecg3_out};
}

std::pair<double, double> RollingPairNormQuantileNormalizer::step(
    double ecg1,
    double ecg2)
{
    const auto result = step_with_extra(ecg1, ecg2, std::nullopt);
    return {*result[0], *result[1]};
}

RollingScalarQuantileNormalizer::RollingScalarQuantileNormalizer(
    double q_low,
    double q_high,
    std::size_t window_frames,
    double eps,
    double warmup_value,
    bool clip)
    : q_low_(q_low),
      q_high_(q_high),
      window_frames_(window_frames),
      eps_(eps),
      warmup_value_(warmup_value),
      clip_(clip)
{
    if (!(0.0 <= q_low_ && q_low_ < q_high_ && q_high_ <= 1.0))
    {
        throw std::invalid_argument(
            "Require 0 <= q_low < q_high <= 1.");
    }
    if (window_frames_ == 0)
    {
        throw std::invalid_argument("window_frames must be positive.");
    }
    if (eps_ <= 0.0)
    {
        throw std::invalid_argument("eps must be positive.");
    }
    if (!std::isfinite(warmup_value_))
    {
        throw std::invalid_argument("warmup_value must be finite.");
    }
}

void RollingScalarQuantileNormalizer::reset()
{
    history_.clear();
}

double RollingScalarQuantileNormalizer::step(double value)
{
    auto normalized = warmup_value_;
    if (!history_.empty())
    {
        std::vector<double> values(history_.begin(), history_.end());
        std::sort(values.begin(), values.end());
        const auto quantile = [&values](double q)
        {
            const auto position = q * static_cast<double>(values.size() - 1);
            const auto lower = static_cast<std::size_t>(std::floor(position));
            const auto upper = static_cast<std::size_t>(std::ceil(position));
            const auto fraction = position - static_cast<double>(lower);
            return values[lower] + fraction * (values[upper] - values[lower]);
        };
        const auto low = quantile(q_low_);
        const auto high = quantile(q_high_);
        const auto scale = high - low;
        if (scale > eps_)
        {
            normalized = (value - low) / scale;
        }
    }

    if (clip_)
    {
        normalized = std::clamp(normalized, -0.1, 1.1);
    }
    if (history_.size() == window_frames_)
    {
        history_.pop_front();
    }
    history_.push_back(value);
    return normalized;
}

SignalQualityTracker::SignalQualityTracker(
    SignalQualityMode mode,
    SignalQualityStats stats,
    double eps,
    double ecg_flat_threshold,
    double puls_flat_threshold,
    std::size_t puls_flat_window_frames,
    std::size_t puls_median_window_frames)
    : mode_(mode),
      stats_(std::move(stats)),
      eps_(eps),
      ecg_flat_threshold_(ecg_flat_threshold),
      puls_flat_threshold_(puls_flat_threshold),
      puls_flat_window_frames_(std::max<std::size_t>(2, puls_flat_window_frames)),
      puls_median_window_frames_(puls_median_window_frames)
{
}

void SignalQualityTracker::reset()
{
    prev_ecg_.reset();
    prev_ecg_diff_.reset();
    prev_puls_.reset();
    prev_prev_puls_.reset();
    puls_flat_history_.clear();
    puls_history_.clear();
}

SignalQualitySample SignalQualityTracker::step_ecg_sample(
    double ecg1_norm,
    double ecg2_norm,
    double ecg_scale)
{
    const auto scale = std::max(ecg_scale, eps_);
    double du = 0.0;
    double dv = 0.0;
    if (prev_ecg_.has_value())
    {
        du = ecg1_norm - prev_ecg_->first;
        dv = ecg2_norm - prev_ecg_->second;
    }

    double ddu = 0.0;
    double ddv = 0.0;
    if (prev_ecg_diff_.has_value())
    {
        ddu = du - prev_ecg_diff_->first;
        ddv = dv - prev_ecg_diff_->second;
    }
    prev_ecg_ = std::pair{ecg1_norm, ecg2_norm};
    prev_ecg_diff_ = std::pair{du, dv};
    return {
        du * du + dv * dv,
        ddu * ddu + ddv * ddv,
        ecg1_norm,
        ecg2_norm,
        scale,
    };
}

std::map<std::string, double> SignalQualityTracker::build_components(
    const std::vector<SignalQualitySample> &block,
    std::optional<double> puls_value_norm,
    bool puls_available)
{
    if (!enabled())
    {
        return {};
    }
    if (block.empty())
    {
        throw std::invalid_argument("Cannot build signal quality without ECG samples.");
    }

    double rough_sum = 0.0;
    double curvature_sum = 0.0;
    std::vector<double> ecg1;
    std::vector<double> ecg2;
    ecg1.reserve(block.size());
    ecg2.reserve(block.size());
    for (const auto &sample : block)
    {
        rough_sum += sample[0];
        curvature_sum += sample[1];
        ecg1.push_back(sample[2]);
        ecg2.push_back(sample[3]);
    }
    const auto rough = std::sqrt(rough_sum / static_cast<double>(block.size()));
    const auto curvature =
        std::sqrt(curvature_sum / static_cast<double>(block.size()));
    const auto flat = block.size() < 2
        ? 0.0
        : static_cast<double>(
              range_of(ecg1) + range_of(ecg2) < ecg_flat_threshold_);
    const auto scale_jump = std::abs(std::log(
        std::max(block.back()[4], eps_) /
        std::max(block.front()[4], eps_)));

    const auto &ecg_names = ecg_quality_component_names();
    std::map<std::string, double> components{
        {ecg_names[0], std::log(rough + eps_)},
        {ecg_names[1], std::log(curvature + eps_)},
        {ecg_names[2], flat},
        {ecg_names[3], std::log(scale_jump + eps_)},
    };

    if (wants_pulse() && puls_available && puls_value_norm.has_value())
    {
        const auto pulse = *puls_value_norm;
        double step = 0.0;
        double pulse_curvature = 0.0;
        if (prev_puls_.has_value())
        {
            step = std::abs(pulse - *prev_puls_);
            if (prev_prev_puls_.has_value())
            {
                pulse_curvature = std::abs(
                    pulse - 2.0 * *prev_puls_ + *prev_prev_puls_);
            }
        }

        double pulse_flat = 0.0;
        if (puls_flat_history_.size() + 1 >= puls_flat_window_frames_)
        {
            std::vector<double> flat_window(
                puls_flat_history_.begin(),
                puls_flat_history_.end());
            flat_window.push_back(pulse);
            pulse_flat = static_cast<double>(
                range_of(flat_window) < puls_flat_threshold_);
        }

        const auto pulse_outlier = puls_history_.empty()
            ? 0.0
            : std::abs(pulse - median_of(puls_history_));
        const auto &pulse_names = puls_quality_component_names();
        components.insert({
            {pulse_names[0], std::log(step + eps_)},
            {pulse_names[1], std::log(pulse_curvature + eps_)},
            {pulse_names[2], pulse_flat},
            {pulse_names[3], std::log(pulse_outlier + eps_)},
        });

        prev_prev_puls_ = prev_puls_;
        prev_puls_ = pulse;
        append_bounded(
            puls_flat_history_,
            pulse,
            puls_flat_window_frames_ - 1);
        append_bounded(
            puls_history_,
            pulse,
            std::max<std::size_t>(1, puls_median_window_frames_));
    }
    return components;
}

double SignalQualityTracker::feature_from_components(
    const std::map<std::string, double> &components) const
{
    if (!enabled())
    {
        return 0.0;
    }
    return signal_quality_feature_from_components(components, stats_, eps_);
}

FeatureFramer::FeatureFramer(FeatureFramerConfig config)
    : config_(std::move(config)),
      feature_columns_(live_feature_columns(
          config_.frame_size_raw,
          config_.include_puls,
          config_.include_ecg3
              ? std::vector<std::string>{"ecg1", "ecg2", "ecg3"}
              : std::vector<std::string>{"ecg1", "ecg2"},
          config_.ecg_slope_mode,
          config_.signal_quality_mode)),
      rotation_cos_(std::cos(config_.ecg_rotation_angle)),
      rotation_sin_(std::sin(config_.ecg_rotation_angle)),
      ecg_normalizer_(
          config_.ecg_quantile,
          config_.ecg_norm_window_samples,
          config_.ecg_norm_eps,
          config_.ecg_norm_warmup_scale),
      puls_normalizer_(
          config_.puls_quantile_low,
          config_.puls_quantile_high,
          config_.puls_norm_window_frames,
          config_.puls_norm_eps,
          config_.puls_norm_warmup_value,
          config_.puls_clip),
      quality_tracker_(
          config_.signal_quality_mode,
          config_.signal_quality_stats)
{
    if (config_.frame_size_raw == 0)
    {
        throw std::invalid_argument("frame_size_raw must be positive.");
    }
    if (config_.frame_size_raw < 2 &&
        config_.ecg_slope_mode != EcgSlopeMode::none)
    {
        throw std::invalid_argument("Slope features require frame_size_raw >= 2.");
    }
    if ((config_.ecg_slope_mode == EcgSlopeMode::slow ||
         config_.ecg_slope_mode == EcgSlopeMode::fast_slow) &&
        config_.ecg_slope_slow_window_samples < 2)
    {
        throw std::invalid_argument(
            "ecg_slope_slow_window_samples must be at least 2.");
    }
    if (config_.ecg_slope_mode == EcgSlopeMode::fast ||
        config_.ecg_slope_mode == EcgSlopeMode::fast_slow)
    {
        slope_weights_[config_.frame_size_raw] =
            linear_slope_weights(config_.frame_size_raw);
    }
    if (config_.ecg_slope_mode == EcgSlopeMode::slow ||
        config_.ecg_slope_mode == EcgSlopeMode::fast_slow)
    {
        slope_weights_[config_.ecg_slope_slow_window_samples] =
            linear_slope_weights(config_.ecg_slope_slow_window_samples);
    }
    reset();
}

void FeatureFramer::reset()
{
    raw_block_.clear();
    slope_history_.clear();
    next_sample_idx_ = 0;
    next_frame_idx_ = 0;
    last_sample_idx_.reset();
    ecg_normalizer_.reset();
    puls_normalizer_.reset();
    quality_tracker_.reset();
}

std::vector<double> FeatureFramer::linear_slope_weights(
    std::size_t window_samples)
{
    const auto mean = 0.5 * static_cast<double>(window_samples - 1);
    std::vector<double> centered(window_samples);
    for (std::size_t index = 0; index < window_samples; ++index)
    {
        centered[index] = static_cast<double>(index) - mean;
    }
    const auto denominator = std::inner_product(
        centered.begin(),
        centered.end(),
        centered.begin(),
        0.0);
    for (auto &value : centered)
    {
        value /= denominator;
    }
    return centered;
}

double FeatureFramer::window_slope(
    std::size_t channel,
    std::size_t window_samples) const
{
    if (slope_history_.size() < window_samples)
    {
        return 0.0;
    }
    const auto &weights = slope_weights_.at(window_samples);
    const auto start = slope_history_.size() - window_samples;
    double slope = 0.0;
    for (std::size_t index = 0; index < window_samples; ++index)
    {
        const auto &sample = slope_history_[start + index];
        slope += weights[index] * (channel == 0 ? sample.first : sample.second);
    }
    return slope / ecg_normalizer_.last_scale();
}

std::vector<float> FeatureFramer::build_slope_features() const
{
    if (config_.ecg_slope_mode == EcgSlopeMode::none)
    {
        return {};
    }
    if (config_.ecg_slope_mode == EcgSlopeMode::fast)
    {
        const auto fast1 = window_slope(0, config_.frame_size_raw);
        const auto fast2 = window_slope(1, config_.frame_size_raw);
        return {static_cast<float>(fast1), static_cast<float>(fast2)};
    }
    const auto slow1 = window_slope(
        0,
        config_.ecg_slope_slow_window_samples);
    const auto slow2 = window_slope(
        1,
        config_.ecg_slope_slow_window_samples);
    if (config_.ecg_slope_mode == EcgSlopeMode::slow)
    {
        return {static_cast<float>(slow1), static_cast<float>(slow2)};
    }
    const auto fast1 = window_slope(0, config_.frame_size_raw);
    const auto fast2 = window_slope(1, config_.frame_size_raw);
    return {
        static_cast<float>(fast1),
        static_cast<float>(slow1),
        static_cast<float>(fast2),
        static_cast<float>(slow2),
    };
}

std::pair<double, double> FeatureFramer::rotate_ecg_pair(
    double ecg1,
    double ecg2) const
{
    if (config_.ecg_rotation_angle == 0.0)
    {
        return {ecg1, ecg2};
    }
    return {
        rotation_cos_ * ecg1 - rotation_sin_ * ecg2,
        rotation_sin_ * ecg1 + rotation_cos_ * ecg2,
    };
}

std::optional<FeatureFrame> FeatureFramer::process_sample(const RawSample &sample)
{
    const auto sample_idx = sample.sample_idx.value_or(next_sample_idx_);
    const auto time_ms = sample.time_ms.has_value()
        ? sample.time_ms
        : std::optional<double>(
              static_cast<double>(sample_idx) / config_.raw_hz * 1000.0);

    const auto sample_gap = last_sample_idx_.has_value() &&
        sample_idx != *last_sample_idx_ + 1;
    if (sample_gap &&
        (config_.ecg_slope_mode != EcgSlopeMode::none ||
         quality_tracker_.enabled()))
    {
        if (config_.ecg_slope_mode != EcgSlopeMode::none)
        {
            slope_history_.clear();
        }
        if (quality_tracker_.enabled())
        {
            quality_tracker_.reset();
        }
        raw_block_.clear();
    }

    next_sample_idx_ = sample_idx + 1;
    last_sample_idx_ = sample_idx;

    auto ecg1_zeroed = sample.ecg1 - config_.ecg1_offset;
    auto ecg2_zeroed = sample.ecg2 - config_.ecg2_offset;
    std::tie(ecg1_zeroed, ecg2_zeroed) =
        rotate_ecg_pair(ecg1_zeroed, ecg2_zeroed);

    std::optional<double> ecg3_zeroed;
    if (config_.include_ecg3)
    {
        if (!sample.ecg3.has_value())
        {
            throw std::invalid_argument("include_ecg3=True requires ECG3 values.");
        }
        ecg3_zeroed = config_.ecg_rotation_angle == 0.0
            ? *sample.ecg3 - config_.ecg3_offset
            : ecg2_zeroed - ecg1_zeroed;
    }

    if (config_.ecg_slope_mode != EcgSlopeMode::none)
    {
        const auto capacity = std::max(
            config_.frame_size_raw,
            config_.ecg_slope_slow_window_samples);
        if (slope_history_.size() == capacity)
        {
            slope_history_.pop_front();
        }
        slope_history_.push_back({ecg1_zeroed, ecg2_zeroed});
    }

    const auto normalized = ecg_normalizer_.step_with_extra(
        ecg1_zeroed,
        ecg2_zeroed,
        ecg3_zeroed);
    const auto quality = quality_tracker_.step_ecg_sample(
        *normalized[0],
        *normalized[1],
        ecg_normalizer_.last_scale());
    const auto puls_available = sample.puls.has_value();
    const auto puls_zeroed = puls_available
        ? *sample.puls - config_.puls_offset
        : 0.0;

    raw_block_.push_back({
        *normalized[0],
        *normalized[1],
        normalized[2],
        puls_zeroed,
        puls_available,
        quality,
        sample_idx,
        time_ms,
    });
    if (raw_block_.size() < config_.frame_size_raw)
    {
        return std::nullopt;
    }

    const auto first_pulse = raw_block_.front().puls_zeroed;
    const auto pulse_is_constant = std::all_of(
        raw_block_.begin(),
        raw_block_.end(),
        [&](const BufferedSample &item)
        {
            return std::abs(item.puls_zeroed - first_pulse) <=
                config_.puls_constant_tol;
        });
    if (config_.include_puls &&
        config_.align_to_puls_blocks &&
        !pulse_is_constant)
    {
        raw_block_.erase(raw_block_.begin());
        return std::nullopt;
    }
    if (config_.include_puls &&
        config_.verify_puls_constant_within_block &&
        !pulse_is_constant)
    {
        throw std::invalid_argument(
            "puls values inside an 8-sample block are not constant as expected.");
    }

    auto block = std::move(raw_block_);
    raw_block_.clear();
    const auto all_pulse_available = std::all_of(
        block.begin(),
        block.end(),
        [](const BufferedSample &item) { return item.puls_available; });

    std::optional<double> puls_value_norm;
    if (config_.include_puls ||
        (quality_tracker_.wants_pulse() && all_pulse_available))
    {
        puls_value_norm = puls_normalizer_.step(block.back().puls_zeroed);
    }

    std::vector<SignalQualitySample> quality_block;
    quality_block.reserve(block.size());
    for (const auto &item : block)
    {
        quality_block.push_back(item.quality);
    }
    auto quality_components = quality_tracker_.build_components(
        quality_block,
        puls_value_norm,
        all_pulse_available);

    std::vector<float> features;
    features.reserve(feature_columns_.size());
    for (const auto &item : block)
    {
        features.push_back(static_cast<float>(item.ecg1_norm));
    }
    for (const auto &item : block)
    {
        features.push_back(static_cast<float>(item.ecg2_norm));
    }
    if (config_.include_ecg3)
    {
        for (const auto &item : block)
        {
            features.push_back(static_cast<float>(*item.ecg3_norm));
        }
    }
    const auto slopes = build_slope_features();
    features.insert(features.end(), slopes.begin(), slopes.end());
    if (config_.include_puls)
    {
        features.push_back(static_cast<float>(*puls_value_norm));
    }
    if (quality_tracker_.enabled())
    {
        features.push_back(static_cast<float>(
            quality_tracker_.feature_from_components(quality_components)));
    }

    FeatureFrame result;
    result.frame_idx = next_frame_idx_++;
    result.start_sample_idx = block.front().sample_idx;
    result.end_sample_idx = block.back().sample_idx;
    result.start_time_ms = block.front().time_ms;
    result.end_time_ms = block.back().time_ms;
    result.features = std::move(features);
    result.auxiliary_features = std::move(quality_components);
    return result;
}

StreamingTriggerDetector::StreamingTriggerDetector(
    double threshold,
    std::int64_t holdoff_frames)
    : threshold_(threshold),
      emission_threshold_(threshold),
      holdoff_frames_(holdoff_frames),
      last_event_frame_(-1'000'000'000'000LL)
{
}

void StreamingTriggerDetector::set_threshold(double threshold)
{
    set_thresholds(threshold, threshold);
}

void StreamingTriggerDetector::set_thresholds(
    double threshold, double emission_threshold)
{
    if (!std::isfinite(threshold) || threshold < 0.0 || threshold > 1.0 ||
        !std::isfinite(emission_threshold) || emission_threshold < threshold ||
        emission_threshold > 1.0)
    {
        throw std::invalid_argument(
            "Thresholds must satisfy 0 <= detection <= emission <= 1");
    }
    if (threshold != threshold_ || emission_threshold != emission_threshold_)
    {
        threshold_ = threshold;
        emission_threshold_ = emission_threshold;
        candidate_.reset();
    }
}

void StreamingTriggerDetector::reset()
{
    last_event_frame_ = -1'000'000'000'000LL;
    candidate_.reset();
    awaiting_below_threshold_ = false;
}

std::optional<TriggerEvent> StreamingTriggerDetector::update(
    double probability,
    std::int64_t frame_idx,
    std::int64_t sample_idx,
    std::optional<double> time_ms,
    std::string sid)
{
    // After early emission, consume the tail of the same detection excursion.
    // Check this even during holdoff so a later excursion can rearm normally.
    if (awaiting_below_threshold_)
    {
        if (probability >= threshold_)
        {
            return std::nullopt;
        }
        awaiting_below_threshold_ = false;
    }
    if (frame_idx - last_event_frame_ < holdoff_frames_)
    {
        candidate_.reset();
        return std::nullopt;
    }
    if (candidate_ && emission_threshold_ > threshold_ &&
        candidate_->probability >= emission_threshold_ &&
        probability < emission_threshold_)
    {
        awaiting_below_threshold_ = probability >= threshold_;
        return emit_candidate(frame_idx, sample_idx, time_ms);
    }
    if (probability >= threshold_)
    {
        Candidate next{
            std::move(sid),
            frame_idx,
            sample_idx,
            time_ms,
            probability,
        };
        if (!candidate_.has_value() || probability > candidate_->probability)
        {
            candidate_ = std::move(next);
        }
        return std::nullopt;
    }
    if (!candidate_.has_value())
    {
        return std::nullopt;
    }
    return emit_candidate(frame_idx, sample_idx, time_ms);
}

std::optional<TriggerEvent> StreamingTriggerDetector::flush(
    std::int64_t frame_idx,
    std::int64_t sample_idx,
    std::optional<double> time_ms,
    std::string sid)
{
    static_cast<void>(sid);
    if (!candidate_.has_value())
    {
        return std::nullopt;
    }
    return emit_candidate(frame_idx, sample_idx, time_ms);
}

TriggerEvent StreamingTriggerDetector::emit_candidate(
    std::int64_t frame_idx,
    std::int64_t sample_idx,
    std::optional<double> time_ms)
{
    const auto peak = *candidate_;
    candidate_.reset();
    last_event_frame_ = frame_idx;
    return {
        peak.sid,
        peak.frame_idx,
        peak.sample_idx,
        peak.time_ms,
        frame_idx,
        sample_idx,
        time_ms,
        peak.probability,
    };
}

} // namespace dkfz::live
