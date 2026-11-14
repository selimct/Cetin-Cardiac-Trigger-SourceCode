/*
 * Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
 * SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
 * Developed at the German Cancer Research Center (DKFZ).
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace dkfz::live
{

inline constexpr std::int64_t kPreprocessingContractVersion = 4;
// Eight 400 Hz samples span 20 ms, independently of the model frame cadence.
inline constexpr std::size_t kFastSlopeWindowSamples = 8;
inline constexpr const char *kSignalQualityFeature = "signal_quality";

enum class EcgSlopeMode
{
    none,
    fast,
    fast_slow,
    slow,
};

enum class SignalQualityMode
{
    none,
    ecg,
    ecg_puls,
};

EcgSlopeMode parse_ecg_slope_mode(const std::string &value);
SignalQualityMode parse_signal_quality_mode(const std::string &value);
std::string to_string(EcgSlopeMode value);
std::string to_string(SignalQualityMode value);

const std::vector<std::string> &ecg_quality_component_names();
const std::vector<std::string> &puls_quality_component_names();

std::vector<std::string> signal_quality_component_names(
    SignalQualityMode mode,
    bool include_puls = true);

std::vector<std::string> live_feature_columns(
    std::size_t frame_size_raw,
    bool include_puls,
    const std::vector<std::string> &ecg_columns,
    EcgSlopeMode slope_mode,
    SignalQualityMode quality_mode);

struct SignalQualityStats
{
    bool present = false;
    std::string transform;
    std::vector<std::string> ecg_component_cols;
    std::vector<std::string> puls_component_cols;
    std::map<std::string, double> median;
    std::map<std::string, double> iqr;
    std::map<std::string, double> min_iqr;
    std::optional<double> z_clip;
    double ecg_weight = 1.0;
    double puls_weight = 0.0;
    double eps = 1e-6;
};

double signal_quality_feature_from_components(
    const std::map<std::string, double> &components,
    const SignalQualityStats &stats,
    double eps = 1e-6);

class RollingPairNormQuantileNormalizer
{
  public:
    RollingPairNormQuantileNormalizer(
        double q = 0.999,
        std::size_t window_samples = 4000,
        double eps = 1e-8,
        double warmup_scale = 1.0);

    void reset();
    std::pair<double, double> step(double ecg1, double ecg2);
    std::array<std::optional<double>, 3> step_with_extra(
        double ecg1,
        double ecg2,
        std::optional<double> ecg3);

    double q() const noexcept { return q_; }
    std::size_t window_samples() const noexcept { return window_samples_; }
    double eps() const noexcept { return eps_; }
    double warmup_scale() const noexcept { return warmup_scale_; }
    double last_scale() const noexcept { return last_scale_; }

  private:
    double current_quantile() const;

    double q_;
    std::size_t window_samples_;
    double eps_;
    double warmup_scale_;
    double last_scale_;
    std::deque<double> norm_history_;
};

class RollingScalarQuantileNormalizer
{
  public:
    RollingScalarQuantileNormalizer(
        double q_low = 0.05,
        double q_high = 0.95,
        std::size_t window_frames = 500,
        double eps = 1e-8,
        double warmup_value = 0.5,
        bool clip = true);

    void reset();
    double step(double value);

    double q_low() const noexcept { return q_low_; }
    double q_high() const noexcept { return q_high_; }
    std::size_t window_frames() const noexcept { return window_frames_; }
    double eps() const noexcept { return eps_; }
    double warmup_value() const noexcept { return warmup_value_; }
    bool clip() const noexcept { return clip_; }

  private:
    double q_low_;
    double q_high_;
    std::size_t window_frames_;
    double eps_;
    double warmup_value_;
    bool clip_;
    std::deque<double> history_;
};

using SignalQualitySample = std::array<double, 5>;

class SignalQualityTracker
{
  public:
    SignalQualityTracker(
        SignalQualityMode mode = SignalQualityMode::none,
        SignalQualityStats stats = {},
        double eps = 1e-6,
        double ecg_flat_threshold = 0.002,
        double puls_flat_threshold = 0.001,
        std::size_t puls_flat_window_frames = 10,
        std::size_t puls_median_window_frames = 200);

    void reset();
    bool enabled() const noexcept { return mode_ != SignalQualityMode::none; }
    bool wants_pulse() const noexcept { return mode_ == SignalQualityMode::ecg_puls; }

    SignalQualitySample step_ecg_sample(
        double ecg1_norm,
        double ecg2_norm,
        double ecg_scale);

    std::map<std::string, double> build_components(
        const std::vector<SignalQualitySample> &ecg_quality_block,
        std::optional<double> puls_value_norm,
        bool puls_available);

    double feature_from_components(
        const std::map<std::string, double> &components) const;

  private:
    SignalQualityMode mode_;
    SignalQualityStats stats_;
    double eps_;
    double ecg_flat_threshold_;
    double puls_flat_threshold_;
    std::size_t puls_flat_window_frames_;
    std::size_t puls_median_window_frames_;
    std::optional<std::pair<double, double>> prev_ecg_;
    std::optional<std::pair<double, double>> prev_ecg_diff_;
    std::optional<double> prev_puls_;
    std::optional<double> prev_prev_puls_;
    std::deque<double> puls_flat_history_;
    std::deque<double> puls_history_;
};

struct FeatureFramerConfig
{
    double raw_hz = 400.0;
    std::size_t frame_size_raw = 8;
    bool include_puls = true;
    double ecg1_offset = 0.0;
    double ecg2_offset = 0.0;
    double ecg3_offset = 0.0;
    bool include_ecg3 = false;
    EcgSlopeMode ecg_slope_mode = EcgSlopeMode::none;
    std::size_t ecg_slope_slow_window_samples = 40;
    SignalQualityMode signal_quality_mode = SignalQualityMode::none;
    SignalQualityStats signal_quality_stats;
    std::size_t signal_quality_puls_flat_window_frames = 10;
    std::size_t signal_quality_puls_median_window_frames = 200;
    bool reset_on_discontinuity = false;
    double ecg_rotation_angle = 0.0;
    double puls_offset = 0.0;
    double ecg_quantile = 0.999;
    std::size_t ecg_norm_window_samples = 4000;
    double ecg_norm_eps = 1e-8;
    double ecg_norm_warmup_scale = 1.0;
    double puls_quantile_low = 0.05;
    double puls_quantile_high = 0.95;
    std::size_t puls_norm_window_frames = 500;
    double puls_norm_eps = 1e-8;
    double puls_norm_warmup_value = 0.5;
    bool puls_clip = true;
    bool align_to_puls_blocks = true;
    bool verify_puls_constant_within_block = false;
    double puls_constant_tol = 1e-12;
};

struct RawSample
{
    double ecg1 = 0.0;
    double ecg2 = 0.0;
    std::optional<double> ecg3;
    std::optional<double> puls;
    std::optional<std::int64_t> sample_idx;
    std::optional<double> time_ms;
};

struct FeatureFrame
{
    std::int64_t frame_idx = 0;
    std::int64_t start_sample_idx = 0;
    std::int64_t end_sample_idx = 0;
    std::optional<double> start_time_ms;
    std::optional<double> end_time_ms;
    std::vector<float> features;
    std::map<std::string, double> auxiliary_features;
};

class FeatureFramer
{
  public:
    explicit FeatureFramer(FeatureFramerConfig config = {});

    void reset();
    std::optional<FeatureFrame> process_sample(const RawSample &sample);

    const FeatureFramerConfig &config() const noexcept { return config_; }
    const std::vector<std::string> &feature_columns() const noexcept
    {
        return feature_columns_;
    }

  private:
    struct BufferedSample
    {
        double ecg1_norm;
        double ecg2_norm;
        std::optional<double> ecg3_norm;
        double puls_zeroed;
        bool puls_available;
        SignalQualitySample quality;
        std::int64_t sample_idx;
        std::optional<double> time_ms;
    };

    static std::vector<double> linear_slope_weights(std::size_t window_samples);
    double window_slope(std::size_t channel, std::size_t window_samples) const;
    std::vector<float> build_slope_features() const;
    std::pair<double, double> rotate_ecg_pair(double ecg1, double ecg2) const;

    FeatureFramerConfig config_;
    std::vector<std::string> feature_columns_;
    double rotation_cos_;
    double rotation_sin_;
    RollingPairNormQuantileNormalizer ecg_normalizer_;
    RollingScalarQuantileNormalizer puls_normalizer_;
    SignalQualityTracker quality_tracker_;
    std::map<std::size_t, std::vector<double>> slope_weights_;
    std::deque<std::pair<double, double>> slope_history_;
    std::vector<BufferedSample> raw_block_;
    std::int64_t next_sample_idx_ = 0;
    std::int64_t next_frame_idx_ = 0;
    std::optional<std::int64_t> last_sample_idx_;
};

struct TriggerPosition
{
    std::int64_t frame_idx;
    std::int64_t sample_idx;
    std::optional<double> time_ms;
};

struct TriggerEvent
{
    std::string sid;
    std::int64_t peak_frame_idx = 0;
    std::int64_t peak_sample_idx = 0;
    std::optional<double> peak_time_ms;
    std::int64_t emitted_on_frame_idx = 0;
    std::int64_t emitted_on_sample_idx = 0;
    std::optional<double> emitted_on_time_ms;
    double probability = 0.0;
    std::optional<std::uint64_t> decision_monotonic_ns;
    std::optional<std::uint64_t> gpio_asserted_monotonic_ns;
};

class StreamingTriggerDetector
{
  public:
    StreamingTriggerDetector(double threshold, std::int64_t holdoff_frames);

    void reset();
    // Drop a pending peak when changing policy; retain the last emitted holdoff.
    void set_threshold(double threshold);
    void set_thresholds(double detection_threshold, double emission_threshold);
    void set_recovery_threshold(std::optional<double> threshold);
    bool last_event_was_recovery() const noexcept { return last_event_was_recovery_; }
    std::optional<TriggerEvent> update(
        double probability,
        std::int64_t frame_idx,
        std::int64_t sample_idx,
        std::optional<double> time_ms,
        std::string sid = "0",
        std::optional<double> recovery_probability = std::nullopt,
        std::optional<TriggerPosition> target = std::nullopt);
    std::optional<TriggerEvent> flush(
        std::int64_t frame_idx,
        std::int64_t sample_idx,
        std::optional<double> time_ms,
        std::string sid = "0");

  private:
    struct Candidate
    {
        std::string sid;
        std::int64_t frame_idx;
        std::int64_t sample_idx;
        std::optional<double> time_ms;
        double probability;
    };

    TriggerEvent emit_candidate(
        std::int64_t frame_idx,
        std::int64_t sample_idx,
        std::optional<double> time_ms);

    std::optional<TriggerEvent> update_primary(
        double probability, std::int64_t frame_idx, std::int64_t sample_idx,
        std::optional<double> time_ms, std::string sid,
        std::optional<TriggerPosition> target);

    double threshold_;
    double emission_threshold_;
    bool awaiting_below_threshold_ = false;
    std::int64_t holdoff_frames_;
    std::int64_t last_event_frame_;
    std::optional<Candidate> candidate_;
    std::optional<Candidate> previous_input_;
    std::optional<double> recovery_threshold_;
    bool last_event_was_recovery_ = false;
};

} // namespace dkfz::live
