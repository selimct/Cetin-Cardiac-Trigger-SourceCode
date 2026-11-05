/*
 * Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
 * SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
 * Developed at the German Cancer Research Center (DKFZ).
 */

#include "dkfz_live/live_inference_pipeline.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace dkfz::live
{

static_assert(std::atomic<double>::is_always_lock_free,
              "Live threshold control requires lock-free double atomics");

LiveInferencePipeline::LiveInferencePipeline(
    DeploymentManifest manifest,
    std::unique_ptr<LogitEngine> engine,
    bool early_emission)
    : manifest_(std::move(manifest)),
      engine_(std::move(engine)),
      framer_(manifest_.framer),
      trigger_(manifest_.manual_threshold.value_or(manifest_.threshold), manifest_.holdoff_frames),
      active_policy_(encode_policy(manifest_.manual_threshold.value_or(manifest_.threshold), early_emission)),
      pending_policy_(std::numeric_limits<double>::quiet_NaN())
{
    validate_deployment_manifest(manifest_);
    const auto threshold = active_threshold();
    trigger_.set_thresholds(threshold, early_emission
        ? std::max(threshold, manifest_.emission_reference_threshold()) : threshold);
    if (!engine_)
    {
        throw std::invalid_argument("LiveInferencePipeline requires an engine");
    }
    if (framer_.feature_columns() != manifest_.feature_columns)
    {
        throw std::invalid_argument(
            "Native framer feature order does not match deployment manifest");
    }
}

void LiveInferencePipeline::request_threshold(double threshold)
{
    request_threshold(threshold, active_policy().early_emission);
}

double LiveInferencePipeline::encode_policy(double threshold, bool early_emission)
{
    return std::copysign(threshold, early_emission ? 1.0 : -1.0);
}

TriggerPolicy LiveInferencePipeline::active_policy() const noexcept
{
    const auto encoded = active_policy_.load();
    return {std::abs(encoded), !std::signbit(encoded)};
}

void LiveInferencePipeline::request_threshold(double threshold, bool early_emission)
{
    if (!std::isfinite(threshold) || threshold < 0.0 || threshold > 1.0)
    {
        throw std::invalid_argument("Threshold must be finite and within [0, 1]");
    }
    pending_policy_.store(encode_policy(threshold, early_emission));
}

void LiveInferencePipeline::reset()
{
    framer_.reset();
    trigger_.reset();
    feature_window_.clear();
    last_frame_.reset();
}

float LiveInferencePipeline::sigmoid(float value)
{
    if (value >= 0.0F)
    {
        const auto z = std::exp(-value);
        return 1.0F / (1.0F + z);
    }
    const auto z = std::exp(value);
    return z / (1.0F + z);
}

std::optional<InferenceResult> LiveInferencePipeline::process_sample(
    const RawSample &sample,
    const std::string &sid)
{
    auto frame = framer_.process_sample(sample);
    if (!frame.has_value())
    {
        return std::nullopt;
    }
    // Unchanged frames need only a read, without cache-line ownership traffic
    // from an atomic read-modify-write on every frame.
    if (std::isfinite(pending_policy_.load(std::memory_order_relaxed)))
    {
        const auto requested = pending_policy_.exchange(
            std::numeric_limits<double>::quiet_NaN());
        const auto threshold = std::abs(requested);
        const auto emission_threshold = std::signbit(requested)
            ? threshold : std::max(threshold, manifest_.emission_reference_threshold());
        trigger_.set_thresholds(threshold, emission_threshold);
        active_policy_.store(requested);
    }
    last_frame_ = *frame;
    feature_window_.push_back(frame->features);
    if (feature_window_.size() > manifest_.window_size)
    {
        feature_window_.pop_front();
    }
    if (feature_window_.size() < manifest_.window_size)
    {
        return std::nullopt;
    }

    std::vector<float> contiguous;
    contiguous.reserve(manifest_.window_size * manifest_.input_dim);
    for (const auto &features : feature_window_)
    {
        contiguous.insert(contiguous.end(), features.begin(), features.end());
    }
    if (contiguous.size() != manifest_.window_size * manifest_.input_dim)
    {
        throw std::runtime_error("Framer produced an invalid inference window size");
    }

    const auto model_output = engine_->infer_logit(contiguous);
    const auto probability = manifest_.model_output_is_probability
        ? model_output
        : sigmoid(model_output);
    if (!std::isfinite(probability))
    {
        throw std::runtime_error("Model produced a non-finite probability");
    }
    auto event = trigger_.update(
        probability,
        frame->frame_idx,
        frame->end_sample_idx,
        frame->end_time_ms,
        sid);
    return InferenceResult{
        std::move(*frame), model_output, probability, std::move(event)};
}

std::optional<TriggerEvent> LiveInferencePipeline::flush(const std::string &sid)
{
    if (!last_frame_.has_value())
    {
        return std::nullopt;
    }
    return trigger_.flush(
        last_frame_->frame_idx,
        last_frame_->end_sample_idx,
        last_frame_->end_time_ms,
        sid);
}

} // namespace dkfz::live
