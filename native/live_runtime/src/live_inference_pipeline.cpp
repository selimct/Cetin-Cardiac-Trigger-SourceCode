/*
 * Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
 * SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
 * Developed at the German Cancer Research Center (DKFZ).
 */

#include "dkfz_live/live_inference_pipeline.hpp"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <limits>
#include <stdexcept>

namespace dkfz::live
{

static_assert(std::atomic<double>::is_always_lock_free && std::atomic<std::uint64_t>::is_always_lock_free && std::atomic<bool>::is_always_lock_free,
              "Live threshold control requires lock-free double atomics");

LiveInferencePipeline::LiveInferencePipeline(
    DeploymentManifest manifest,
    std::unique_ptr<LogitEngine> engine,
    bool early_emission)
    : manifest_(std::move(manifest)),
      engine_(std::move(engine)),
      framer_(manifest_.framer),
      trigger_(manifest_.manual_threshold.value_or(manifest_.threshold), manifest_.holdoff_frames),
      applied_policy_{manifest_.manual_threshold.value_or(manifest_.threshold), early_emission,
                      manifest_.recovery_threshold, 0},
      desired_policy_(applied_policy_)
{
    contiguous_.reserve(manifest_.window_size * manifest_.input_dim);
    validate_deployment_manifest(manifest_);
    trigger_.set_recovery_threshold(manifest_.recovery_threshold);
    active_mailbox_.store(applied_policy_);
    const auto threshold = applied_policy_.threshold;
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
    std::lock_guard<std::mutex> guard(request_mutex_);
    auto policy = desired_policy_;
    policy.threshold = threshold;
    enqueue_policy(policy);
}

void PolicyMailbox::store(const TriggerPolicy &policy) noexcept
{
    sequence.fetch_add(1);
    threshold.store(policy.threshold);
    early.store(policy.early_emission);
    recovery.store(policy.recovery_threshold.value_or(-1));
    request_id.store(policy.recovery_request_id);
    sequence.fetch_add(1);
}

bool PolicyMailbox::load(TriggerPolicy &policy, std::uint64_t &version) const noexcept
{
    const auto before = sequence.load();
    if (before & 1U) return false;
    policy.threshold = threshold.load();
    policy.early_emission = early.load();
    const auto value = recovery.load();
    policy.recovery_threshold = value < 0 ? std::nullopt : std::optional<double>(value);
    policy.recovery_request_id = request_id.load();
    version = sequence.load();
    return before == version;
}

TriggerPolicy LiveInferencePipeline::active_policy() const noexcept
{
    TriggerPolicy policy{};
    std::uint64_t version{};
    // Readback runs on the supervisor, never the inference thread.
    while (!active_mailbox_.load(policy, version)) {}
    return policy;
}

void LiveInferencePipeline::request_policy(const TriggerPolicy &policy)
{
    std::lock_guard<std::mutex> guard(request_mutex_);
    enqueue_policy(policy);
}

void LiveInferencePipeline::enqueue_policy(const TriggerPolicy &policy)
{
    if (!std::isfinite(policy.threshold) || policy.threshold < 0 || policy.threshold > 1)
        throw std::invalid_argument("Threshold must be finite and within [0, 1]");
    if (policy.recovery_threshold && (manifest_.prediction_delay_frames || !manifest_.recovery_head ||
        !std::isfinite(*policy.recovery_threshold) || *policy.recovery_threshold <= 0 ||
        *policy.recovery_threshold > 1))
        throw std::invalid_argument("Recovery requires two outputs and a threshold in (0, 1]");
    desired_policy_ = policy;
    pending_mailbox_.store(policy);
}

void LiveInferencePipeline::request_threshold(double threshold, bool early_emission)
{
    std::lock_guard<std::mutex> guard(request_mutex_);
    auto policy = desired_policy_;
    policy.threshold = threshold;
    policy.early_emission = early_emission;
    enqueue_policy(policy);
}

void LiveInferencePipeline::reset()
{
    framer_.reset();
    trigger_.reset();
    feature_window_.clear();
    last_frame_.reset();
    target_frames_.clear();
    previous_raw_idx_.reset();
    previous_raw_time_.reset();
    previous_sid_.reset();
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
    if (manifest_.prediction_delay_frames && previous_sid_ && *previous_sid_ != sid) reset();
    const auto raw_idx = sample.sample_idx.value_or(previous_raw_idx_.value_or(-1) + 1);
    if (manifest_.prediction_delay_frames && previous_sid_ &&
        (*previous_sid_ != sid || raw_idx != *previous_raw_idx_ + 1)) reset();
    previous_raw_idx_ = raw_idx;
    previous_raw_time_ = sample.time_ms.value_or(static_cast<double>(raw_idx) / manifest_.framer.raw_hz * 1000);
    previous_sid_ = sid;
    auto effective_sample = sample;
    if (manifest_.prediction_delay_frames) effective_sample.sample_idx = raw_idx;
    auto frame = framer_.process_sample(effective_sample);
    if (!frame.has_value())
    {
        return std::nullopt;
    }
    TriggerPolicy requested{};
    std::uint64_t version{};
    if (pending_mailbox_.load(requested, version) && version != applied_version_)
    {
        if (requested.threshold != applied_policy_.threshold ||
            requested.early_emission != applied_policy_.early_emission)
        {
            trigger_.set_thresholds(requested.threshold, requested.early_emission
                ? std::max(requested.threshold, manifest_.emission_reference_threshold())
                : requested.threshold);
        }
        if (requested.recovery_threshold != applied_policy_.recovery_threshold)
            trigger_.set_recovery_threshold(requested.recovery_threshold);
        applied_policy_ = requested;
        applied_version_ = version;
        active_mailbox_.store(applied_policy_);
    }
    last_frame_ = *frame;
    target_frames_.push_back(*frame);
    if (target_frames_.size() > static_cast<std::size_t>(manifest_.prediction_delay_frames + 1))
        target_frames_.pop_front();
    feature_window_.push_back(frame->features);
    if (feature_window_.size() > manifest_.window_size)
    {
        feature_window_.pop_front();
    }
    if (feature_window_.size() < manifest_.window_size)
    {
        return std::nullopt;
    }

    const auto window_started = std::chrono::steady_clock::now();
    contiguous_.clear();
    for (const auto &features : feature_window_)
    {
        contiguous_.insert(contiguous_.end(), features.begin(), features.end());
    }
    if (contiguous_.size() != manifest_.window_size * manifest_.input_dim)
    {
        throw std::runtime_error("Framer produced an invalid inference window size");
    }

    const auto inference_started = std::chrono::steady_clock::now();
    const auto outputs = engine_->infer_logits(contiguous_);
    const auto trigger_started = std::chrono::steady_clock::now();
    if (outputs.size() != (manifest_.recovery_head ? 2U : 1U))
        throw std::runtime_error("Engine output count does not match recovery_head in manifest");
    const auto model_output = outputs[0];
    std::optional<float> recovery_probability;
    if (manifest_.recovery_head)
    {
        recovery_probability = manifest_.model_output_is_probability ? outputs[1] : sigmoid(outputs[1]);
        if (!std::isfinite(*recovery_probability) || *recovery_probability < 0 || *recovery_probability > 1)
            throw std::runtime_error("Model produced an invalid recovery probability");
    }
    const auto probability = manifest_.model_output_is_probability
        ? model_output
        : sigmoid(model_output);
    if (!std::isfinite(probability))
    {
        throw std::runtime_error("Model produced a non-finite probability");
    }
    const auto &target = target_frames_.front();
    auto event = trigger_.update(
        probability,
        frame->frame_idx,
        frame->end_sample_idx,
        frame->end_time_ms,
        sid,
        recovery_probability,
        manifest_.prediction_delay_frames ? std::optional<TriggerPosition>(
            TriggerPosition{target.frame_idx, target.end_sample_idx, target.end_time_ms}) : std::nullopt);
    if (event) event->decision_monotonic_ns = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
    const auto completed = std::chrono::steady_clock::now();
    auto ns = [](auto a, auto b) { return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count()); };
    return InferenceResult{
        target, model_output, probability, std::move(event), recovery_probability,
        trigger_.last_event_was_recovery(), manifest_.prediction_delay_frames ? frame : std::nullopt,
        ns(window_started, inference_started), ns(inference_started, trigger_started),
        ns(trigger_started, completed), engine_->last_timing()};
}

std::optional<TriggerEvent> LiveInferencePipeline::flush(const std::string &sid)
{
    if (!last_frame_.has_value())
    {
        return std::nullopt;
    }
    auto event = trigger_.flush(
        last_frame_->frame_idx,
        manifest_.prediction_delay_frames ? *previous_raw_idx_ : last_frame_->end_sample_idx,
        manifest_.prediction_delay_frames ? previous_raw_time_ : last_frame_->end_time_ms,
        sid);
    if (event) event->decision_monotonic_ns = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
    return event;
}

} // namespace dkfz::live
