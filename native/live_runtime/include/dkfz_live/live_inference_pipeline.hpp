/*
 * Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
 * SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
 * Developed at the German Cancer Research Center (DKFZ).
 */

#pragma once

#include "dkfz_live/deployment_manifest.hpp"
#include "dkfz_live/inference_engine.hpp"
#include "dkfz_live/live_core.hpp"

#include <atomic>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace dkfz::live
{

struct InferenceResult
{
    FeatureFrame frame;
    float model_output = 0.0F;
    float probability = 0.0F;
    std::optional<TriggerEvent> trigger;
    std::optional<float> recovery_probability;
    bool recovery_emitted_now = false;
    std::optional<FeatureFrame> observation_frame;
    std::uint64_t window_assembly_ns = 0, inference_ns = 0, trigger_ns = 0;
    EngineTiming engine_timing;
};

struct TriggerPolicy
{
    double threshold;
    bool early_emission;
    std::optional<double> recovery_threshold = std::nullopt;
    std::uint64_t recovery_request_id = 0;
};

// One supervisor writer and one inference reader. Atomic fields avoid data
// races; a bounded read defers a concurrent update until a later frame.
// No locks, allocations, or retry loops run on the inference thread.
struct PolicyMailbox
{
    std::atomic<std::uint64_t> sequence{0}, request_id{0};
    std::atomic<double> threshold{0}, recovery{-1};
    std::atomic<bool> early{false};
    void store(const TriggerPolicy &policy) noexcept;
    bool load(TriggerPolicy &policy, std::uint64_t &version) const noexcept;
};

class LiveInferencePipeline
{
  public:
    LiveInferencePipeline(
        DeploymentManifest manifest,
        std::unique_ptr<LogitEngine> engine,
        bool early_emission = true);

    void reset();
    // Thread-safe request; the inference thread applies it at the next frame.
    void request_threshold(double threshold);
    void request_threshold(double threshold, bool early_emission);
    void request_policy(const TriggerPolicy &policy);
    TriggerPolicy active_policy() const noexcept;
    double active_threshold() const noexcept { return active_policy().threshold; }
    std::optional<InferenceResult> process_sample(
        const RawSample &sample,
        const std::string &sid = "0");
    std::optional<TriggerEvent> flush(const std::string &sid = "0");

    const DeploymentManifest &manifest() const noexcept { return manifest_; }
    const std::optional<FeatureFrame> &last_frame() const noexcept
    {
        return last_frame_;
    }

  private:
    static float sigmoid(float value);
    void enqueue_policy(const TriggerPolicy &policy);

    DeploymentManifest manifest_;
    std::unique_ptr<LogitEngine> engine_;
    FeatureFramer framer_;
    StreamingTriggerDetector trigger_;
    PolicyMailbox active_mailbox_, pending_mailbox_;
    // Only request writers take this mutex. Inference never touches it.
    std::mutex request_mutex_;
    TriggerPolicy applied_policy_, desired_policy_;
    std::uint64_t applied_version_ = 0;
    std::deque<std::vector<float>> feature_window_;
    std::vector<float> contiguous_;
    std::optional<FeatureFrame> last_frame_;
    std::deque<FeatureFrame> target_frames_;
    std::optional<std::int64_t> previous_raw_idx_;
    std::optional<double> previous_raw_time_;
    std::optional<std::string> previous_sid_;
};

} // namespace dkfz::live
