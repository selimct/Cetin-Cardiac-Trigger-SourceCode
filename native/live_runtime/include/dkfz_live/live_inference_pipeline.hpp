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
};

struct TriggerPolicy
{
    double threshold;
    bool early_emission;
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
    static double encode_policy(double threshold, bool early_emission);

    DeploymentManifest manifest_;
    std::unique_ptr<LogitEngine> engine_;
    FeatureFramer framer_;
    StreamingTriggerDetector trigger_;
    // Thresholds are nonnegative: use the sign bit for the mode so each policy
    // transfers atomically without locks or an extra idle-frame atomic read.
    std::atomic<double> active_policy_;
    std::atomic<double> pending_policy_;
    std::deque<std::vector<float>> feature_window_;
    std::optional<FeatureFrame> last_frame_;
};

} // namespace dkfz::live
