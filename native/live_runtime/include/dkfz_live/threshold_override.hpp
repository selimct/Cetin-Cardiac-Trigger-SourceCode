/*
 * Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
 * SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
 * Developed at the German Cancer Research Center (DKFZ).
 */

#pragma once

#include "dkfz_live/deployment_manifest.hpp"

#include <filesystem>
#include <optional>
#include <string>

namespace dkfz::live
{

struct ThresholdOverrideUpdate
{
    std::optional<double> threshold;
    std::string error;
    std::optional<bool> early_emission = std::nullopt;
};

// Called by the supervisor, never from the inference thread. Repeated file
// contents/errors are suppressed; missing files leave the active policy alone.
class ThresholdOverrideReader
{
  public:
    ThresholdOverrideReader(std::filesystem::path path, DeploymentManifest manifest);
    ThresholdOverrideUpdate poll();

  private:
    std::filesystem::path path_;
    DeploymentManifest manifest_;
    std::string last_state_;
};

} // namespace dkfz::live
