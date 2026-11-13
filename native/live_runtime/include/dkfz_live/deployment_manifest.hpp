/*
 * Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
 * SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
 * Developed at the German Cancer Research Center (DKFZ).
 */

#pragma once

#include "dkfz_live/live_core.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace dkfz::live
{

inline constexpr const char *kDeploymentManifestSchema =
    "dkfz-live-deployment-v1";

struct DeploymentManifest
{
    std::filesystem::path source_path;
    std::string model_name;
    std::string model_variant = "best";
    std::filesystem::path engine_path;
    std::optional<std::uint64_t> engine_size_bytes;
    std::optional<std::string> engine_crc32;
    std::string input_tensor = "x";
    std::string output_tensor = "logit";
    std::size_t window_size = 0;
    std::size_t input_dim = 0;
    double threshold = 0.5;
    std::string prediction_mode = "current";
    std::int64_t prediction_delay_frames = 0;
    bool recovery_head = false;
    std::optional<double> recovery_threshold;
    std::optional<double> manual_threshold;
    std::optional<double> fp_threshold;
    std::optional<double> der_threshold;
    double emission_reference_threshold() const noexcept
    { return fp_threshold.value_or(threshold); }
    std::map<std::string, std::string> source_settings;
    std::int64_t holdoff_frames = 0;
    bool model_output_is_probability = false;
    FeatureFramerConfig framer;
    std::vector<std::string> feature_columns;
    std::string ecg1_field = "ecg1_raw";
    std::string ecg2_field = "ecg2_raw";
    std::string ecg3_field;
    std::string puls_field = "puls_raw";
    std::string time_field = "time_ms";
    std::string sample_index_field = "raw_idx";
    std::string sid_field = "sid";
    std::int64_t preprocessing_contract_version = kPreprocessingContractVersion;
};

DeploymentManifest load_deployment_manifest(
    const std::filesystem::path &path);

void validate_deployment_manifest(const DeploymentManifest &manifest);

// Reload only this editable field; reject any other model/config change.
std::optional<double> reload_manual_threshold(const DeploymentManifest &manifest);

void verify_deployment_engine_fingerprint(
    const DeploymentManifest &manifest);

} // namespace dkfz::live
