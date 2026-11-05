/*
 * Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
 * SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
 * Developed at the German Cancer Research Center (DKFZ).
 */

#include "dkfz_live/deployment_manifest.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cctype>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string_view>

namespace dkfz::live
{
namespace
{

using Values = std::map<std::string, std::string>;

std::string trim(std::string value)
{
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos)
    {
        return {};
    }
    const auto last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

Values read_values(const std::filesystem::path &path)
{
    std::ifstream input(path);
    if (!input)
    {
        throw std::runtime_error(
            "Unable to open deployment manifest: " + path.string());
    }

    Values values;
    std::string line;
    std::size_t line_number = 0;
    while (std::getline(input, line))
    {
        ++line_number;
        const auto comment = line.find('#');
        if (comment != std::string::npos)
        {
            line.erase(comment);
        }
        line = trim(line);
        if (line.empty())
        {
            continue;
        }
        const auto equals = line.find('=');
        if (equals == std::string::npos)
        {
            throw std::runtime_error(
                path.string() + ":" + std::to_string(line_number) +
                ": expected key=value");
        }
        auto key = trim(line.substr(0, equals));
        auto value = trim(line.substr(equals + 1));
        if (key.empty())
        {
            throw std::runtime_error(
                path.string() + ":" + std::to_string(line_number) +
                ": key may not be empty");
        }
        if (!values.emplace(key, value).second)
        {
            throw std::runtime_error(
                path.string() + ":" + std::to_string(line_number) +
                ": duplicate key " + key);
        }
    }
    return values;
}

const std::string &required(
    const Values &values,
    const std::string &key)
{
    const auto found = values.find(key);
    if (found == values.end() || found->second.empty())
    {
        throw std::runtime_error(
            "Deployment manifest is missing required key: " + key);
    }
    return found->second;
}

std::string optional(
    const Values &values,
    const std::string &key,
    std::string fallback = {})
{
    const auto found = values.find(key);
    return found == values.end() ? std::move(fallback) : found->second;
}

double parse_double(
    const std::string &value,
    const std::string &key)
{
    std::size_t consumed = 0;
    const auto parsed = std::stod(value, &consumed);
    if (consumed != value.size() || !std::isfinite(parsed))
    {
        throw std::runtime_error("Invalid finite number for " + key + ": " + value);
    }
    return parsed;
}

std::int64_t parse_integer(
    const std::string &value,
    const std::string &key)
{
    std::size_t consumed = 0;
    const auto parsed = std::stoll(value, &consumed, 10);
    if (consumed != value.size())
    {
        throw std::runtime_error("Invalid integer for " + key + ": " + value);
    }
    return parsed;
}

std::size_t parse_size(
    const std::string &value,
    const std::string &key)
{
    const auto parsed = parse_integer(value, key);
    if (parsed < 0)
    {
        throw std::runtime_error(key + " may not be negative");
    }
    return static_cast<std::size_t>(parsed);
}

bool parse_bool(
    const std::string &value,
    const std::string &key)
{
    if (value == "true" || value == "1")
    {
        return true;
    }
    if (value == "false" || value == "0")
    {
        return false;
    }
    throw std::runtime_error("Invalid boolean for " + key + ": " + value);
}

std::vector<std::string> split_csv(const std::string &value)
{
    if (value.empty())
    {
        return {};
    }
    std::vector<std::string> result;
    std::istringstream stream(value);
    std::string item;
    while (std::getline(stream, item, ','))
    {
        item = trim(item);
        if (item.empty())
        {
            throw std::runtime_error("CSV manifest values may not contain empty items");
        }
        result.push_back(std::move(item));
    }
    return result;
}

void read_number_map(
    const Values &values,
    std::string_view prefix,
    std::map<std::string, double> &destination)
{
    for (const auto &[key, value] : values)
    {
        if (key.compare(0, prefix.size(), prefix) == 0)
        {
            const auto component = key.substr(prefix.size());
            if (component.empty())
            {
                throw std::runtime_error(
                    "Signal-quality component key may not be empty: " + key);
            }
            destination.emplace(component, parse_double(value, key));
        }
    }
}

} // namespace

DeploymentManifest load_deployment_manifest(
    const std::filesystem::path &path)
{
    const auto values = read_values(path);
    if (required(values, "schema") != kDeploymentManifestSchema)
    {
        throw std::runtime_error(
            "Unsupported deployment manifest schema: " +
            required(values, "schema"));
    }
    const auto contract_version = parse_integer(
        required(values, "preprocessing_contract_version"),
        "preprocessing_contract_version");
    if (contract_version != kPreprocessingContractVersion)
    {
        throw std::runtime_error(
            "Unsupported preprocessing_contract_version=" +
            std::to_string(contract_version) + "; this binary supports " +
            std::to_string(kPreprocessingContractVersion));
    }

    DeploymentManifest manifest;
    manifest.source_path = std::filesystem::absolute(path);
    manifest.source_settings = values;
    manifest.model_name = required(values, "model_name");
    manifest.model_variant = optional(values, "model_variant", "best");
    manifest.engine_path = required(values, "engine_path");
    if (manifest.engine_path.is_relative())
    {
        manifest.engine_path = manifest.source_path.parent_path() /
                               manifest.engine_path;
    }
    manifest.engine_path = manifest.engine_path.lexically_normal();
    const auto engine_size = optional(values, "engine_size_bytes");
    const auto engine_hash = optional(values, "engine_crc32");
    if (!engine_size.empty())
    {
        manifest.engine_size_bytes = static_cast<std::uint64_t>(
            parse_size(engine_size, "engine_size_bytes"));
    }
    if (!engine_hash.empty())
    {
        manifest.engine_crc32 = engine_hash;
    }
    manifest.input_tensor = optional(values, "input_tensor", "x");
    manifest.output_tensor = optional(values, "output_tensor", "logit");
    manifest.window_size = parse_size(required(values, "window_size"), "window_size");
    manifest.input_dim = parse_size(required(values, "input_dim"), "input_dim");
    manifest.threshold = parse_double(required(values, "threshold"), "threshold");
    const auto manual = optional(values, "manual_threshold");
    const auto fp = optional(values, "fp_threshold");
    const auto der = optional(values, "der_threshold");
    if (!fp.empty()) { manifest.fp_threshold = parse_double(fp, "fp_threshold"); }
    if (!der.empty()) { manifest.der_threshold = parse_double(der, "der_threshold"); }
    if (!manual.empty())
    {
        manifest.manual_threshold = parse_double(manual, "manual_threshold");
    }
    manifest.holdoff_frames = parse_integer(
        required(values, "holdoff_frames"), "holdoff_frames");
    manifest.model_output_is_probability = parse_bool(
        optional(values, "model_output_is_probability", "false"),
        "model_output_is_probability");

    auto &framer = manifest.framer;
    framer.raw_hz = parse_double(required(values, "raw_hz"), "raw_hz");
    framer.frame_size_raw = parse_size(
        required(values, "frame_size_raw"), "frame_size_raw");
    framer.include_puls = parse_bool(
        required(values, "include_puls"), "include_puls");
    framer.include_ecg3 = parse_bool(
        optional(values, "include_ecg3", "false"), "include_ecg3");
    framer.ecg1_offset = parse_double(
        required(values, "ecg1_offset"), "ecg1_offset");
    framer.ecg2_offset = parse_double(
        required(values, "ecg2_offset"), "ecg2_offset");
    framer.ecg3_offset = parse_double(
        optional(values, "ecg3_offset", "0"), "ecg3_offset");
    framer.ecg_slope_mode = parse_ecg_slope_mode(
        optional(values, "ecg_slope_mode", "none"));
    framer.ecg_slope_slow_window_samples = parse_size(
        optional(values, "ecg_slope_slow_window_samples", "40"),
        "ecg_slope_slow_window_samples");
    framer.signal_quality_mode = parse_signal_quality_mode(
        optional(values, "signal_quality_mode", "none"));
    framer.ecg_rotation_angle = parse_double(
        required(values, "ecg_rotation_angle"), "ecg_rotation_angle");
    framer.puls_offset = parse_double(
        required(values, "puls_offset"), "puls_offset");
    framer.ecg_quantile = parse_double(
        required(values, "ecg_quantile"), "ecg_quantile");
    framer.ecg_norm_window_samples = parse_size(
        required(values, "ecg_norm_window_samples"),
        "ecg_norm_window_samples");
    framer.ecg_norm_eps = parse_double(
        optional(values, "ecg_norm_eps", "1e-8"), "ecg_norm_eps");
    framer.ecg_norm_warmup_scale = parse_double(
        optional(values, "ecg_norm_warmup_scale", "1"),
        "ecg_norm_warmup_scale");
    framer.puls_quantile_low = parse_double(
        required(values, "puls_quantile_low"), "puls_quantile_low");
    framer.puls_quantile_high = parse_double(
        required(values, "puls_quantile_high"), "puls_quantile_high");
    framer.puls_norm_window_frames = parse_size(
        required(values, "puls_norm_window_frames"),
        "puls_norm_window_frames");
    framer.puls_norm_eps = parse_double(
        optional(values, "puls_norm_eps", "1e-8"), "puls_norm_eps");
    framer.puls_norm_warmup_value = parse_double(
        required(values, "puls_norm_warmup_value"),
        "puls_norm_warmup_value");
    framer.puls_clip = parse_bool(
        required(values, "puls_clip"), "puls_clip");
    framer.align_to_puls_blocks = parse_bool(
        required(values, "align_to_puls_blocks"), "align_to_puls_blocks");
    framer.verify_puls_constant_within_block = parse_bool(
        optional(values, "verify_puls_constant_within_block", "false"),
        "verify_puls_constant_within_block");
    framer.puls_constant_tol = parse_double(
        optional(values, "puls_constant_tol", "1e-12"),
        "puls_constant_tol");

    auto &quality = framer.signal_quality_stats;
    quality.present = parse_bool(
        optional(values, "signal_quality_stats_present", "false"),
        "signal_quality_stats_present");
    quality.transform = optional(values, "signal_quality_transform");
    quality.ecg_component_cols = split_csv(
        optional(values, "signal_quality_ecg_component_cols"));
    quality.puls_component_cols = split_csv(
        optional(values, "signal_quality_puls_component_cols"));
    read_number_map(values, "signal_quality_median.", quality.median);
    read_number_map(values, "signal_quality_iqr.", quality.iqr);
    read_number_map(values, "signal_quality_min_iqr.", quality.min_iqr);
    const auto z_clip = optional(values, "signal_quality_z_clip");
    if (!z_clip.empty())
    {
        quality.z_clip = parse_double(z_clip, "signal_quality_z_clip");
    }
    quality.ecg_weight = parse_double(
        optional(values, "signal_quality_ecg_weight", "1"),
        "signal_quality_ecg_weight");
    quality.puls_weight = parse_double(
        optional(values, "signal_quality_puls_weight", "0"),
        "signal_quality_puls_weight");
    quality.eps = parse_double(
        optional(values, "signal_quality_eps", "1e-6"),
        "signal_quality_eps");

    manifest.feature_columns = split_csv(required(values, "feature_columns"));
    manifest.ecg1_field = optional(values, "ecg1_field", "ecg1_raw");
    manifest.ecg2_field = optional(values, "ecg2_field", "ecg2_raw");
    manifest.ecg3_field = optional(values, "ecg3_field");
    manifest.puls_field = optional(values, "puls_field", "puls_raw");
    manifest.time_field = optional(values, "time_field", "time_ms");
    manifest.sample_index_field = optional(
        values, "sample_index_field", "raw_idx");
    manifest.sid_field = optional(values, "sid_field", "sid");

    validate_deployment_manifest(manifest);
    return manifest;
}

void validate_deployment_manifest(const DeploymentManifest &manifest)
{
    if (manifest.model_name.empty())
    {
        throw std::invalid_argument("model_name may not be empty");
    }
    if (manifest.engine_path.empty())
    {
        throw std::invalid_argument("engine_path may not be empty");
    }
    if (manifest.engine_size_bytes.has_value() !=
        manifest.engine_crc32.has_value())
    {
        throw std::invalid_argument(
            "engine_size_bytes and engine_crc32 must be provided together");
    }
    if (manifest.engine_crc32.has_value() &&
        (manifest.engine_crc32->size() != 8 ||
         !std::all_of(
             manifest.engine_crc32->begin(),
             manifest.engine_crc32->end(),
             [](char character)
             {
                 return std::isxdigit(
                            static_cast<unsigned char>(character)) != 0 &&
                        !(character >= 'A' && character <= 'F');
             })))
    {
        throw std::invalid_argument(
            "engine_crc32 must contain 8 lowercase hex digits");
    }
    if (manifest.window_size == 0 || manifest.input_dim == 0)
    {
        throw std::invalid_argument("window_size and input_dim must be positive");
    }
    if (manifest.window_size >
            static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) ||
        manifest.input_dim >
            static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) ||
        manifest.input_dim >
            std::numeric_limits<std::size_t>::max() / manifest.window_size)
    {
        throw std::invalid_argument("Configured inference shape is too large");
    }
    if (manifest.manual_threshold &&
        !(*manifest.manual_threshold >= 0.0 && *manifest.manual_threshold <= 1.0))
    {
        throw std::invalid_argument("manual_threshold must be empty or in [0, 1]");
    }
    for (const auto value : {manifest.fp_threshold, manifest.der_threshold})
    {
        if (value && !(*value >= 0.0 && *value <= 1.0))
        { throw std::invalid_argument("fp_threshold and der_threshold must be in [0, 1]"); }
    }
    if (!(manifest.threshold >= 0.0 && manifest.threshold <= 1.0))
    {
        throw std::invalid_argument("threshold must be in [0, 1]");
    }
    if (manifest.holdoff_frames < 0)
    {
        throw std::invalid_argument("holdoff_frames may not be negative");
    }
    if (manifest.framer.raw_hz <= 0.0 || manifest.framer.frame_size_raw == 0)
    {
        throw std::invalid_argument("raw_hz and frame_size_raw must be positive");
    }
    if (manifest.framer.include_ecg3 && manifest.ecg3_field.empty())
    {
        throw std::invalid_argument(
            "include_ecg3=true requires a non-empty ecg3_field");
    }
    if (manifest.framer.signal_quality_mode != SignalQualityMode::none &&
        !manifest.framer.signal_quality_stats.present)
    {
        throw std::invalid_argument(
            "signal quality is enabled but calibrated statistics are missing");
    }
    if (manifest.framer.signal_quality_mode != SignalQualityMode::none)
    {
        const auto &stats = manifest.framer.signal_quality_stats;
        if (stats.transform != "exp_neg_half_weighted_chi2_iqr" &&
            stats.transform != "log1p_weighted_chi2_iqr")
        {
            throw std::invalid_argument(
                "Unsupported signal-quality transform: " + stats.transform);
        }
        auto expected_components = ecg_quality_component_names();
        if (stats.ecg_component_cols != expected_components)
        {
            throw std::invalid_argument(
                "signal-quality ECG component order does not match the contract");
        }
        if (manifest.framer.signal_quality_mode == SignalQualityMode::ecg_puls)
        {
            if (stats.puls_component_cols != puls_quality_component_names())
            {
                throw std::invalid_argument(
                    "signal-quality pulse component order does not match the contract");
            }
            expected_components.insert(
                expected_components.end(),
                stats.puls_component_cols.begin(),
                stats.puls_component_cols.end());
        }
        else if (!stats.puls_component_cols.empty())
        {
            throw std::invalid_argument(
                "ECG-only signal quality may not contain pulse components");
        }
        for (const auto &component : expected_components)
        {
            if (stats.median.count(component) == 0 ||
                stats.iqr.count(component) == 0 ||
                stats.min_iqr.count(component) == 0)
            {
                throw std::invalid_argument(
                    "signal-quality statistics are incomplete for component: " +
                    component);
            }
        }
        if (stats.ecg_weight < 0.0 || stats.puls_weight < 0.0 ||
            stats.ecg_weight + stats.puls_weight <= 0.0 || stats.eps <= 0.0)
        {
            throw std::invalid_argument(
                "signal-quality weights and epsilon are invalid");
        }
    }

    std::vector<std::string> ecg_columns{"ecg1", "ecg2"};
    if (manifest.framer.include_ecg3)
    {
        ecg_columns.push_back("ecg3");
    }
    const auto generated = live_feature_columns(
        manifest.framer.frame_size_raw,
        manifest.framer.include_puls,
        ecg_columns,
        manifest.framer.ecg_slope_mode,
        manifest.framer.signal_quality_mode);
    if (generated != manifest.feature_columns)
    {
        throw std::invalid_argument(
            "feature_columns do not match the native preprocessing contract");
    }
    if (generated.size() != manifest.input_dim)
    {
        throw std::invalid_argument(
            "input_dim=" + std::to_string(manifest.input_dim) +
            " does not match feature_columns size " +
            std::to_string(generated.size()));
    }
}

void verify_deployment_engine_fingerprint(const DeploymentManifest &manifest)
{
    if (!manifest.engine_size_bytes.has_value())
    {
        return;
    }
    std::ifstream input(manifest.engine_path, std::ios::binary);
    if (!input)
    {
        throw std::runtime_error(
            "Unable to open fingerprinted TensorRT engine: " +
            manifest.engine_path.string());
    }
    std::uint32_t checksum = 0xFFFFFFFFU;
    std::uint64_t size = 0;
    std::array<char, 64 * 1024> buffer{};
    while (input.read(buffer.data(), static_cast<std::streamsize>(buffer.size())) ||
           input.gcount() > 0)
    {
        const auto count = static_cast<std::size_t>(input.gcount());
        size += count;
        for (std::size_t index = 0; index < count; ++index)
        {
            checksum ^= static_cast<unsigned char>(buffer[index]);
            for (unsigned bit = 0; bit < 8; ++bit)
            {
                const auto mask = static_cast<std::uint32_t>(
                    -(static_cast<std::int32_t>(checksum & 1U)));
                checksum = (checksum >> 1U) ^ (0xEDB88320U & mask);
            }
        }
    }
    if (!input.eof())
    {
        throw std::runtime_error(
            "Unable to read fingerprinted TensorRT engine: " +
            manifest.engine_path.string());
    }
    std::ostringstream hash_text;
    checksum ^= 0xFFFFFFFFU;
    hash_text << std::hex << std::setfill('0') << std::setw(8) << checksum;
    if (size != *manifest.engine_size_bytes ||
        hash_text.str() != *manifest.engine_crc32)
    {
        throw std::runtime_error(
            "TensorRT engine fingerprint does not match its deployment manifest");
    }
}

std::optional<double> reload_manual_threshold(const DeploymentManifest &manifest)
{
    auto values = read_values(manifest.source_path);
    const auto manual = optional(values, "manual_threshold");
    auto baseline = manifest.source_settings;
    values.erase("manual_threshold");
    baseline.erase("manual_threshold");
    if (values != baseline)
    {
        throw std::runtime_error(
            "Only manual_threshold can change live in the model manifest; "
            "other model/config changes require a restart");
    }
    if (manual.empty()) { return std::nullopt; }
    const auto threshold = parse_double(manual, "manual_threshold");
    if (!(threshold >= 0.0 && threshold <= 1.0))
    {
        throw std::runtime_error("manual_threshold must be empty or in [0, 1]");
    }
    return threshold;
}

} // namespace dkfz::live
