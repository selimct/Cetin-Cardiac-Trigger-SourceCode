/*
 * Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
 * SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
 * Developed at the German Cancer Research Center (DKFZ).
 */

#include "dkfz_live/threshold_override.hpp"

#include <cmath>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace dkfz::live
{
namespace
{
std::string trim(const std::string &value)
{
    const auto begin = value.find_first_not_of(" \t\r\n");
    return begin == std::string::npos ? "" : value.substr(
        begin, value.find_last_not_of(" \t\r\n") - begin + 1);
}

ThresholdOverrideUpdate parse_override(
    const std::string &text, const DeploymentManifest &manifest)
{
    std::map<std::string, std::string> values;
    std::istringstream input(text);
    std::string line;
    while (std::getline(input, line))
    {
        line = trim(line);
        if (line.empty() || line.front() == '#')
        {
            continue;
        }
        const auto equal = line.find('=');
        if (equal == std::string::npos ||
            !values.emplace(trim(line.substr(0, equal)),
                            trim(line.substr(equal + 1))).second)
        {
            throw std::runtime_error("Malformed or duplicate override field");
        }
    }
    if ((values.size() != 5 && values.size() != 6) ||
        (values.size() == 6 && !values.count("early_emission")) ||
        !values.count("model_name") ||
        !values.count("model_variant") || !values.count("engine_size_bytes") ||
        !values.count("engine_crc32") || !values.count("threshold"))
    {
        throw std::runtime_error("Override requires model identity, engine fingerprint, and threshold");
    }
    if (!manifest.engine_size_bytes || !manifest.engine_crc32 ||
        values.at("model_name") != manifest.model_name ||
        values.at("model_variant") != manifest.model_variant ||
        values.at("engine_size_bytes") != std::to_string(*manifest.engine_size_bytes) ||
        values.at("engine_crc32") != *manifest.engine_crc32)
    {
        throw std::runtime_error("Override does not match the deployed model/engine");
    }
    const auto &value = values.at("threshold");
    std::size_t used = 0;
    const auto threshold = std::stod(value, &used);
    if (used != value.size() || !std::isfinite(threshold) ||
        threshold < 0.0 || threshold > 1.0)
    {
        throw std::runtime_error("Override threshold must be finite and within [0, 1]");
    }
    std::optional<bool> early_emission;
    if (values.count("early_emission"))
    {
        const auto &mode = values.at("early_emission");
        if (mode != "true" && mode != "false")
        {
            throw std::runtime_error("early_emission must be true or false");
        }
        early_emission = mode == "true";
    }
    return {threshold, "", early_emission};
}
} // namespace

ThresholdOverrideReader::ThresholdOverrideReader(
    std::filesystem::path path, DeploymentManifest manifest)
    : path_(std::move(path)), manifest_(std::move(manifest)) {}

ThresholdOverrideUpdate ThresholdOverrideReader::poll()
{
    if (path_.empty())
    {
        return {};
    }
    try
    {
        if (!std::filesystem::exists(path_))
        {
            last_state_ = "missing";
            return {};
        }
        if (!std::filesystem::is_regular_file(path_))
        {
            throw std::runtime_error("Threshold override must be a regular file");
        }
        std::ifstream input(path_, std::ios::binary);
        if (!input)
        {
            throw std::runtime_error("Cannot open threshold override");
        }
        std::string contents(1025, '\0');
        input.read(contents.data(), static_cast<std::streamsize>(contents.size()));
        contents.resize(static_cast<std::size_t>(input.gcount()));
        if (input.bad())
        {
            throw std::runtime_error("Cannot read threshold override");
        }
        if (contents.size() > 1024)
        {
            throw std::runtime_error("Threshold override exceeds 1024 bytes");
        }
        const auto state = "contents:" + contents;
        if (state == last_state_)
        {
            return {};
        }
        last_state_ = state;
        try
        {
            return parse_override(contents, manifest_);
        }
        catch (const std::exception &error)
        {
            return {std::nullopt, error.what()};
        }
    }
    catch (const std::exception &error)
    {
        const auto state = "error:" + std::string(error.what());
        if (state == last_state_)
        {
            return {};
        }
        last_state_ = state;
        return {std::nullopt, error.what()};
    }
}
} // namespace dkfz::live
