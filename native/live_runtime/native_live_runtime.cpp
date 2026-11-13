/*
 * Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
 * SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
 * Developed at the German Cancer Research Center (DKFZ).
 */

#include "dkfz_live/deployment_manifest.hpp"
#include "dkfz_live/inference_engine.hpp"
#include "dkfz_live/live_inference_pipeline.hpp"

#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

namespace fs = std::filesystem;
using dkfz::live::DeploymentManifest;
using dkfz::live::InferenceResult;
using dkfz::live::RawSample;
using dkfz::live::TriggerEvent;

namespace
{

struct Options
{
    fs::path manifest_path;
    fs::path input_path;
    std::string sid_override;
    bool emit_probabilities = false;
    bool validate_only = false;
    bool check_engine = false;
    std::uint64_t max_samples = 0;
};

std::string json_escape(const std::string &value)
{
    std::string output;
    output.reserve(value.size() + 8);
    for (const auto character : value)
    {
        switch (character)
        {
        case '\\':
            output += "\\\\";
            break;
        case '"':
            output += "\\\"";
            break;
        case '\n':
            output += "\\n";
            break;
        case '\r':
            output += "\\r";
            break;
        case '\t':
            output += "\\t";
            break;
        default:
            output += character;
        }
    }
    return output;
}

std::optional<std::size_t> find_json_value(
    const std::string &line,
    const std::string &key)
{
    const auto token = '"' + key + '"';
    auto position = line.find(token);
    while (position != std::string::npos)
    {
        position += token.size();
        while (position < line.size() &&
               std::isspace(static_cast<unsigned char>(line[position])) != 0)
        {
            ++position;
        }
        if (position < line.size() && line[position] == ':')
        {
            ++position;
            while (position < line.size() &&
                   std::isspace(static_cast<unsigned char>(line[position])) != 0)
            {
                ++position;
            }
            return position;
        }
        position = line.find(token, position);
    }
    return std::nullopt;
}

std::optional<double> json_number(
    const std::string &line,
    const std::string &key,
    bool required)
{
    const auto position = find_json_value(line, key);
    if (!position.has_value())
    {
        if (required)
        {
            throw std::runtime_error("Missing numeric JSON field: " + key);
        }
        return std::nullopt;
    }
    if (line.compare(*position, 4, "null") == 0)
    {
        if (required)
        {
            throw std::runtime_error("Required JSON field is null: " + key);
        }
        return std::nullopt;
    }
    const char *start = line.c_str() + *position;
    char *end = nullptr;
    const auto parsed = std::strtod(start, &end);
    if (end == start || !std::isfinite(parsed))
    {
        throw std::runtime_error("Invalid finite JSON number for field: " + key);
    }
    return parsed;
}

std::optional<std::string> json_scalar_string(
    const std::string &line,
    const std::string &key)
{
    const auto position = find_json_value(line, key);
    if (!position.has_value() || line.compare(*position, 4, "null") == 0)
    {
        return std::nullopt;
    }
    if (line[*position] == '"')
    {
        std::string result;
        bool escaped = false;
        for (auto index = *position + 1; index < line.size(); ++index)
        {
            const auto character = line[index];
            if (escaped)
            {
                result += character;
                escaped = false;
            }
            else if (character == '\\')
            {
                escaped = true;
            }
            else if (character == '"')
            {
                return result;
            }
            else
            {
                result += character;
            }
        }
        throw std::runtime_error("Unterminated JSON string for field: " + key);
    }

    auto end = *position;
    while (end < line.size() && line[end] != ',' && line[end] != '}')
    {
        ++end;
    }
    auto result = line.substr(*position, end - *position);
    const auto first = result.find_first_not_of(" \t\r\n");
    const auto last = result.find_last_not_of(" \t\r\n");
    if (first == std::string::npos)
    {
        return std::nullopt;
    }
    return result.substr(first, last - first + 1);
}

std::int64_t checked_integer(double value, const std::string &key)
{
    if (std::floor(value) != value ||
        value < static_cast<double>(std::numeric_limits<std::int64_t>::min()) ||
        value > static_cast<double>(std::numeric_limits<std::int64_t>::max()))
    {
        throw std::runtime_error("JSON field is not an int64: " + key);
    }
    return static_cast<std::int64_t>(value);
}

void print_help(const char *program)
{
    std::cout
        << "Usage: " << program << " --manifest PATH [options]\n"
        << "Run the DKFZ C++ preprocessing, TensorRT, and trigger pipeline.\n\n"
        << "  --manifest PATH          native deployment manifest\n"
        << "  --input PATH             UART-sample JSONL; stdin by default\n"
        << "  --sid VALUE              override stream/session identifier\n"
        << "  --emit-probabilities     emit one JSON record per inferred frame\n"
        << "  --max-samples N          stop after N raw samples; 0 is unlimited\n"
        << "  --validate-only          validate manifest without loading TensorRT\n"
        << "  --check-engine           with --validate-only, also load the engine\n";
}

Options parse_args(int argc, char **argv)
{
    Options options;
    auto require_value = [&](int &index, const std::string &name)
    {
        if (index + 1 >= argc)
        {
            throw std::runtime_error("Missing value for " + name);
        }
        return std::string(argv[++index]);
    };

    for (int index = 1; index < argc; ++index)
    {
        const std::string option = argv[index];
        if (option == "--help" || option == "-h")
        {
            print_help(argv[0]);
            std::exit(0);
        }
        if (option == "--manifest")
        {
            options.manifest_path = require_value(index, option);
        }
        else if (option == "--input")
        {
            options.input_path = require_value(index, option);
        }
        else if (option == "--sid")
        {
            options.sid_override = require_value(index, option);
        }
        else if (option == "--emit-probabilities")
        {
            options.emit_probabilities = true;
        }
        else if (option == "--validate-only")
        {
            options.validate_only = true;
        }
        else if (option == "--check-engine")
        {
            options.check_engine = true;
        }
        else if (option == "--max-samples")
        {
            const auto value = require_value(index, option);
            std::size_t consumed = 0;
            options.max_samples = std::stoull(value, &consumed, 10);
            if (consumed != value.size())
            {
                throw std::runtime_error("Invalid value for --max-samples: " + value);
            }
        }
        else
        {
            throw std::runtime_error("Unknown option: " + option);
        }
    }
    if (options.manifest_path.empty())
    {
        throw std::runtime_error("--manifest PATH is required");
    }
    if (options.check_engine && !options.validate_only)
    {
        throw std::runtime_error("--check-engine requires --validate-only");
    }
    return options;
}

std::unique_ptr<dkfz::live::LogitEngine> make_engine(
    const DeploymentManifest &manifest)
{
    dkfz::live::verify_deployment_engine_fingerprint(manifest);
    return dkfz::live::make_tensorrt_logit_engine(
        manifest.engine_path,
        manifest.input_tensor,
        manifest.output_tensor,
        manifest.window_size,
        manifest.input_dim);
}

RawSample parse_sample(
    const std::string &line,
    const DeploymentManifest &manifest,
    std::int64_t fallback_sample_index)
{
    RawSample sample;
    sample.ecg1 = *json_number(line, manifest.ecg1_field, true);
    sample.ecg2 = *json_number(line, manifest.ecg2_field, true);
    if (manifest.framer.include_ecg3)
    {
        sample.ecg3 = json_number(line, manifest.ecg3_field, true);
    }
    if (manifest.framer.include_puls)
    {
        sample.puls = json_number(line, manifest.puls_field, true);
    }
    else if (manifest.framer.signal_quality_mode ==
             dkfz::live::SignalQualityMode::ecg_puls)
    {
        sample.puls = json_number(line, manifest.puls_field, false);
    }
    const auto index = json_number(line, manifest.sample_index_field, false);
    if (index.has_value())
    {
        sample.sample_idx = checked_integer(*index, manifest.sample_index_field);
    }
    else
    {
        sample.sample_idx = fallback_sample_index;
    }
    sample.time_ms = json_number(line, manifest.time_field, false);
    if (!sample.time_ms.has_value())
    {
        sample.time_ms = 1000.0 * static_cast<double>(*sample.sample_idx) /
                         manifest.framer.raw_hz;
    }
    return sample;
}

void emit_event(const TriggerEvent &event)
{
    std::cout
        << "{\"schema\":\"dkfz-live-trigger-v1\""
        << ",\"sid\":\"" << json_escape(event.sid) << '"'
        << ",\"peak_frame_idx_50hz\":" << event.peak_frame_idx
        << ",\"peak_sample_idx_400hz\":" << event.peak_sample_idx
        << ",\"peak_time_ms\":";
    if (event.peak_time_ms.has_value())
    {
        std::cout << std::setprecision(17) << *event.peak_time_ms;
    }
    else
    {
        std::cout << "null";
    }
    std::cout
        << ",\"emitted_on_frame_idx_50hz\":" << event.emitted_on_frame_idx
        << ",\"emitted_on_sample_idx_400hz\":" << event.emitted_on_sample_idx
        << ",\"emitted_on_time_ms\":";
    if (event.emitted_on_time_ms.has_value())
    {
        std::cout << std::setprecision(17) << *event.emitted_on_time_ms;
    }
    else
    {
        std::cout << "null";
    }
    std::cout
        << ",\"decision_monotonic_ns\":";
    if (event.decision_monotonic_ns) std::cout << *event.decision_monotonic_ns; else std::cout << "null";
    std::cout << ",\"probability\":" << std::setprecision(9) << event.probability
        << "}\n";
}

void emit_probability(const InferenceResult &result, const std::string &sid)
{
    std::cout
        << "{\"schema\":\"dkfz-live-probability-v1\""
        << ",\"sid\":\"" << json_escape(sid) << '"'
        << ",\"frame_idx_50hz\":" << result.frame.frame_idx
        << ",\"start_sample_idx_400hz\":" << result.frame.start_sample_idx
        << ",\"end_sample_idx_400hz\":" << result.frame.end_sample_idx
        << ",\"time_ms\":";
    if (result.frame.end_time_ms.has_value())
    {
        std::cout << std::setprecision(17) << *result.frame.end_time_ms;
    }
    else
    {
        std::cout << "null";
    }
    std::cout
        << ",\"model_output\":" << std::setprecision(9) << result.model_output
        << ",\"probability\":" << result.probability
        << ",\"trigger_emitted_now\":"
        << (result.trigger.has_value() ? "true" : "false");
    if (result.observation_frame) {
        const auto &observed = *result.observation_frame;
        std::cout << ",\"prediction_delay_frames\":1"
                  << ",\"observed_frame_idx\":" << observed.frame_idx
                  << ",\"observed_raw_idx\":" << observed.start_sample_idx
                  << ",\"observed_raw_frame_end_idx\":" << observed.end_sample_idx
                  << ",\"observed_time_ms\":";
        if (observed.end_time_ms) std::cout << std::setprecision(17) << *observed.end_time_ms;
        else std::cout << "null";
    }
    if (result.recovery_probability)
    {
        std::cout << ",\"recovery_prob\":" << *result.recovery_probability
                  << ",\"recovery_emitted_now\":" << (result.recovery_emitted_now ? "true" : "false");
    }
    std::cout << "}\n";
}

int run(const Options &options)
{
    const auto manifest = dkfz::live::load_deployment_manifest(
        options.manifest_path);
    if (options.validate_only)
    {
        if (options.check_engine)
        {
            static_cast<void>(make_engine(manifest));
        }
        std::cout
            << "validated manifest model=" << manifest.model_name
            << " variant=" << manifest.model_variant
            << " input=[1," << manifest.window_size << ',' << manifest.input_dim
            << "] TensorRT="
            << (dkfz::live::tensorrt_backend_available() ? "enabled" : "disabled")
            << '\n';
        return 0;
    }

    dkfz::live::LiveInferencePipeline pipeline(manifest, make_engine(manifest));
    std::ifstream file;
    std::istream *input = &std::cin;
    if (!options.input_path.empty())
    {
        file.open(options.input_path);
        if (!file)
        {
            throw std::runtime_error(
                "Unable to open replay input: " + options.input_path.string());
        }
        input = &file;
    }

    std::cerr
        << "Native TensorRT live runtime ready: model=" << manifest.model_name
        << " variant=" << manifest.model_variant
        << " engine=" << manifest.engine_path
        << " raw_hz=" << manifest.framer.raw_hz
        << " frame_hz="
        << manifest.framer.raw_hz /
               static_cast<double>(manifest.framer.frame_size_raw)
        << " window_size=" << manifest.window_size
        << " threshold=" << manifest.manual_threshold.value_or(manifest.threshold) << '\n';

    std::string line;
    std::uint64_t line_number = 0;
    std::uint64_t samples = 0;
    std::uint64_t inferences = 0;
    std::uint64_t events = 0;
    std::int64_t sid_sample_index = 0;
    std::string last_sid = options.sid_override.empty() ? "0" : options.sid_override;
    while (std::getline(*input, line))
    {
        ++line_number;
        if (line.empty())
        {
            continue;
        }
        try
        {
            const auto sid = options.sid_override.empty()
                ? json_scalar_string(line, manifest.sid_field).value_or("0")
                : options.sid_override;
            if (sid != last_sid && samples > 0)
            {
                const auto event = manifest.prediction_delay_frames ? std::optional<TriggerEvent>{} : pipeline.flush(last_sid);
                if (event.has_value())
                {
                    emit_event(*event);
                    ++events;
                }
                pipeline.reset();
                sid_sample_index = 0;
            }
            last_sid = sid;
            const auto result = pipeline.process_sample(
                parse_sample(line, manifest, sid_sample_index), sid);
            ++samples;
            ++sid_sample_index;
            if (result.has_value())
            {
                ++inferences;
                if (options.emit_probabilities)
                {
                    emit_probability(*result, sid);
                }
                if (result->trigger.has_value())
                {
                    emit_event(*result->trigger);
                    ++events;
                }
            }
        }
        catch (const std::exception &error)
        {
            throw std::runtime_error(
                "Input line " + std::to_string(line_number) + ": " + error.what());
        }
        if (options.max_samples > 0 && samples >= options.max_samples)
        {
            break;
        }
    }

    if (samples > 0)
    {
        const auto event = pipeline.flush(last_sid);
        if (event.has_value())
        {
            emit_event(*event);
            ++events;
        }
    }
    std::cout.flush();
    std::cerr
        << "Native TensorRT live runtime finished: samples=" << samples
        << " inferences=" << inferences
        << " events=" << events << '\n';
    return 0;
}

} // namespace

int main(int argc, char **argv)
{
    try
    {
        return run(parse_args(argc, argv));
    }
    catch (const std::exception &error)
    {
        std::cerr << "native_live_runtime: " << error.what() << '\n';
        return 1;
    }
}
