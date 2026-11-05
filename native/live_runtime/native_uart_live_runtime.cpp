/*
 * Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
 * SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
 * Developed at the German Cancer Research Center (DKFZ).
 */

#include "dkfz_live/deployment_manifest.hpp"
#include "dkfz_live/diagnostics.hpp"
#include "dkfz_live/inference_engine.hpp"
#include "dkfz_live/integrated_runtime.hpp"

#include <atomic>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>

namespace fs = std::filesystem;

namespace
{

std::atomic<bool> g_stop{false};

void signal_handler(int)
{
    g_stop.store(true, std::memory_order_relaxed);
}

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

std::uint64_t parse_unsigned(
    const std::string &value,
    const std::string &key)
{
    if (value.empty() || value.front() == '-')
    {
        throw std::runtime_error(
            "Invalid unsigned integer for " + key + ": " + value);
    }
    std::size_t consumed = 0;
    const auto parsed = std::stoull(value, &consumed, 10);
    if (consumed != value.size())
    {
        throw std::runtime_error("Invalid unsigned integer for " + key + ": " + value);
    }
    return parsed;
}

template <typename T>
T parse_unsigned_as(const std::string &value, const std::string &key)
{
    const auto parsed = parse_unsigned(value, key);
    if (parsed > static_cast<std::uint64_t>(std::numeric_limits<T>::max()))
    {
        throw std::runtime_error(
            "Unsigned integer is too large for " + key + ": " + value);
    }
    return static_cast<T>(parsed);
}

bool parse_bool(const std::string &value, const std::string &key)
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

double parse_finite_double(
    const std::string &value,
    const std::string &key)
{
    std::size_t consumed = 0;
    const auto parsed = std::stod(value, &consumed);
    if (consumed != value.size() || !std::isfinite(parsed))
    {
        throw std::runtime_error(
            "Invalid finite number for " + key + ": " + value);
    }
    return parsed;
}

std::uint64_t milliseconds_to_ns(
    const std::string &value,
    const std::string &key)
{
    const auto milliseconds = parse_unsigned(value, key);
    if (milliseconds >
        std::numeric_limits<std::uint64_t>::max() / 1'000'000ULL)
    {
        throw std::runtime_error("Millisecond value is too large for " + key);
    }
    return milliseconds * 1'000'000ULL;
}

std::uint64_t microseconds_to_ns(
    const std::string &value,
    const std::string &key)
{
    const auto microseconds = parse_unsigned(value, key);
    if (microseconds >
        std::numeric_limits<std::uint64_t>::max() / 1'000ULL)
    {
        throw std::runtime_error("Microsecond value is too large for " + key);
    }
    return microseconds * 1'000ULL;
}

struct Options
{
    fs::path config_path = "config/native_uart_live.conf";
    fs::path manifest_path;
    dkfz::live::IntegratedRuntimeConfig runtime;
    bool validate_only = false;
    bool check_engine = false;
};

void apply_config_value(
    Options &options,
    const std::string &key,
    const std::string &value)
{
    if (key == "manifest_path")
    {
        options.manifest_path = value;
    }
    else if (key == "threshold_override_path")
    {
        options.runtime.threshold_override_path = value;
    }
    else if (key == "early_emission")
    {
        options.runtime.early_emission = parse_bool(value, key);
    }
    else if (key == "device")
    {
        options.runtime.source.device = value;
    }
    else if (key == "baud_rate")
    {
        options.runtime.source.baud_rate =
            parse_unsigned_as<std::uint32_t>(value, key);
    }
    else if (key == "sample_rate_hz")
    {
        options.runtime.sample_rate_hz =
            parse_unsigned_as<std::uint32_t>(value, key);
    }
    else if (key == "uart_ecg1_offset")
    {
        options.runtime.uart_offsets.ecg1 = parse_finite_double(value, key);
    }
    else if (key == "uart_ecg2_offset")
    {
        options.runtime.uart_offsets.ecg2 = parse_finite_double(value, key);
    }
    else if (key == "uart_puls_offset")
    {
        options.runtime.uart_offsets.puls = parse_finite_double(value, key);
    }
    else if (key == "capture_path")
    {
        options.runtime.source.capture_path = value;
    }
    else if (key == "max_capture_bytes")
    {
        options.runtime.source.max_capture_bytes = parse_unsigned(value, key);
    }
    else if (key == "replay_path")
    {
        options.runtime.source.replay_path = value;
    }
    else if (key == "max_samples")
    {
        options.runtime.max_samples = parse_unsigned(value, key);
    }
    else if (key == "queue_capacity")
    {
        options.runtime.queue_capacity =
            parse_unsigned_as<std::size_t>(value, key);
    }
    else if (key == "status_interval_ms")
    {
        options.runtime.status_interval_ns = milliseconds_to_ns(value, key);
    }
    else if (key == "paused_status_interval_ms")
    {
        options.runtime.paused_status_interval_ns =
            milliseconds_to_ns(value, key);
    }
    else if (key == "sid")
    {
        options.runtime.sid = value;
    }
    else if (key == "input_reset_enabled")
    {
        options.runtime.input_reset.enabled = parse_bool(value, key);
    }
    else if (key == "puls_baseline_reset_ms")
    {
        options.runtime.input_reset.puls_baseline_reset_ms =
            parse_unsigned(value, key);
    }
    else if (key == "ecg_baseline_reset_ms")
    {
        options.runtime.input_reset.ecg_baseline_reset_ms =
            parse_unsigned(value, key);
    }
    else if (key == "stuck_signal_reset_ms")
    {
        options.runtime.input_reset.stuck_signal_reset_ms =
            parse_unsigned(value, key);
    }
    else if (key == "input_recovery_ms")
    {
        options.runtime.input_reset.recovery_ms = parse_unsigned(value, key);
    }
    else if (key == "emit_probabilities")
    {
        options.runtime.emit_probabilities = parse_bool(value, key);
    }
    else if (key == "trigger_output")
    {
        options.runtime.output.mode =
            dkfz::live::parse_runtime_output_mode(value);
    }
    else if (key == "trigger_output_queue_capacity")
    {
        options.runtime.output.queue_capacity =
            parse_unsigned_as<std::size_t>(value, key);
    }
    else if (key == "trigger_gpio_chip")
    {
        options.runtime.output.gpio.chip = value;
    }
    else if (key == "trigger_gpio_line_offset")
    {
        options.runtime.output.gpio.line_offset =
            parse_unsigned_as<std::uint32_t>(value, key);
    }
    else if (key == "trigger_gpio_active_high")
    {
        options.runtime.output.gpio.active_high = parse_bool(value, key);
    }
    else if (key == "trigger_gpio_pulse_width_us")
    {
        options.runtime.output.gpio.pulse_width_ns =
            microseconds_to_ns(value, key);
    }
    else if (key == "trigger_gpio_idle_width_us")
    {
        options.runtime.output.gpio.idle_width_ns =
            microseconds_to_ns(value, key);
    }
    else if (key == "trigger_gpio_consumer")
    {
        options.runtime.output.gpio.consumer = value;
    }
    else if (key == "allow_gpio_during_replay")
    {
        options.runtime.output.allow_gpio_during_replay =
            parse_bool(value, key);
    }
    else if (key == "watchdog_enabled")
    {
        options.runtime.watchdog.enabled = parse_bool(value, key);
    }
    else if (key == "watchdog_startup_grace_ms")
    {
        options.runtime.watchdog.startup_grace_ns = milliseconds_to_ns(value, key);
    }
    else if (key == "uart_timeout_ms")
    {
        options.runtime.watchdog.uart_timeout_ns = milliseconds_to_ns(value, key);
    }
    else if (key == "sample_timeout_ms")
    {
        options.runtime.watchdog.sample_timeout_ns = milliseconds_to_ns(value, key);
    }
    else if (key == "inference_timeout_ms")
    {
        options.runtime.watchdog.inference_timeout_ns = milliseconds_to_ns(value, key);
    }
    else if (key == "output_timeout_ms")
    {
        options.runtime.watchdog.output_timeout_ns =
            milliseconds_to_ns(value, key);
    }
    else if (key == "hard_exit_on_watchdog")
    {
        options.runtime.hard_exit_on_watchdog = parse_bool(value, key);
    }
    else
    {
        throw std::runtime_error("Unknown integrated runtime config key: " + key);
    }
}

void load_config(Options &options, const fs::path &path, bool required)
{
    std::ifstream input(path);
    if (!input)
    {
        if (required)
        {
            throw std::runtime_error("Unable to open config file: " + path.string());
        }
        return;
    }
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
        try
        {
            apply_config_value(
                options,
                trim(line.substr(0, equals)),
                trim(line.substr(equals + 1)));
        }
        catch (const std::exception &error)
        {
            throw std::runtime_error(
                path.string() + ":" + std::to_string(line_number) + ": " +
                error.what());
        }
    }
}

void print_help(const char *program)
{
    std::cout
        << "Usage: " << program << " [options]\n"
        << "Run UART acquisition, C++ preprocessing, TensorRT, and triggering "
           "in one process.\n\n"
        << "  --config PATH                integrated key=value config\n"
        << "  --manifest PATH              native model deployment manifest\n"
        << "  --threshold-override PATH    watch a model-bound threshold override file\n"
        << "  --early-emission             emit on crossing the higher manifest threshold (default)\n"
        << "  --no-early-emission          wait for the active detection threshold\n"
        << "  --device PATH                UART device\n"
        << "  --baud-rate HZ               exact UART baud rate\n"
        << "  --sample-rate HZ             raw sample rate\n"
        << "  --uart-ecg1-offset VALUE    source ECG1 offset\n"
        << "  --uart-ecg2-offset VALUE    source ECG2 offset\n"
        << "  --uart-puls-offset VALUE    source pulse offset\n"
        << "  --capture-byte-file PATH     optional raw UART capture\n"
        << "  --max-capture-bytes N        capture size limit\n"
        << "  --replay-byte-file PATH      replay raw UART bytes\n"
        << "  --max-samples N              bounded run; 0 is unlimited\n"
        << "  --queue-capacity N           acquisition/inference queue size\n"
        << "  --status-interval-ms N       structured status interval\n"
        << "  --paused-status-interval-ms N\n"
        << "                               status interval while input is paused\n"
        << "  --puls-baseline-reset-ms N  pulse continuously at raw 2048\n"
        << "  --ecg-baseline-reset-ms N   both ECG channels at raw 2048\n"
        << "  --stuck-signal-reset-ms N   all raw signals unchanged; 0 disables\n"
        << "  --input-recovery-ms N       stable input required before resume\n"
        << "  --disable-input-reset       disable automatic input pause/recovery\n"
        << "  --trigger-output MODE        none, jsonl, gpio, or gpio+jsonl\n"
        << "  --trigger-output-queue-capacity N\n"
        << "  --trigger-gpio-chip CHIP     /dev path or kernel chip label\n"
        << "  --trigger-gpio-line-offset N GPIO line offset on the chip\n"
        << "  --trigger-gpio-active-high   active-HIGH output pulse\n"
        << "  --trigger-gpio-active-low    active-LOW output pulse\n"
        << "  --trigger-gpio-pulse-width-us N\n"
        << "  --trigger-gpio-idle-width-us N minimum inactive interval\n"
        << "  --trigger-gpio-consumer NAME GPIO line ownership label\n"
        << "  --allow-gpio-during-replay   explicitly enable physical replay pulses\n"
        << "  --watchdog-startup-grace-ms N\n"
        << "  --uart-timeout-ms N\n"
        << "  --sample-timeout-ms N\n"
        << "  --inference-timeout-ms N\n"
        << "  --output-timeout-ms N\n"
        << "  --disable-watchdog           disable internal deadlines\n"
        << "  --emit-probabilities         JSONL parity/debug frame records\n"
        << "  --sid VALUE                  fixed stream identifier\n"
        << "  --validate-only              validate config without running\n"
        << "  --check-engine               also load TensorRT with --validate-only\n";
}

Options parse_args(int argc, char **argv)
{
    Options options;
    bool explicit_config = false;
    for (int index = 1; index < argc; ++index)
    {
        if (std::string_view(argv[index]) == "--config")
        {
            if (index + 1 >= argc)
            {
                throw std::runtime_error("Missing value for --config");
            }
            options.config_path = argv[++index];
            explicit_config = true;
        }
    }
    load_config(options, options.config_path, explicit_config);

    auto require_value = [&](int &index, const std::string &option)
    {
        if (index + 1 >= argc)
        {
            throw std::runtime_error("Missing value for " + option);
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
        if (option == "--config")
        {
            ++index;
        }
        else if (option == "--manifest")
        {
            options.manifest_path = require_value(index, option);
        }
        else if (option == "--threshold-override")
        {
            options.runtime.threshold_override_path = require_value(index, option);
        }
        else if (option == "--early-emission" || option == "--no-early-emission")
        {
            options.runtime.early_emission = option == "--early-emission";
        }
        else if (option == "--device")
        {
            options.runtime.source.device = require_value(index, option);
        }
        else if (option == "--baud-rate")
        {
            options.runtime.source.baud_rate = parse_unsigned_as<std::uint32_t>(
                require_value(index, option), option);
        }
        else if (option == "--sample-rate")
        {
            options.runtime.sample_rate_hz = parse_unsigned_as<std::uint32_t>(
                require_value(index, option), option);
        }
        else if (option == "--uart-ecg1-offset")
        {
            options.runtime.uart_offsets.ecg1 = parse_finite_double(
                require_value(index, option), option);
        }
        else if (option == "--uart-ecg2-offset")
        {
            options.runtime.uart_offsets.ecg2 = parse_finite_double(
                require_value(index, option), option);
        }
        else if (option == "--uart-puls-offset")
        {
            options.runtime.uart_offsets.puls = parse_finite_double(
                require_value(index, option), option);
        }
        else if (option == "--capture-byte-file")
        {
            options.runtime.source.capture_path = require_value(index, option);
        }
        else if (option == "--max-capture-bytes")
        {
            options.runtime.source.max_capture_bytes = parse_unsigned(
                require_value(index, option), option);
        }
        else if (option == "--replay-byte-file")
        {
            options.runtime.source.replay_path = require_value(index, option);
        }
        else if (option == "--max-samples")
        {
            options.runtime.max_samples = parse_unsigned(
                require_value(index, option), option);
        }
        else if (option == "--queue-capacity")
        {
            options.runtime.queue_capacity = parse_unsigned_as<std::size_t>(
                require_value(index, option), option);
        }
        else if (option == "--status-interval-ms")
        {
            options.runtime.status_interval_ns = milliseconds_to_ns(
                require_value(index, option), option);
        }
        else if (option == "--paused-status-interval-ms")
        {
            options.runtime.paused_status_interval_ns = milliseconds_to_ns(
                require_value(index, option), option);
        }
        else if (option == "--puls-baseline-reset-ms")
        {
            options.runtime.input_reset.puls_baseline_reset_ms =
                parse_unsigned(require_value(index, option), option);
        }
        else if (option == "--ecg-baseline-reset-ms")
        {
            options.runtime.input_reset.ecg_baseline_reset_ms =
                parse_unsigned(require_value(index, option), option);
        }
        else if (option == "--stuck-signal-reset-ms")
        {
            options.runtime.input_reset.stuck_signal_reset_ms =
                parse_unsigned(require_value(index, option), option);
        }
        else if (option == "--input-recovery-ms")
        {
            options.runtime.input_reset.recovery_ms =
                parse_unsigned(require_value(index, option), option);
        }
        else if (option == "--disable-input-reset")
        {
            options.runtime.input_reset.enabled = false;
        }
        else if (option == "--trigger-output")
        {
            options.runtime.output.mode =
                dkfz::live::parse_runtime_output_mode(
                    require_value(index, option));
        }
        else if (option == "--trigger-output-queue-capacity")
        {
            options.runtime.output.queue_capacity =
                parse_unsigned_as<std::size_t>(
                    require_value(index, option), option);
        }
        else if (option == "--trigger-gpio-chip")
        {
            options.runtime.output.gpio.chip = require_value(index, option);
        }
        else if (option == "--trigger-gpio-line-offset")
        {
            options.runtime.output.gpio.line_offset =
                parse_unsigned_as<std::uint32_t>(
                    require_value(index, option), option);
        }
        else if (option == "--trigger-gpio-active-high")
        {
            options.runtime.output.gpio.active_high = true;
        }
        else if (option == "--trigger-gpio-active-low")
        {
            options.runtime.output.gpio.active_high = false;
        }
        else if (option == "--trigger-gpio-pulse-width-us")
        {
            options.runtime.output.gpio.pulse_width_ns =
                microseconds_to_ns(require_value(index, option), option);
        }
        else if (option == "--trigger-gpio-idle-width-us")
        {
            options.runtime.output.gpio.idle_width_ns =
                microseconds_to_ns(require_value(index, option), option);
        }
        else if (option == "--trigger-gpio-consumer")
        {
            options.runtime.output.gpio.consumer = require_value(index, option);
        }
        else if (option == "--allow-gpio-during-replay")
        {
            options.runtime.output.allow_gpio_during_replay = true;
        }
        else if (option == "--watchdog-startup-grace-ms")
        {
            options.runtime.watchdog.startup_grace_ns = milliseconds_to_ns(
                require_value(index, option), option);
        }
        else if (option == "--uart-timeout-ms")
        {
            options.runtime.watchdog.uart_timeout_ns = milliseconds_to_ns(
                require_value(index, option), option);
        }
        else if (option == "--sample-timeout-ms")
        {
            options.runtime.watchdog.sample_timeout_ns = milliseconds_to_ns(
                require_value(index, option), option);
        }
        else if (option == "--inference-timeout-ms")
        {
            options.runtime.watchdog.inference_timeout_ns = milliseconds_to_ns(
                require_value(index, option), option);
        }
        else if (option == "--output-timeout-ms")
        {
            options.runtime.watchdog.output_timeout_ns = milliseconds_to_ns(
                require_value(index, option), option);
        }
        else if (option == "--disable-watchdog")
        {
            options.runtime.watchdog.enabled = false;
        }
        else if (option == "--emit-probabilities")
        {
            options.runtime.emit_probabilities = true;
        }
        else if (option == "--sid")
        {
            options.runtime.sid = require_value(index, option);
        }
        else if (option == "--validate-only")
        {
            options.validate_only = true;
        }
        else if (option == "--check-engine")
        {
            options.check_engine = true;
        }
        else
        {
            throw std::runtime_error("Unknown option: " + option);
        }
    }
    if (options.manifest_path.empty())
    {
        throw std::runtime_error("manifest_path or --manifest PATH is required");
    }
    if (options.check_engine && !options.validate_only)
    {
        throw std::runtime_error("--check-engine requires --validate-only");
    }
    if (!options.runtime.source.capture_path.empty() &&
        !options.runtime.source.replay_path.empty())
    {
        throw std::runtime_error(
            "capture_byte_file cannot be combined with replay_byte_file");
    }
    options.runtime.watchdog.monitor_uart =
        options.runtime.source.replay_path.empty();
    return options;
}

void validate_integrated_compatibility(
    const Options &options,
    const dkfz::live::DeploymentManifest &manifest)
{
    if (options.runtime.source.baud_rate == 0)
    {
        throw std::runtime_error("baud_rate must be positive");
    }
    if (options.runtime.sample_rate_hz == 0)
    {
        throw std::runtime_error("sample_rate_hz must be positive");
    }
    if (options.runtime.source.replay_path.empty() &&
        options.runtime.source.device.empty())
    {
        throw std::runtime_error("device must not be empty for live acquisition");
    }
    if (std::abs(
            manifest.framer.raw_hz -
            static_cast<double>(options.runtime.sample_rate_hz)) > 1e-9)
    {
        throw std::runtime_error(
            "sample_rate_hz does not match manifest raw_hz");
    }
    if (options.runtime.queue_capacity == 0)
    {
        throw std::runtime_error("queue_capacity must be positive");
    }
    dkfz::live::validate_runtime_output_config(
        options.runtime.output,
        !options.runtime.source.replay_path.empty());
    if (options.runtime.emit_probabilities &&
        !dkfz::live::runtime_output_has_jsonl(options.runtime.output.mode))
    {
        throw std::runtime_error(
            "emit_probabilities requires trigger_output=jsonl or gpio+jsonl");
    }
    if (dkfz::live::runtime_output_has_gpio(options.runtime.output.mode) &&
        options.runtime.watchdog.enabled &&
        options.runtime.watchdog.output_timeout_ns > 0 &&
        (options.runtime.watchdog.output_timeout_ns <=
             options.runtime.output.gpio.pulse_width_ns ||
         options.runtime.watchdog.output_timeout_ns -
                 options.runtime.output.gpio.pulse_width_ns <=
             options.runtime.output.gpio.idle_width_ns))
    {
        throw std::runtime_error(
            "output_timeout_ms must exceed the GPIO pulse plus idle width");
    }
    if (manifest.framer.ecg1_offset != 0.0 ||
        manifest.framer.ecg2_offset != 0.0 ||
        manifest.framer.ecg3_offset != 0.0 ||
        manifest.framer.puls_offset != 0.0)
    {
        throw std::runtime_error(
            "Integrated UART input removes its source-specific offsets; "
            "deployment manifest signal offsets must be zero");
    }
}

std::unique_ptr<dkfz::live::LogitEngine> make_engine(
    const dkfz::live::DeploymentManifest &manifest)
{
    dkfz::live::verify_deployment_engine_fingerprint(manifest);
    return dkfz::live::make_tensorrt_logit_engine(
        manifest.engine_path,
        manifest.input_tensor,
        manifest.output_tensor,
        manifest.window_size,
        manifest.input_dim);
}

} // namespace

int main(int argc, char **argv)
{
    dkfz::live::DiagnosticLogger logger(std::cerr);
    try
    {
        const auto options = parse_args(argc, argv);
        const auto manifest = dkfz::live::load_deployment_manifest(
            options.manifest_path);
        validate_integrated_compatibility(options, manifest);
        if (options.validate_only)
        {
            if (options.check_engine)
            {
                static_cast<void>(make_engine(manifest));
            }
            logger.emit(
                dkfz::live::DiagnosticSeverity::info,
                "configuration_valid",
                "configuration",
                "Integrated runtime configuration is valid",
                {
                    {"model", manifest.model_name},
                    {"manifest", options.manifest_path.string()},
                    {"engine_checked", options.check_engine},
                    {"output_mode", dkfz::live::runtime_output_mode_name(
                         options.runtime.output.mode)},
                });
            return 0;
        }

        std::unique_ptr<dkfz::live::LogitEngine> engine;
        try
        {
            engine = make_engine(manifest);
        }
        catch (const std::exception &error)
        {
            logger.emit(
                dkfz::live::DiagnosticSeverity::fatal,
                "engine_initialization_failed",
                "tensorrt",
                error.what());
            return static_cast<int>(dkfz::live::IntegratedExitCode::inference);
        }

        std::signal(SIGINT, signal_handler);
        std::signal(SIGTERM, signal_handler);
        std::signal(SIGPIPE, signal_handler);
        dkfz::live::IntegratedLiveRuntime runtime(
            manifest,
            std::move(engine),
            options.runtime,
            std::cout,
            std::cerr);
        const auto summary = runtime.run(g_stop);
        return static_cast<int>(summary.exit_code);
    }
    catch (const std::exception &error)
    {
        logger.emit(
            dkfz::live::DiagnosticSeverity::fatal,
            "configuration_failed",
            "configuration",
            error.what());
        return static_cast<int>(dkfz::live::IntegratedExitCode::configuration);
    }
}
