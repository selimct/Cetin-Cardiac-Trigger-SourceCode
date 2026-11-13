/*
 * Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
 * SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
 * Developed at the German Cancer Research Center (DKFZ).
 */

#include "dkfz_uart/uart_stream.hpp"

#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
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

    std::string trim(std::string text)
    {
        const auto first = text.find_first_not_of(" \t\r\n");
        if (first == std::string::npos)
        {
            return {};
        }
        const auto last = text.find_last_not_of(" \t\r\n");
        return text.substr(first, last - first + 1);
    }

    template <typename T>
    T parse_unsigned(const std::string &value, const std::string &key)
    {
        if (value.empty() || value.front() == '-')
        {
            throw std::runtime_error("Invalid value for " + key + ": " + value);
        }
        std::size_t consumed = 0;
        const auto parsed = std::stoull(value, &consumed, 0);
        if (consumed != value.size() ||
            parsed > static_cast<unsigned long long>(std::numeric_limits<T>::max()))
        {
            throw std::runtime_error("Invalid value for " + key + ": " + value);
        }
        return static_cast<T>(parsed);
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

    struct Config
    {
        fs::path config_path = "config/uart_receiver.conf";
        dkfz::uart::ByteSourceConfig source;
        dkfz::uart::UartSignalOffsets uart_offsets;
        std::uint32_t sample_rate_hz = 400;
        std::uint64_t max_samples = 0;
    };

    void apply_config_value(
        Config &config,
        const std::string &key,
        const std::string &value)
    {
        if (key == "device")
        {
            config.source.device = value;
        }
        else if (key == "baud_rate")
        {
            config.source.baud_rate = parse_unsigned<std::uint32_t>(value, key);
        }
        else if (key == "sample_rate_hz")
        {
            config.sample_rate_hz = parse_unsigned<std::uint32_t>(value, key);
        }
        else if (key == "uart_ecg1_offset")
        {
            config.uart_offsets.ecg1 = parse_finite_double(value, key);
        }
        else if (key == "uart_ecg2_offset")
        {
            config.uart_offsets.ecg2 = parse_finite_double(value, key);
        }
        else if (key == "uart_puls_offset")
        {
            config.uart_offsets.puls = parse_finite_double(value, key);
        }
        else if (key == "max_capture_bytes")
        {
            config.source.max_capture_bytes =
                parse_unsigned<std::uint64_t>(value, key);
        }
    }

    void load_config_file(Config &config, const fs::path &path, bool required)
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
            apply_config_value(
                config,
                trim(line.substr(0, equals)),
                trim(line.substr(equals + 1)));
        }
    }

    void print_help(const char *program)
    {
        std::cout
            << "Usage: " << program << " [options]\n"
            << "Read J12 pin-10 and emit source-calibrated ECG/pulse JSONL.\n\n"
            << "  --config PATH             key=value configuration file\n"
            << "  --device PATH             UART device, for example /dev/ttyTHS1\n"
            << "  --baud-rate HZ            exact UART baud rate\n"
            << "  --sample-rate HZ          emitted ECG sample rate (default 400)\n"
            << "  --uart-ecg1-offset VALUE source ECG1 offset (default 2048)\n"
            << "  --uart-ecg2-offset VALUE source ECG2 offset (default 2048)\n"
            << "  --uart-puls-offset VALUE source pulse offset (default 2048)\n"
            << "  --capture-byte-file PATH  optionally save received UART bytes\n"
            << "  --max-capture-bytes N     stop capture at N bytes; 0 is unlimited\n"
            << "  --replay-byte-file PATH   parse saved bytes instead of the UART\n"
            << "  --max-samples N           stop after N samples; 0 is unlimited\n";
    }

    Config parse_args(int argc, char **argv)
    {
        Config config;
        bool explicit_config = false;
        for (int index = 1; index < argc; ++index)
        {
            if (std::string_view(argv[index]) == "--config")
            {
                if (index + 1 >= argc)
                {
                    throw std::runtime_error("Missing value for --config");
                }
                config.config_path = argv[++index];
                explicit_config = true;
            }
        }
        load_config_file(config, config.config_path, explicit_config);

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
            else if (option == "--device")
            {
                config.source.device = require_value(index, option);
            }
            else if (option == "--baud-rate")
            {
                config.source.baud_rate = parse_unsigned<std::uint32_t>(
                    require_value(index, option), option);
            }
            else if (option == "--sample-rate")
            {
                config.sample_rate_hz = parse_unsigned<std::uint32_t>(
                    require_value(index, option), option);
            }
            else if (option == "--uart-ecg1-offset")
            {
                config.uart_offsets.ecg1 = parse_finite_double(
                    require_value(index, option), option);
            }
            else if (option == "--uart-ecg2-offset")
            {
                config.uart_offsets.ecg2 = parse_finite_double(
                    require_value(index, option), option);
            }
            else if (option == "--uart-puls-offset")
            {
                config.uart_offsets.puls = parse_finite_double(
                    require_value(index, option), option);
            }
            else if (option == "--capture-byte-file")
            {
                config.source.capture_path = require_value(index, option);
            }
            else if (option == "--max-capture-bytes")
            {
                config.source.max_capture_bytes = parse_unsigned<std::uint64_t>(
                    require_value(index, option), option);
            }
            else if (option == "--replay-byte-file")
            {
                config.source.replay_path = require_value(index, option);
            }
            else if (option == "--max-samples")
            {
                config.max_samples = parse_unsigned<std::uint64_t>(
                    require_value(index, option), option);
            }
            else
            {
                throw std::runtime_error("Unknown option: " + option);
            }
        }
        if (config.sample_rate_hz == 0)
        {
            throw std::runtime_error("sample_rate_hz must be positive");
        }
        return config;
    }

    void emit_sample(
        const dkfz::uart::DecodedSample &sample,
        const dkfz::uart::UartSignalOffsets &offsets)
    {
        const auto calibrated = dkfz::uart::calibrate_signals(sample, offsets);
        std::cout
            << "{\"schema\":\"uart-pin10-sample-v1\""
            << ",\"raw_idx\":" << sample.raw_index
            << ",\"time_ms\":" << std::fixed << std::setprecision(6)
            << sample.time_ms
            << ",\"monotonic_ns\":" << sample.monotonic_ns
            << ",\"realtime_ns\":" << sample.realtime_ns
            << ",\"ecg1_raw\":" << calibrated.ecg1
            << ",\"ecg2_raw\":" << calibrated.ecg2
            << ",\"ecg3_raw\":" << calibrated.ecg3
            << ",\"puls_raw\":" << calibrated.puls
            << ",\"resp_raw\":" << (sample.resp_raw
                    ? std::to_string(*sample.resp_raw) : "null")
            << ",\"ext\":" << static_cast<unsigned>(sample.ext)
            << ",\"pr_sign\":" << static_cast<unsigned>(sample.pr_sign)
            << ",\"puls_updated\":" << (sample.puls_updated ? "true" : "false")
            << ",\"source_packet\":" << sample.source_packet
            << ",\"sample_in_packet\":" << sample.sample_in_packet
            << "}\n"
            << std::flush;
    }

    void print_status(
        const dkfz::uart::ByteSourceStats &source,
        const dkfz::uart::DecoderStats &decoder)
    {
        std::cerr
            << "status bytes=" << source.bytes_read
            << " capture_bytes=" << source.capture_bytes
            << " packets=" << decoder.groups
            << " samples=" << decoder.samples
            << " malformed_packets=" << decoder.malformed_groups
            << " malformed_samples="
            << decoder.malformed_groups * dkfz::uart::kSamplesPerGroup
            << " discarded_bytes=" << decoder.bytes_discarded << '\n';
    }

    int run(const Config &config)
    {
        std::signal(SIGINT, signal_handler);
        std::signal(SIGTERM, signal_handler);
        std::signal(SIGPIPE, signal_handler);

        dkfz::uart::UartByteDecoder decoder(
            config.sample_rate_hz,
            [&](const dkfz::uart::DecodedSample &sample)
            {
                emit_sample(sample, config.uart_offsets);
                return std::cout.good() &&
                       !g_stop.load(std::memory_order_relaxed) &&
                       (config.max_samples == 0 ||
                        sample.raw_index + 1 < config.max_samples);
            });
        auto last_status = std::chrono::steady_clock::now();

        std::cerr
            << "UART receiver stream ready: source="
            << (config.source.replay_path.empty()
                    ? config.source.device.string()
                    : config.source.replay_path.string())
            << " baud=" << config.source.baud_rate
            << " format=8E1 output=JSONL capture="
            << (config.source.capture_path.empty()
                    ? "off"
                    : config.source.capture_path.string())
            << " uart_ecg1_offset=" << config.uart_offsets.ecg1
            << " uart_ecg2_offset=" << config.uart_offsets.ecg2
            << " uart_puls_offset=" << config.uart_offsets.puls
            << '\n';

        const auto source_stats = dkfz::uart::run_byte_source(
            config.source,
            [&](const std::uint8_t *bytes, std::size_t size)
            {
                decoder.process(bytes, size);
                return !decoder.stopped() &&
                       !g_stop.load(std::memory_order_relaxed);
            },
            g_stop,
            [&](const dkfz::uart::ByteSourceStats &stats)
            {
                const auto now = std::chrono::steady_clock::now();
                if (now - last_status >= std::chrono::seconds(5))
                {
                    print_status(stats, decoder.stats());
                    last_status = now;
                }
            });

        const auto &decoder_stats = decoder.stats();
        std::cerr
            << "UART receiver stream finished: bytes=" << source_stats.bytes_read
            << " capture_bytes=" << source_stats.capture_bytes
            << " capture_limit_reached="
            << (source_stats.capture_limit_reached ? "yes" : "no")
            << " discarded=" << decoder_stats.bytes_discarded
            << " packets=" << decoder_stats.groups
            << " malformed_packets=" << decoder_stats.malformed_groups
            << " malformed_samples="
            << decoder_stats.malformed_groups * dkfz::uart::kSamplesPerGroup
            << " samples=" << decoder_stats.samples
            << " pulse_queue_high_water=" << decoder_stats.pulse_queue_high_water
            << '\n';
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
        std::cerr << "fatal: " << error.what() << '\n';
        return 1;
    }
}
