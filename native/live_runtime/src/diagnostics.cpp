/*
 * Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
 * SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
 * Developed at the German Cancer Research Center (DKFZ).
 */

#include "dkfz_live/diagnostics.hpp"

#include <cerrno>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <ostream>
#include <stdexcept>
#include <time.h>
#include <type_traits>

namespace dkfz::live
{
namespace
{

std::uint64_t clock_ns(clockid_t clock_id)
{
    timespec value{};
    if (::clock_gettime(clock_id, &value) != 0)
    {
        throw std::runtime_error(
            "clock_gettime failed: " + std::string(std::strerror(errno)));
    }
    return static_cast<std::uint64_t>(value.tv_sec) * 1'000'000'000ULL +
           static_cast<std::uint64_t>(value.tv_nsec);
}

std::string escape_json(const std::string &value)
{
    std::string result;
    result.reserve(value.size() + 8);
    constexpr char hex[] = "0123456789abcdef";
    for (const auto character : value)
    {
        const auto byte = static_cast<unsigned char>(character);
        switch (character)
        {
        case '"':
            result += "\\\"";
            break;
        case '\\':
            result += "\\\\";
            break;
        case '\b':
            result += "\\b";
            break;
        case '\f':
            result += "\\f";
            break;
        case '\n':
            result += "\\n";
            break;
        case '\r':
            result += "\\r";
            break;
        case '\t':
            result += "\\t";
            break;
        default:
            if (byte < 0x20U)
            {
                result += "\\u00";
                result += hex[(byte >> 4U) & 0x0FU];
                result += hex[byte & 0x0FU];
            }
            else
            {
                result += character;
            }
        }
    }
    return result;
}

void write_value(std::ostream &output, const DiagnosticValue &value)
{
    std::visit(
        [&](const auto &typed)
        {
            using ValueType = std::decay_t<decltype(typed)>;
            if constexpr (std::is_same_v<ValueType, std::string>)
            {
                output << '"' << escape_json(typed) << '"';
            }
            else if constexpr (std::is_same_v<ValueType, bool>)
            {
                output << (typed ? "true" : "false");
            }
            else if constexpr (std::is_same_v<ValueType, double>)
            {
                if (std::isfinite(typed))
                {
                    output << std::setprecision(17) << typed;
                }
                else
                {
                    output << "null";
                }
            }
            else
            {
                output << typed;
            }
        },
        value);
}

} // namespace

std::string diagnostic_severity_name(DiagnosticSeverity severity)
{
    switch (severity)
    {
    case DiagnosticSeverity::debug:
        return "debug";
    case DiagnosticSeverity::info:
        return "info";
    case DiagnosticSeverity::warning:
        return "warning";
    case DiagnosticSeverity::error:
        return "error";
    case DiagnosticSeverity::fatal:
        return "fatal";
    }
    return "unknown";
}

DiagnosticLogger::DiagnosticLogger(std::ostream &output) : output_(output) {}

void DiagnosticLogger::emit(
    DiagnosticSeverity severity,
    const std::string &code,
    const std::string &component,
    const std::string &message,
    std::initializer_list<DiagnosticField> fields)
{
    const auto monotonic_ns = clock_ns(CLOCK_MONOTONIC);
    const auto realtime_ns = clock_ns(CLOCK_REALTIME);
    std::lock_guard<std::mutex> lock(mutex_);
    output_
        << "{\"schema\":\"dkfz-live-diagnostic-v1\""
        << ",\"severity\":\"" << diagnostic_severity_name(severity) << '"'
        << ",\"code\":\"" << escape_json(code) << '"'
        << ",\"component\":\"" << escape_json(component) << '"'
        << ",\"monotonic_ns\":" << monotonic_ns
        << ",\"realtime_ns\":" << realtime_ns
        << ",\"message\":\"" << escape_json(message) << '"';
    for (const auto &field : fields)
    {
        output_ << ",\"" << escape_json(field.name) << "\":";
        write_value(output_, field.value);
    }
    output_ << "}\n" << std::flush;
}

} // namespace dkfz::live
