/*
 * Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
 * SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
 * Developed at the German Cancer Research Center (DKFZ).
 */

#pragma once

#include <cstdint>
#include <initializer_list>
#include <iosfwd>
#include <mutex>
#include <string>
#include <variant>
#include <vector>

namespace dkfz::live
{

enum class DiagnosticSeverity
{
    debug,
    info,
    warning,
    error,
    fatal,
};

using DiagnosticValue =
    std::variant<std::string, std::int64_t, std::uint64_t, double, bool>;

struct DiagnosticField
{
    std::string name;
    DiagnosticValue value;
};

class DiagnosticLogger
{
  public:
    explicit DiagnosticLogger(std::ostream &output);

    void emit(
        DiagnosticSeverity severity,
        const std::string &code,
        const std::string &component,
        const std::string &message,
        std::initializer_list<DiagnosticField> fields = {});

  private:
    std::ostream &output_;
    std::mutex mutex_;
};

std::string diagnostic_severity_name(DiagnosticSeverity severity);

} // namespace dkfz::live
