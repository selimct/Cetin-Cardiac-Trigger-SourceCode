/*
 * Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
 * SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
 * Developed at the German Cancer Research Center (DKFZ).
 */
#pragma once

#include "dkfz_live/diagnostics.hpp"
#include <stdexcept>

namespace dkfz::live
{
class WorkerSchedulingError : public std::runtime_error
{
  public:
    using std::runtime_error::runtime_error;
};

struct ThreadSchedulingConfig
{
    // Zero preserves the inherited scheduler. Standalone/replay defaults stay
    // unprivileged; the service explicitly requests a bounded RR priority.
    int realtime_priority = 0;
    bool require_realtime = false;
};

void validate_thread_scheduling(const ThreadSchedulingConfig &config);
// Read-only prerequisite check. Does not change any thread policy.
void check_required_realtime_limit(const ThreadSchedulingConfig &config);
// Apply once inside each acquisition/inference/output worker, before its loop.
// Names and actual policies are reported even when RT is disabled/unavailable.
void configure_worker_scheduling(
    const ThreadSchedulingConfig &config, const char *name, DiagnosticLogger &logger);
} // namespace dkfz::live
