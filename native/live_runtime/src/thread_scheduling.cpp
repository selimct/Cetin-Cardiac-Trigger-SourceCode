/*
 * Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
 * SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
 * Developed at the German Cancer Research Center (DKFZ).
 */
#include "dkfz_live/thread_scheduling.hpp"

#include <cerrno>
#include <cstring>
#include <pthread.h>
#include <sched.h>
#include <stdexcept>
#include <string>
#include <sys/resource.h>

namespace dkfz::live
{
void validate_thread_scheduling(const ThreadSchedulingConfig &config)
{
    if (config.realtime_priority < 0 || config.realtime_priority > 20)
        throw std::invalid_argument("realtime_priority must be within [0, 20]");
    if (config.require_realtime && config.realtime_priority == 0)
        throw std::invalid_argument("require_realtime needs realtime_priority > 0");
}

void check_required_realtime_limit(const ThreadSchedulingConfig &config)
{
    validate_thread_scheduling(config);
    if (!config.require_realtime) return;
    rlimit limit{};
    if (getrlimit(RLIMIT_RTPRIO, &limit) != 0)
        throw std::runtime_error("Cannot inspect RLIMIT_RTPRIO");
    if (limit.rlim_cur < static_cast<rlim_t>(config.realtime_priority))
        throw std::runtime_error(
            "Required realtime priority exceeds RLIMIT_RTPRIO; use the service "
            "profile with LimitRTPRIO=20 (no scheduler change was applied)");
    // A required profile must also bound a runaway worker. Do not change
    // process limits here: systemd/the isolated launcher supplies them.
    if (getrlimit(RLIMIT_RTTIME, &limit) != 0 || limit.rlim_max == RLIM_INFINITY ||
        limit.rlim_cur == 0 || limit.rlim_max > 500'000)
        throw std::runtime_error(
            "Required realtime scheduling needs a finite RLIMIT_RTTIME <=500ms; "
            "the service profile uses LimitRTTIME=200ms");
}

void configure_worker_scheduling(
    const ThreadSchedulingConfig &config, const char *name, DiagnosticLogger &logger)
{
    validate_thread_scheduling(config);
    const int name_error = pthread_setname_np(pthread_self(), name);
    int error = 0;
    std::string prerequisite_error;
    if (config.realtime_priority > 0)
    {
        auto bounded = config;
        bounded.require_realtime = true;
        try { check_required_realtime_limit(bounded); }
        catch (const std::runtime_error &failure)
        {
            prerequisite_error = failure.what();
            error = EPERM;
        }
    }
    if (config.realtime_priority > 0 && !error)
    {
        // Normal-priority main/CUDA initialization/supervision is retained.
        // RR time-slices with equal-priority peers; the process RT CPU limit
        // and existing watchdog bound a worker that stops blocking normally.
        sched_param requested{};
        requested.sched_priority = config.realtime_priority;
        error = pthread_setschedparam(pthread_self(), SCHED_RR, &requested);
    }
    int policy = SCHED_OTHER;
    sched_param actual{};
    const int query_error = pthread_getschedparam(pthread_self(), &policy, &actual);
    if (!error) error = query_error;
    const bool applied = !error && policy == SCHED_RR &&
                         actual.sched_priority == config.realtime_priority;
    const bool degraded = config.realtime_priority > 0 && !applied;
    logger.emit(degraded ? DiagnosticSeverity::warning : DiagnosticSeverity::info,
        "worker_scheduling", "scheduling",
        degraded ? "Requested realtime scheduling unavailable" : "Worker scheduling verified",
        {{"thread", std::string(name)},
         {"requested_realtime_priority", static_cast<std::int64_t>(config.realtime_priority)},
         {"policy", std::string(query_error ? "unknown" :
             policy == SCHED_RR ? "rr" : policy == SCHED_FIFO ? "fifo" :
             policy == SCHED_OTHER ? "other" : "other_policy")},
         {"priority", static_cast<std::int64_t>(actual.sched_priority)},
         {"realtime_applied", applied}, {"degraded", degraded},
         {"error", static_cast<std::int64_t>(error)},
         {"name_error", static_cast<std::int64_t>(name_error)},
         {"prerequisite_error", prerequisite_error}});
    if (degraded && config.require_realtime)
        throw WorkerSchedulingError(std::string(name) + " required SCHED_RR unavailable: " +
            (error ? std::strerror(error) : "policy/priority mismatch"));
}
} // namespace dkfz::live
