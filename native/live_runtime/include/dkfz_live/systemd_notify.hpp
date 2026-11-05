/*
 * Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
 * SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
 * Developed at the German Cancer Research Center (DKFZ).
 */

#pragma once

#include <chrono>
#include <optional>
#include <string>

namespace dkfz::live
{

class SystemdNotifier
{
  public:
    SystemdNotifier();
    ~SystemdNotifier();
    SystemdNotifier(const SystemdNotifier &) = delete;
    SystemdNotifier &operator=(const SystemdNotifier &) = delete;

    bool available() const noexcept { return socket_fd_ >= 0; }
    bool send(const std::string &message) const noexcept;
    void ready(const std::string &status) const noexcept;
    void status(const std::string &status) const noexcept;
    void watchdog() const noexcept;
    void stopping(const std::string &status) const noexcept;
    std::optional<std::chrono::microseconds> watchdog_interval() const noexcept;

  private:
    int socket_fd_ = -1;
    std::string socket_path_;
    std::optional<std::chrono::microseconds> watchdog_interval_;
};

} // namespace dkfz::live
