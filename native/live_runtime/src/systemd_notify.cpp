/*
 * Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
 * SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
 * Developed at the German Cancer Research Center (DKFZ).
 */

#include "dkfz_live/systemd_notify.hpp"

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

namespace dkfz::live
{

SystemdNotifier::SystemdNotifier()
{
    const auto *notify_socket = std::getenv("NOTIFY_SOCKET");
    if (notify_socket == nullptr || *notify_socket == '\0')
    {
        return;
    }
    socket_path_ = notify_socket;
    if (socket_path_.size() >= sizeof(sockaddr_un::sun_path))
    {
        socket_path_.clear();
        return;
    }
    socket_fd_ = ::socket(AF_UNIX, SOCK_DGRAM | SOCK_CLOEXEC, 0);
    if (socket_fd_ < 0)
    {
        socket_path_.clear();
        return;
    }

    const auto *watchdog_usec = std::getenv("WATCHDOG_USEC");
    const auto *watchdog_pid = std::getenv("WATCHDOG_PID");
    bool pid_matches = true;
    if (watchdog_pid != nullptr && *watchdog_pid != '\0')
    {
        try
        {
            pid_matches = std::stoll(watchdog_pid) ==
                          static_cast<long long>(::getpid());
        }
        catch (...)
        {
            pid_matches = false;
        }
    }
    if (watchdog_usec != nullptr && *watchdog_usec != '\0' && pid_matches)
    {
        try
        {
            const auto usec = std::stoull(watchdog_usec);
            if (usec > 0 &&
                usec <= static_cast<unsigned long long>(
                            std::numeric_limits<long long>::max()))
            {
                watchdog_interval_ = std::chrono::microseconds(usec / 2);
            }
        }
        catch (...)
        {
        }
    }
}

SystemdNotifier::~SystemdNotifier()
{
    if (socket_fd_ >= 0)
    {
        ::close(socket_fd_);
    }
}

bool SystemdNotifier::send(const std::string &message) const noexcept
{
    if (!available())
    {
        return false;
    }
    sockaddr_un address{};
    address.sun_family = AF_UNIX;
    socklen_t address_size = 0;
    if (!socket_path_.empty() && socket_path_.front() == '@')
    {
        address.sun_path[0] = '\0';
        std::memcpy(
            address.sun_path + 1,
            socket_path_.data() + 1,
            socket_path_.size() - 1);
        address_size = static_cast<socklen_t>(
            offsetof(sockaddr_un, sun_path) + socket_path_.size());
    }
    else
    {
        std::memcpy(
            address.sun_path,
            socket_path_.data(),
            socket_path_.size());
        address.sun_path[socket_path_.size()] = '\0';
        address_size = static_cast<socklen_t>(
            offsetof(sockaddr_un, sun_path) + socket_path_.size() + 1);
    }
    const auto sent = ::sendto(
        socket_fd_,
        message.data(),
        message.size(),
        MSG_NOSIGNAL,
        reinterpret_cast<const sockaddr *>(&address),
        address_size);
    return sent == static_cast<ssize_t>(message.size());
}

void SystemdNotifier::ready(const std::string &status_text) const noexcept
{
    static_cast<void>(send("READY=1\nSTATUS=" + status_text));
}

void SystemdNotifier::status(const std::string &status_text) const noexcept
{
    static_cast<void>(send("STATUS=" + status_text));
}

void SystemdNotifier::watchdog() const noexcept
{
    static_cast<void>(send("WATCHDOG=1"));
}

void SystemdNotifier::stopping(const std::string &status_text) const noexcept
{
    static_cast<void>(send("STOPPING=1\nSTATUS=" + status_text));
}

std::optional<std::chrono::microseconds>
SystemdNotifier::watchdog_interval() const noexcept
{
    return watchdog_interval_;
}

} // namespace dkfz::live
