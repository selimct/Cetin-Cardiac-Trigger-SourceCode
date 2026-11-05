/*
 * Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
 * SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
 * Developed at the German Cancer Research Center (DKFZ).
 */

#include "dkfz_uart/uart_stream.hpp"

#include <asm/ioctls.h>
#include <asm/termbits.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstring>
#include <cmath>
#include <fcntl.h>
#include <fstream>
#include <poll.h>
#include <stdexcept>
#include <string>
#include <sys/ioctl.h>
#include <time.h>
#include <unistd.h>

namespace dkfz::uart
{
namespace
{

constexpr std::uint8_t kPulsUpdateSign = 1;

std::uint16_t decode_ecg1(const std::uint8_t *bytes, std::size_t offset)
{
    return static_cast<std::uint16_t>(
        (static_cast<std::uint16_t>(bytes[offset] & 0x0FU) << 8U) |
        bytes[offset + 1]);
}

std::uint16_t decode_ecg2(const std::uint8_t *bytes, std::size_t offset)
{
    return static_cast<std::uint16_t>(
        (static_cast<std::uint16_t>(bytes[offset + 2] & 0x0FU) << 8U) |
        (static_cast<std::uint16_t>(bytes[offset + 3] & 0x0FU) << 4U) |
        (bytes[offset + 4] & 0x0FU));
}

std::uint8_t decode_ext(const std::uint8_t *bytes, std::size_t sample_in_group)
{
    constexpr std::array<std::size_t, 2> offsets{0, 14};
    const auto frame_offset = offsets[sample_in_group / 2];
    return static_cast<std::uint8_t>(bytes[frame_offset + 5] & 0x0FU);
}

void configure_uart(int fd, std::uint32_t baud_rate)
{
    termios2 settings{};
    if (::ioctl(fd, TCGETS2, &settings) != 0)
    {
        throw std::runtime_error(
            "TCGETS2 failed: " + std::string(std::strerror(errno)));
    }

    settings.c_iflag = INPCK | IGNPAR;
    settings.c_oflag = 0;
    settings.c_lflag = 0;
    settings.c_cflag &=
        static_cast<tcflag_t>(~(CBAUD | CSIZE | CSTOPB | PARODD | CRTSCTS));
    settings.c_cflag |= BOTHER | CS8 | CLOCAL | CREAD | PARENB;
    settings.c_ispeed = baud_rate;
    settings.c_ospeed = baud_rate;
    settings.c_cc[VMIN] = 1;
    settings.c_cc[VTIME] = 0;

    if (::ioctl(fd, TCSETS2, &settings) != 0)
    {
        throw std::runtime_error(
            "TCSETS2 failed: " + std::string(std::strerror(errno)));
    }
    if (::ioctl(fd, TCFLSH, TCIFLUSH) != 0)
    {
        throw std::runtime_error(
            "UART input flush failed: " + std::string(std::strerror(errno)));
    }
}

class OptionalByteCapture
{
  public:
    OptionalByteCapture(
        const std::filesystem::path &path,
        std::uint64_t maximum_bytes)
        : maximum_bytes_(maximum_bytes)
    {
        if (path.empty())
        {
            return;
        }
        if (path.has_parent_path())
        {
            std::filesystem::create_directories(path.parent_path());
        }
        fd_ = ::open(path.c_str(), O_CREAT | O_EXCL | O_WRONLY | O_CLOEXEC, 0640);
        if (fd_ < 0)
        {
            throw std::runtime_error(
                "Unable to create capture " + path.string() + ": " +
                std::strerror(errno));
        }
        path_ = path;
    }

    ~OptionalByteCapture()
    {
        if (fd_ >= 0)
        {
            ::close(fd_);
        }
    }

    std::size_t write(const std::uint8_t *data, std::size_t bytes)
    {
        if (fd_ < 0)
        {
            return bytes;
        }
        std::size_t accepted = bytes;
        if (maximum_bytes_ > 0)
        {
            const auto remaining = maximum_bytes_ - bytes_written_;
            accepted = static_cast<std::size_t>(
                std::min<std::uint64_t>(remaining, bytes));
        }

        std::size_t pending = accepted;
        while (pending > 0)
        {
            const auto written = ::write(fd_, data, pending);
            if (written < 0)
            {
                if (errno == EINTR)
                {
                    continue;
                }
                throw std::runtime_error(
                    "Capture write failed: " + std::string(std::strerror(errno)));
            }
            if (written == 0)
            {
                throw std::runtime_error("Capture write returned zero");
            }
            const auto count = static_cast<std::size_t>(written);
            data += count;
            pending -= count;
            bytes_written_ += count;
        }
        if (accepted < bytes ||
            (maximum_bytes_ > 0 && bytes_written_ >= maximum_bytes_))
        {
            limit_reached_ = true;
        }
        return accepted;
    }

    void finish()
    {
        if (fd_ < 0)
        {
            return;
        }
        if (::fdatasync(fd_) != 0)
        {
            const auto message = std::string(std::strerror(errno));
            ::close(fd_);
            fd_ = -1;
            throw std::runtime_error("fdatasync failed for capture: " + message);
        }
        ::close(fd_);
        fd_ = -1;
    }

    std::uint64_t bytes_written() const noexcept { return bytes_written_; }
    bool limit_reached() const noexcept { return limit_reached_; }

  private:
    int fd_ = -1;
    std::filesystem::path path_;
    std::uint64_t maximum_bytes_ = 0;
    std::uint64_t bytes_written_ = 0;
    bool limit_reached_ = false;
};

class FileDescriptor
{
  public:
    explicit FileDescriptor(int value) : value_(value) {}
    ~FileDescriptor()
    {
        if (value_ >= 0)
        {
            ::close(value_);
        }
    }
    FileDescriptor(const FileDescriptor &) = delete;
    FileDescriptor &operator=(const FileDescriptor &) = delete;
    int get() const noexcept { return value_; }

  private:
    int value_;
};

} // namespace

std::uint64_t monotonic_clock_ns()
{
    timespec value{};
    if (::clock_gettime(CLOCK_MONOTONIC, &value) != 0)
    {
        throw std::runtime_error(
            "clock_gettime(CLOCK_MONOTONIC) failed: " +
            std::string(std::strerror(errno)));
    }
    return static_cast<std::uint64_t>(value.tv_sec) * 1'000'000'000ULL +
           static_cast<std::uint64_t>(value.tv_nsec);
}

std::uint64_t realtime_clock_ns()
{
    timespec value{};
    if (::clock_gettime(CLOCK_REALTIME, &value) != 0)
    {
        throw std::runtime_error(
            "clock_gettime(CLOCK_REALTIME) failed: " +
            std::string(std::strerror(errno)));
    }
    return static_cast<std::uint64_t>(value.tv_sec) * 1'000'000'000ULL +
           static_cast<std::uint64_t>(value.tv_nsec);
}

CalibratedSignals calibrate_signals(
    const DecodedSample &sample,
    const UartSignalOffsets &offsets)
{
    if (!std::isfinite(offsets.ecg1) ||
        !std::isfinite(offsets.ecg2) ||
        !std::isfinite(offsets.puls))
    {
        throw std::invalid_argument("UART signal offsets must be finite");
    }
    CalibratedSignals result;
    result.ecg1 = static_cast<double>(sample.ecg1_raw) - offsets.ecg1;
    result.ecg2 = static_cast<double>(sample.ecg2_raw) - offsets.ecg2;
    result.ecg3 = result.ecg2 - result.ecg1;
    result.puls = static_cast<double>(sample.puls_raw) - offsets.puls;
    return result;
}

UartByteDecoder::UartByteDecoder(
    std::uint32_t sample_rate_hz,
    SampleCallback callback,
    std::optional<std::uint64_t> monotonic_start_ns,
    std::optional<std::uint64_t> realtime_start_ns)
    : sample_rate_hz_(sample_rate_hz),
      callback_(std::move(callback)),
      monotonic_start_ns_(monotonic_start_ns.value_or(monotonic_clock_ns())),
      realtime_start_ns_(realtime_start_ns.value_or(realtime_clock_ns()))
{
    if (sample_rate_hz_ == 0)
    {
        throw std::invalid_argument("sample_rate_hz must be positive");
    }
    if (!callback_)
    {
        throw std::invalid_argument("UartByteDecoder requires a sample callback");
    }
}

void UartByteDecoder::process(const std::uint8_t *data, std::size_t size)
{
    if (stopped_ || size == 0)
    {
        return;
    }
    if (data == nullptr)
    {
        throw std::invalid_argument("UART byte data may not be null");
    }
    buffer_.insert(buffer_.end(), data, data + size);
    parse_available();
}

void UartByteDecoder::process(const std::vector<std::uint8_t> &data)
{
    process(data.data(), data.size());
}

bool UartByteDecoder::has_group_anchors(std::size_t start) const
{
    constexpr std::array<std::size_t, 4> offsets{0, 7, 14, 21};
    return std::all_of(
        offsets.begin(),
        offsets.end(),
        [&](std::size_t frame)
        {
            return (buffer_[start + frame + 3] & 0xF0U) == 0xD0U &&
                   (buffer_[start + frame + 4] & 0xF0U) == 0xD0U;
        });
}

void UartByteDecoder::discard_prefix(std::size_t count)
{
    if (count == 0)
    {
        return;
    }
    stats_.bytes_discarded += count;
    consume_prefix(count);
}

void UartByteDecoder::consume_prefix(std::size_t count)
{
    buffer_.erase(
        buffer_.begin(),
        buffer_.begin() + static_cast<std::ptrdiff_t>(count));
}

void UartByteDecoder::parse_available()
{
    while (buffer_.size() >= kGroupBytes && !stopped_)
    {
        if (synchronized_ && has_group_anchors(0))
        {
            accept_front_group();
            continue;
        }
        if (synchronized_)
        {
            if (buffer_.size() < kGroupBytes + kInterPacketBytes)
            {
                return;
            }
            if (has_group_anchors(kInterPacketBytes))
            {
                consume_prefix(kInterPacketBytes);
                accept_front_group();
                continue;
            }
            synchronized_ = false;
            resync_discarded_ = 0;
        }

        std::optional<std::size_t> group_start;
        const auto last_start = buffer_.size() - kGroupBytes;
        for (std::size_t start = 0; start <= last_start; ++start)
        {
            if (has_group_anchors(start))
            {
                group_start = start;
                break;
            }
        }
        if (!group_start.has_value())
        {
            const auto count = buffer_.size() - (kGroupBytes - 1);
            discard_prefix(count);
            if (ever_synchronized_)
            {
                resync_discarded_ += count;
            }
            return;
        }

        const auto prefix = *group_start;
        discard_prefix(prefix);
        if (ever_synchronized_)
        {
            resync_discarded_ += prefix;
            stats_.malformed_groups +=
                (resync_discarded_ + (kGroupBytes / 2)) / kGroupBytes;
        }
        resync_discarded_ = 0;
        synchronized_ = true;
        ever_synchronized_ = true;
        accept_front_group();
    }
}

void UartByteDecoder::accept_front_group()
{
    accept_group(buffer_.data());
    buffer_.erase(
        buffer_.begin(),
        buffer_.begin() + static_cast<std::ptrdiff_t>(kGroupBytes));
}

void UartByteDecoder::accept_group(const std::uint8_t *bytes)
{
    const auto pr_sign = static_cast<std::uint8_t>((bytes[12] >> 4U) & 0x0FU);
    const auto pulse_value = static_cast<std::uint16_t>(
        (static_cast<std::uint16_t>(bytes[12] & 0x0FU) << 8U) | bytes[13]);
    if (pr_sign == kPulsUpdateSign)
    {
        pending_pulses_.push_back(pulse_value);
        stats_.pulse_queue_high_water = std::max<std::uint64_t>(
            stats_.pulse_queue_high_water,
            pending_pulses_.size());
    }

    constexpr std::array<std::size_t, kSamplesPerGroup> offsets{0, 7, 14, 21};
    const auto group_index = stats_.groups++;
    for (std::size_t sample = 0; sample < offsets.size() && !stopped_; ++sample)
    {
        emit_sample(
            bytes,
            offsets[sample],
            group_index,
            sample,
            pr_sign,
            pulse_value);
    }
}

bool UartByteDecoder::select_pulse(std::uint16_t fallback)
{
    if (pulse_samples_remaining_ > 0)
    {
        return false;
    }
    if (!pending_pulses_.empty())
    {
        current_pulse_ = pending_pulses_.front();
        pending_pulses_.pop_front();
        pulse_samples_remaining_ = 8;
        return true;
    }
    if (!current_pulse_.has_value())
    {
        current_pulse_ = fallback;
    }
    return false;
}

void UartByteDecoder::emit_sample(
    const std::uint8_t *bytes,
    std::size_t offset,
    std::uint64_t group_index,
    std::size_t sample_in_group,
    std::uint8_t pr_sign,
    std::uint16_t pulse_fallback)
{
    const bool pulse_updated = select_pulse(pulse_fallback);
    const auto sample_period_ns = 1'000'000'000ULL / sample_rate_hz_;
    const auto relative_ns = stats_.samples * sample_period_ns;
    const auto ecg1_raw = decode_ecg1(bytes, offset);
    const auto ecg2_raw = decode_ecg2(bytes, offset);
    const DecodedSample sample{
        stats_.samples,
        static_cast<double>(relative_ns) / 1'000'000.0,
        monotonic_start_ns_ + relative_ns,
        realtime_start_ns_ + relative_ns,
        ecg1_raw,
        ecg2_raw,
        *current_pulse_,
        decode_ext(bytes, sample_in_group),
        pr_sign,
        pulse_updated,
        group_index,
        sample_in_group,
    };
    const auto keep_running = callback_(sample);
    ++stats_.samples;
    if (pulse_samples_remaining_ > 0)
    {
        --pulse_samples_remaining_;
    }
    stopped_ = !keep_running;
}

ByteSourceStats run_byte_source(
    const ByteSourceConfig &config,
    const ByteChunkCallback &callback,
    std::atomic<bool> &stop_requested,
    const SourceProgressCallback &progress_callback)
{
    if (config.baud_rate == 0)
    {
        throw std::invalid_argument("baud_rate must be positive");
    }
    if (!callback)
    {
        throw std::invalid_argument("Byte source requires a chunk callback");
    }
    if (!config.replay_path.empty() && !config.capture_path.empty())
    {
        throw std::invalid_argument(
            "capture_path cannot be combined with replay_path");
    }

    ByteSourceStats stats;
    std::array<std::uint8_t, 4096> bytes{};
    if (!config.replay_path.empty())
    {
        std::ifstream input(config.replay_path, std::ios::binary);
        if (!input)
        {
            throw std::runtime_error(
                "Unable to open replay file: " + config.replay_path.string());
        }
        while (!stop_requested.load(std::memory_order_relaxed) &&
               input.read(
                   reinterpret_cast<char *>(bytes.data()),
                   static_cast<std::streamsize>(bytes.size())))
        {
            stats.bytes_read += bytes.size();
            if (!callback(bytes.data(), bytes.size()))
            {
                return stats;
            }
            if (progress_callback)
            {
                progress_callback(stats);
            }
        }
        if (!stop_requested.load(std::memory_order_relaxed))
        {
            const auto remaining = input.gcount();
            if (remaining > 0)
            {
                const auto size = static_cast<std::size_t>(remaining);
                stats.bytes_read += size;
                static_cast<void>(callback(bytes.data(), size));
                if (progress_callback)
                {
                    progress_callback(stats);
                }
            }
            if (!input.eof())
            {
                throw std::runtime_error(
                    "Unable to read replay file: " + config.replay_path.string());
            }
        }
        return stats;
    }

    const int raw_fd = ::open(
        config.device.c_str(), O_RDONLY | O_NOCTTY | O_CLOEXEC);
    if (raw_fd < 0)
    {
        throw std::runtime_error(
            "Unable to open UART " + config.device.string() + ": " +
            std::strerror(errno));
    }
    FileDescriptor fd(raw_fd);
    configure_uart(fd.get(), config.baud_rate);
    OptionalByteCapture capture(config.capture_path, config.max_capture_bytes);
    pollfd descriptor{fd.get(), POLLIN, 0};

    while (!stop_requested.load(std::memory_order_relaxed))
    {
        descriptor.revents = 0;
        const auto poll_result = ::poll(&descriptor, 1, 100);
        if (poll_result < 0)
        {
            if (errno == EINTR)
            {
                continue;
            }
            throw std::runtime_error(
                "UART poll failed: " + std::string(std::strerror(errno)));
        }
        if (poll_result > 0 &&
            (descriptor.revents & (POLLERR | POLLHUP | POLLNVAL)) != 0)
        {
            throw std::runtime_error("UART reported an error or hangup");
        }
        if (poll_result > 0 && (descriptor.revents & POLLIN) != 0)
        {
            const auto count = ::read(fd.get(), bytes.data(), bytes.size());
            if (count < 0)
            {
                if (errno == EINTR || errno == EAGAIN)
                {
                    continue;
                }
                throw std::runtime_error(
                    "UART read failed: " + std::string(std::strerror(errno)));
            }
            if (count > 0)
            {
                const auto size = static_cast<std::size_t>(count);
                stats.bytes_read += size;
                const auto accepted = capture.write(bytes.data(), size);
                stats.capture_bytes = capture.bytes_written();
                stats.capture_limit_reached = capture.limit_reached();
                if (!callback(bytes.data(), accepted) ||
                    stats.capture_limit_reached)
                {
                    break;
                }
            }
        }
        if (progress_callback)
        {
            progress_callback(stats);
        }
    }
    capture.finish();
    return stats;
}

} // namespace dkfz::uart
