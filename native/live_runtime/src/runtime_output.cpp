/*
 * Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
 * SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
 * Developed at the German Cancer Research Center (DKFZ).
 */

#include "dkfz_live/runtime_output.hpp"

#include <linux/gpio.h>

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <iomanip>
#include <limits>
#include <ostream>
#include <stdexcept>
#include <sys/ioctl.h>
#include <time.h>
#include <unistd.h>
#include <utility>

namespace dkfz::live
{
namespace
{

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

void emit_trigger_json(std::ostream &output, const TriggerEvent &event)
{
    output
        << "{\"schema\":\"dkfz-live-trigger-v1\""
        << ",\"sid\":\"" << json_escape(event.sid) << '"'
        << ",\"peak_frame_idx_50hz\":" << event.peak_frame_idx
        << ",\"peak_sample_idx_400hz\":" << event.peak_sample_idx
        << ",\"peak_time_ms\":";
    if (event.peak_time_ms.has_value())
    {
        output << std::setprecision(17) << *event.peak_time_ms;
    }
    else
    {
        output << "null";
    }
    output
        << ",\"emitted_on_frame_idx_50hz\":" << event.emitted_on_frame_idx
        << ",\"emitted_on_sample_idx_400hz\":" << event.emitted_on_sample_idx
        << ",\"emitted_on_time_ms\":";
    if (event.emitted_on_time_ms.has_value())
    {
        output << std::setprecision(17) << *event.emitted_on_time_ms;
    }
    else
    {
        output << "null";
    }
    output
        << ",\"probability\":" << std::setprecision(9) << event.probability
        << "}\n";
}

void emit_probability_json(
    std::ostream &output,
    const ProbabilityOutputRecord &record)
{
    output
        << "{\"schema\":\"dkfz-live-probability-v1\""
        << ",\"sid\":\"" << json_escape(record.sid) << '"'
        << ",\"frame_idx_50hz\":" << record.frame_idx
        << ",\"start_sample_idx_400hz\":" << record.start_sample_idx
        << ",\"end_sample_idx_400hz\":" << record.end_sample_idx
        << ",\"time_ms\":";
    if (record.time_ms.has_value())
    {
        output << std::setprecision(17) << *record.time_ms;
    }
    else
    {
        output << "null";
    }
    output
        << ",\"model_output\":" << std::setprecision(9) << record.model_output
        << ",\"probability\":" << record.probability
        << ",\"trigger_emitted_now\":"
        << (record.trigger_emitted_now ? "true" : "false")
        << "}\n";
}

class FileDescriptor
{
  public:
    explicit FileDescriptor(int value = -1) : value_(value) {}
    ~FileDescriptor()
    {
        if (value_ >= 0)
        {
            ::close(value_);
        }
    }
    FileDescriptor(const FileDescriptor &) = delete;
    FileDescriptor &operator=(const FileDescriptor &) = delete;
    FileDescriptor(FileDescriptor &&other) noexcept : value_(other.release()) {}
    FileDescriptor &operator=(FileDescriptor &&other) noexcept
    {
        if (this != &other)
        {
            if (value_ >= 0)
            {
                ::close(value_);
            }
            value_ = other.release();
        }
        return *this;
    }
    int get() const noexcept { return value_; }
    int release() noexcept
    {
        const auto value = value_;
        value_ = -1;
        return value;
    }

  private:
    int value_;
};

FileDescriptor open_gpio_chip(const std::string &chip)
{
    if (!chip.empty() && chip.front() == '/')
    {
        const auto fd = ::open(chip.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd < 0)
        {
            throw std::runtime_error(
                "Unable to open GPIO chip " + chip + ": " +
                std::strerror(errno));
        }
        return FileDescriptor(fd);
    }

    for (int index = 0; index < 256; ++index)
    {
        const auto path = "/dev/gpiochip" + std::to_string(index);
        FileDescriptor fd(::open(path.c_str(), O_RDONLY | O_CLOEXEC));
        if (fd.get() < 0)
        {
            continue;
        }
        gpiochip_info info{};
        if (::ioctl(fd.get(), GPIO_GET_CHIPINFO_IOCTL, &info) == 0 &&
            chip == info.label)
        {
            return fd;
        }
    }
    throw std::runtime_error(
        "GPIO chip '" + chip + "' was not found or is not accessible");
}

class LinuxGpioTriggerLine final : public TriggerLine
{
  public:
    explicit LinuxGpioTriggerLine(const GpioTriggerOutputConfig &config)
        : active_high_(config.active_high)
    {
        auto chip = open_gpio_chip(config.chip);
        gpiochip_info info{};
        if (::ioctl(chip.get(), GPIO_GET_CHIPINFO_IOCTL, &info) != 0)
        {
            throw std::runtime_error(
                "GPIO_GET_CHIPINFO_IOCTL failed: " +
                std::string(std::strerror(errno)));
        }
        const auto offset = *config.line_offset;
        if (offset >= info.lines)
        {
            throw std::runtime_error(
                "GPIO line offset " + std::to_string(offset) +
                " is outside chip '" + info.label + "' with " +
                std::to_string(info.lines) + " lines");
        }

        gpio_v2_line_request request{};
        request.offsets[0] = offset;
        request.num_lines = 1;
        request.config.flags = GPIO_V2_LINE_FLAG_OUTPUT;
        request.config.num_attrs = 1;
        request.config.attrs[0].attr.id =
            GPIO_V2_LINE_ATTR_ID_OUTPUT_VALUES;
        request.config.attrs[0].attr.values = inactive_level() ? 1ULL : 0ULL;
        request.config.attrs[0].mask = 1ULL;
        std::strncpy(
            request.consumer,
            config.consumer.c_str(),
            sizeof(request.consumer) - 1);
        if (::ioctl(chip.get(), GPIO_V2_GET_LINE_IOCTL, &request) != 0)
        {
            throw std::runtime_error(
                "GPIO_V2_GET_LINE_IOCTL failed for offset " +
                std::to_string(offset) + ": " + std::strerror(errno));
        }
        line_fd_ = FileDescriptor(request.fd);
        set_active(false);
    }

    void set_active(bool active) override
    {
        const auto physical_high = active ? active_high_ : !active_high_;
        gpio_v2_line_values values{};
        values.mask = 1ULL;
        values.bits = physical_high ? 1ULL : 0ULL;
        if (::ioctl(
                line_fd_.get(),
                GPIO_V2_LINE_SET_VALUES_IOCTL,
                &values) != 0)
        {
            throw std::runtime_error(
                "GPIO_V2_LINE_SET_VALUES_IOCTL failed: " +
                std::string(std::strerror(errno)));
        }
    }

  private:
    bool inactive_level() const noexcept { return !active_high_; }

    bool active_high_;
    FileDescriptor line_fd_;
};

std::uint64_t monotonic_now_ns()
{
    timespec now{};
    if (::clock_gettime(CLOCK_MONOTONIC, &now) != 0)
    {
        throw std::runtime_error(
            "clock_gettime failed while timing GPIO output: " +
            std::string(std::strerror(errno)));
    }
    return static_cast<std::uint64_t>(now.tv_sec) * 1'000'000'000ULL +
           static_cast<std::uint64_t>(now.tv_nsec);
}

void sleep_until_monotonic(std::uint64_t deadline_ns)
{
    timespec deadline{};
    deadline.tv_sec = static_cast<time_t>(deadline_ns / 1'000'000'000ULL);
    deadline.tv_nsec = static_cast<long>(deadline_ns % 1'000'000'000ULL);
    int result = 0;
    do
    {
        result = ::clock_nanosleep(
            CLOCK_MONOTONIC,
            TIMER_ABSTIME,
            &deadline,
            nullptr);
    } while (result == EINTR);
    if (result != 0)
    {
        throw std::runtime_error(
            "clock_nanosleep failed while timing GPIO output: " +
            std::string(std::strerror(result)));
    }
}

void sleep_for_monotonic(std::uint64_t duration_ns)
{
    const auto now_ns = monotonic_now_ns();
    if (duration_ns > std::numeric_limits<std::uint64_t>::max() - now_ns)
    {
        throw std::runtime_error("GPIO timing deadline overflow");
    }
    sleep_until_monotonic(now_ns + duration_ns);
}

class ConfiguredRuntimeOutputSink final : public RuntimeOutputSink
{
  public:
    ConfiguredRuntimeOutputSink(
        RuntimeOutputConfig config,
        std::ostream &jsonl_output,
        std::unique_ptr<TriggerLine> trigger_line)
        : config_(std::move(config)),
          jsonl_output_(jsonl_output),
          trigger_line_(std::move(trigger_line))
    {
        if (runtime_output_has_gpio(config_.mode))
        {
            if (!trigger_line_)
            {
                throw std::invalid_argument(
                    "GPIO output mode requires a trigger line");
            }
            set_inactive_and_start_idle();
        }
    }

    ~ConfiguredRuntimeOutputSink() override
    {
        try
        {
            close();
        }
        catch (...)
        {
        }
    }

    void write(const RuntimeOutputRecord &record) override
    {
        if (closed_)
        {
            throw std::runtime_error("Runtime output sink is closed");
        }
        if (record.kind == RuntimeOutputRecordKind::trigger &&
            runtime_output_has_gpio(config_.mode))
        {
            wait_for_minimum_idle();
            trigger_line_->set_active(true);
            try
            {
                sleep_for_monotonic(config_.gpio.pulse_width_ns);
            }
            catch (...)
            {
                try
                {
                    set_inactive_and_start_idle();
                }
                catch (...)
                {
                }
                throw;
            }
            set_inactive_and_start_idle();
        }

        if (runtime_output_has_jsonl(config_.mode))
        {
            if (record.kind == RuntimeOutputRecordKind::trigger)
            {
                emit_trigger_json(jsonl_output_, record.trigger);
            }
            else
            {
                emit_probability_json(jsonl_output_, record.probability);
            }
            jsonl_output_.flush();
            if (!jsonl_output_)
            {
                throw std::runtime_error("Writing JSONL runtime output failed");
            }
        }
    }

    void close() override
    {
        if (closed_)
        {
            return;
        }
        if (trigger_line_)
        {
            set_inactive_and_start_idle();
        }
        closed_ = true;
    }

  private:
    void set_inactive_and_start_idle()
    {
        trigger_line_->set_active(false);
        inactive_since_ns_ = monotonic_now_ns();
    }

    void wait_for_minimum_idle()
    {
        if (!inactive_since_ns_.has_value())
        {
            return;
        }
        if (config_.gpio.idle_width_ns >
            std::numeric_limits<std::uint64_t>::max() - *inactive_since_ns_)
        {
            throw std::runtime_error("GPIO idle timing deadline overflow");
        }
        sleep_until_monotonic(
            *inactive_since_ns_ + config_.gpio.idle_width_ns);
    }

    RuntimeOutputConfig config_;
    std::ostream &jsonl_output_;
    std::unique_ptr<TriggerLine> trigger_line_;
    std::optional<std::uint64_t> inactive_since_ns_;
    bool closed_ = false;
};

} // namespace

RuntimeOutputMode parse_runtime_output_mode(const std::string &value)
{
    if (value == "none")
    {
        return RuntimeOutputMode::none;
    }
    if (value == "jsonl")
    {
        return RuntimeOutputMode::jsonl;
    }
    if (value == "gpio")
    {
        return RuntimeOutputMode::gpio;
    }
    if (value == "gpio+jsonl")
    {
        return RuntimeOutputMode::gpio_jsonl;
    }
    throw std::invalid_argument(
        "Invalid trigger output mode '" + value +
        "'; expected none, jsonl, gpio, or gpio+jsonl");
}

std::string runtime_output_mode_name(RuntimeOutputMode mode)
{
    switch (mode)
    {
    case RuntimeOutputMode::none:
        return "none";
    case RuntimeOutputMode::jsonl:
        return "jsonl";
    case RuntimeOutputMode::gpio:
        return "gpio";
    case RuntimeOutputMode::gpio_jsonl:
        return "gpio+jsonl";
    }
    throw std::invalid_argument("Unknown runtime output mode");
}

bool runtime_output_has_jsonl(RuntimeOutputMode mode) noexcept
{
    return mode == RuntimeOutputMode::jsonl ||
           mode == RuntimeOutputMode::gpio_jsonl;
}

bool runtime_output_has_gpio(RuntimeOutputMode mode) noexcept
{
    return mode == RuntimeOutputMode::gpio ||
           mode == RuntimeOutputMode::gpio_jsonl;
}

void validate_runtime_output_config(
    const RuntimeOutputConfig &config,
    bool replay)
{
    if (config.queue_capacity == 0)
    {
        throw std::invalid_argument(
            "trigger_output_queue_capacity must be positive");
    }
    if (!runtime_output_has_gpio(config.mode))
    {
        return;
    }
    if (config.gpio.chip.empty())
    {
        throw std::invalid_argument(
            "trigger_gpio_chip is required for GPIO output");
    }
    if (!config.gpio.line_offset.has_value())
    {
        throw std::invalid_argument(
            "trigger_gpio_line_offset is required for GPIO output");
    }
    if (config.gpio.pulse_width_ns == 0)
    {
        throw std::invalid_argument(
            "trigger_gpio_pulse_width_us must be positive");
    }
    if (config.gpio.idle_width_ns == 0)
    {
        throw std::invalid_argument(
            "trigger_gpio_idle_width_us must be positive");
    }
    if (config.gpio.consumer.empty())
    {
        throw std::invalid_argument(
            "trigger_gpio_consumer must not be empty");
    }
    if (replay && !config.allow_gpio_during_replay)
    {
        throw std::invalid_argument(
            "GPIO trigger output is disabled during replay; select jsonl/none "
            "or explicitly allow GPIO during replay");
    }
}

RuntimeOutputRecord RuntimeOutputRecord::from_probability(
    ProbabilityOutputRecord value)
{
    RuntimeOutputRecord record;
    record.kind = RuntimeOutputRecordKind::probability;
    record.probability = std::move(value);
    return record;
}

RuntimeOutputRecord RuntimeOutputRecord::from_trigger(TriggerEvent value)
{
    RuntimeOutputRecord record;
    record.kind = RuntimeOutputRecordKind::trigger;
    record.trigger = std::move(value);
    return record;
}

std::unique_ptr<TriggerLine> make_linux_gpio_trigger_line(
    const GpioTriggerOutputConfig &config)
{
    return std::make_unique<LinuxGpioTriggerLine>(config);
}

std::unique_ptr<RuntimeOutputSink> make_runtime_output_sink(
    RuntimeOutputConfig config,
    std::ostream &jsonl_output,
    std::unique_ptr<TriggerLine> trigger_line)
{
    validate_runtime_output_config(config, false);
    if (runtime_output_has_gpio(config.mode) && !trigger_line)
    {
        trigger_line = make_linux_gpio_trigger_line(config.gpio);
    }
    return std::make_unique<ConfiguredRuntimeOutputSink>(
        std::move(config), jsonl_output, std::move(trigger_line));
}

} // namespace dkfz::live
