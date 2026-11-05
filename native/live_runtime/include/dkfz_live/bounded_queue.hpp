/*
 * Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
 * SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
 * Developed at the German Cancer Research Center (DKFZ).
 */

#pragma once

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace dkfz::live
{

enum class QueuePopResult
{
    item,
    timeout,
    closed,
};

template <typename T>
class BoundedQueue
{
  public:
    explicit BoundedQueue(std::size_t capacity) : capacity_(capacity)
    {
        if (capacity_ == 0)
        {
            throw std::invalid_argument("BoundedQueue capacity must be positive");
        }
    }

    bool try_push(T value)
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (closed_ || values_.size() >= capacity_)
            {
                return false;
            }
            values_.push_back(std::move(value));
            if (values_.size() > high_water_mark_)
            {
                high_water_mark_ = values_.size();
            }
        }
        available_.notify_one();
        return true;
    }

    template <typename Rep, typename Period>
    bool wait_push(
        T value,
        const std::chrono::duration<Rep, Period> &timeout)
    {
        {
            std::unique_lock<std::mutex> lock(mutex_);
            if (!space_available_.wait_for(
                    lock,
                    timeout,
                    [&]
                    {
                        return closed_ || values_.size() < capacity_;
                    }))
            {
                return false;
            }
            if (closed_)
            {
                return false;
            }
            values_.push_back(std::move(value));
            if (values_.size() > high_water_mark_)
            {
                high_water_mark_ = values_.size();
            }
        }
        available_.notify_one();
        return true;
    }

    template <typename Rep, typename Period>
    QueuePopResult wait_pop(
        T &destination,
        const std::chrono::duration<Rep, Period> &timeout)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!available_.wait_for(
                lock,
                timeout,
                [&]
                {
                    return closed_ || !values_.empty();
                }))
        {
            return QueuePopResult::timeout;
        }
        if (values_.empty())
        {
            return QueuePopResult::closed;
        }
        destination = std::move(values_.front());
        values_.pop_front();
        lock.unlock();
        space_available_.notify_one();
        return QueuePopResult::item;
    }

    void close()
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            closed_ = true;
        }
        available_.notify_all();
        space_available_.notify_all();
    }

    std::size_t clear()
    {
        std::size_t removed = 0;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            removed = values_.size();
            values_.clear();
        }
        space_available_.notify_all();
        return removed;
    }

    std::size_t size() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return values_.size();
    }

    std::size_t capacity() const noexcept { return capacity_; }

    std::size_t high_water_mark() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return high_water_mark_;
    }

    bool closed() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return closed_;
    }

  private:
    const std::size_t capacity_;
    mutable std::mutex mutex_;
    std::condition_variable available_;
    std::condition_variable space_available_;
    std::deque<T> values_;
    std::size_t high_water_mark_ = 0;
    bool closed_ = false;
};

} // namespace dkfz::live
