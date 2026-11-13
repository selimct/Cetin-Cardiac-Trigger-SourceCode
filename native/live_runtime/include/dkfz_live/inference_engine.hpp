/*
 * Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
 * SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
 * Developed at the German Cancer Research Center (DKFZ).
 */

#pragma once

#include <filesystem>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace dkfz::live
{

struct EngineTiming
{
    std::uint64_t upload_ns = 0, compute_ns = 0, download_ns = 0;
    bool cuda_graph = false;
};

class LogitEngine
{
  public:
    virtual ~LogitEngine() = default;
    virtual float infer_logit(const std::vector<float> &window) = 0;
    virtual std::vector<float> infer_logits(const std::vector<float> &window)
    { return {infer_logit(window)}; }
    virtual EngineTiming last_timing() const noexcept { return {}; }
};

std::unique_ptr<LogitEngine> make_tensorrt_logit_engine(
    const std::filesystem::path &engine_path,
    const std::string &input_tensor,
    const std::string &output_tensor,
    std::size_t window_size,
    std::size_t input_dim,
    bool profile = false);

bool tensorrt_backend_available() noexcept;

} // namespace dkfz::live
