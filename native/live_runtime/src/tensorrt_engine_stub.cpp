/*
 * Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
 * SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
 * Developed at the German Cancer Research Center (DKFZ).
 */

#include "dkfz_live/inference_engine.hpp"

#include <stdexcept>

namespace dkfz::live
{

std::unique_ptr<LogitEngine> make_tensorrt_logit_engine(
    const std::filesystem::path &,
    const std::string &,
    const std::string &,
    std::size_t,
    std::size_t)
{
    throw std::runtime_error(
        "This executable was built without the TensorRT C++ and CUDA runtime "
        "libraries. Reconfigure CMake on the Jetson with "
        "-DDKFZ_ENABLE_TENSORRT=ON.");
}

bool tensorrt_backend_available() noexcept
{
    return false;
}

} // namespace dkfz::live
