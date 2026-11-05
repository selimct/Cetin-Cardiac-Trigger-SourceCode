/*
 * Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
 * SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
 * Developed at the German Cancer Research Center (DKFZ).
 */

#include "dkfz_live/inference_engine.hpp"
#include "dkfz_live/diagnostics.hpp"

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace dkfz::live
{
namespace
{

class TensorRTLogger final : public nvinfer1::ILogger
{
  public:
    void log(Severity severity, const char *message) noexcept override
    {
        if (severity <= Severity::kWARNING)
        {
            try
            {
                diagnostics_.emit(
                    severity <= Severity::kERROR
                        ? DiagnosticSeverity::error
                        : DiagnosticSeverity::warning,
                    "tensorrt_log",
                    "tensorrt",
                    message == nullptr ? "TensorRT emitted an empty message"
                                       : message);
            }
            catch (...)
            {
                // ILogger is noexcept. Runtime error handling must not be
                // replaced by termination if diagnostic output itself fails.
            }
        }
    }

  private:
    DiagnosticLogger diagnostics_{std::cerr};
};

void check_cuda(cudaError_t result, const char *action)
{
    if (result != cudaSuccess)
    {
        throw std::runtime_error(
            std::string(action) + " failed: " + cudaGetErrorString(result));
    }
}

std::vector<char> read_engine(const std::filesystem::path &path)
{
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input)
    {
        throw std::runtime_error("Unable to open TensorRT engine: " + path.string());
    }
    const auto end = input.tellg();
    if (end <= 0)
    {
        throw std::runtime_error("TensorRT engine is empty: " + path.string());
    }
    std::vector<char> bytes(static_cast<std::size_t>(end));
    input.seekg(0, std::ios::beg);
    if (!input.read(bytes.data(), static_cast<std::streamsize>(bytes.size())))
    {
        throw std::runtime_error("Unable to read TensorRT engine: " + path.string());
    }
    return bytes;
}

std::string dims_string(const nvinfer1::Dims &dims)
{
    std::ostringstream output;
    output << '[';
    for (int index = 0; index < dims.nbDims; ++index)
    {
        if (index > 0)
        {
            output << ',';
        }
        output << dims.d[index];
    }
    output << ']';
    return output.str();
}

class TensorRTLogitEngine final : public LogitEngine
{
  public:
    TensorRTLogitEngine(
        const std::filesystem::path &engine_path,
        std::string input_tensor,
        std::string output_tensor,
        std::size_t window_size,
        std::size_t input_dim)
        : input_tensor_(std::move(input_tensor)),
          output_tensor_(std::move(output_tensor)),
          window_size_(window_size),
          input_dim_(input_dim)
    {
        const auto bytes = read_engine(engine_path);
        runtime_.reset(nvinfer1::createInferRuntime(logger_));
        if (!runtime_)
        {
            throw std::runtime_error("Could not create TensorRT runtime");
        }
        engine_.reset(runtime_->deserializeCudaEngine(bytes.data(), bytes.size()));
        if (!engine_)
        {
            throw std::runtime_error(
                "Could not deserialize TensorRT engine: " + engine_path.string());
        }
        context_.reset(engine_->createExecutionContext());
        if (!context_)
        {
            throw std::runtime_error("Could not create TensorRT execution context");
        }
        validate_contract();

        input_bytes_ = window_size_ * input_dim_ * sizeof(float);
        try
        {
            check_cuda(cudaMalloc(&input_device_, input_bytes_), "cudaMalloc input");
            check_cuda(cudaMalloc(&output_device_, sizeof(float)), "cudaMalloc output");
            check_cuda(cudaStreamCreate(&stream_), "cudaStreamCreate");
            if (!context_->setTensorAddress(input_tensor_.c_str(), input_device_) ||
                !context_->setTensorAddress(output_tensor_.c_str(), output_device_))
            {
                throw std::runtime_error("TensorRT rejected tensor device addresses");
            }
        }
        catch (...)
        {
            release_cuda();
            throw;
        }
    }

    ~TensorRTLogitEngine() override
    {
        release_cuda();
    }

    TensorRTLogitEngine(const TensorRTLogitEngine &) = delete;
    TensorRTLogitEngine &operator=(const TensorRTLogitEngine &) = delete;

    float infer_logit(const std::vector<float> &window) override
    {
        if (window.size() != window_size_ * input_dim_)
        {
            throw std::invalid_argument(
                "TensorRT input has " + std::to_string(window.size()) +
                " values; expected " +
                std::to_string(window_size_ * input_dim_));
        }
        check_cuda(
            cudaMemcpyAsync(
                input_device_,
                window.data(),
                input_bytes_,
                cudaMemcpyHostToDevice,
                stream_),
            "cudaMemcpyAsync input");
        if (!context_->enqueueV3(stream_))
        {
            throw std::runtime_error("TensorRT enqueueV3 returned false");
        }
        float output = 0.0F;
        check_cuda(
            cudaMemcpyAsync(
                &output,
                output_device_,
                sizeof(output),
                cudaMemcpyDeviceToHost,
                stream_),
            "cudaMemcpyAsync output");
        check_cuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
        return output;
    }

  private:
    void validate_contract()
    {
        bool found_input = false;
        bool found_output = false;
        for (int index = 0; index < engine_->getNbIOTensors(); ++index)
        {
            const std::string name = engine_->getIOTensorName(index);
            found_input = found_input || name == input_tensor_;
            found_output = found_output || name == output_tensor_;
        }
        if (!found_input || !found_output)
        {
            throw std::invalid_argument(
                "TensorRT engine is missing configured input/output tensor names");
        }
        if (engine_->getTensorIOMode(input_tensor_.c_str()) !=
                nvinfer1::TensorIOMode::kINPUT ||
            engine_->getTensorIOMode(output_tensor_.c_str()) !=
                nvinfer1::TensorIOMode::kOUTPUT)
        {
            throw std::invalid_argument("TensorRT tensor direction is invalid");
        }
        if (engine_->getTensorDataType(input_tensor_.c_str()) !=
                nvinfer1::DataType::kFLOAT ||
            engine_->getTensorDataType(output_tensor_.c_str()) !=
                nvinfer1::DataType::kFLOAT)
        {
            throw std::invalid_argument(
                "Native live runtime requires float32 input and output tensors");
        }
        // TensorRT reports a vectorized dimension of -1 for scalar element
        // layouts; components-per-element is itself -1 when not vectorized.
        if (engine_->getTensorFormat(input_tensor_.c_str()) !=
                nvinfer1::TensorFormat::kLINEAR ||
            engine_->getTensorFormat(output_tensor_.c_str()) !=
                nvinfer1::TensorFormat::kLINEAR ||
            engine_->getTensorVectorizedDim(input_tensor_.c_str()) != -1 ||
            engine_->getTensorVectorizedDim(output_tensor_.c_str()) != -1)
        {
            throw std::invalid_argument(
                "Native live runtime requires scalar, linear TensorRT I/O formats");
        }
        if (engine_->getTensorLocation(input_tensor_.c_str()) !=
                nvinfer1::TensorLocation::kDEVICE ||
            engine_->getTensorLocation(output_tensor_.c_str()) !=
                nvinfer1::TensorLocation::kDEVICE)
        {
            throw std::invalid_argument(
                "Native live runtime requires device-resident TensorRT I/O");
        }

        const auto expected = nvinfer1::Dims3(
            1,
            static_cast<std::int32_t>(window_size_),
            static_cast<std::int32_t>(input_dim_));
        auto input_shape = engine_->getTensorShape(input_tensor_.c_str());
        if (input_shape.nbDims != 3)
        {
            throw std::invalid_argument(
                "TensorRT input shape " + dims_string(input_shape) +
                " is not rank 3");
        }
        for (int axis = 0; axis < expected.nbDims; ++axis)
        {
            if (input_shape.d[axis] >= 0 && input_shape.d[axis] != expected.d[axis])
            {
                throw std::invalid_argument(
                    "TensorRT input shape " + dims_string(input_shape) +
                    " does not match configured [1," +
                    std::to_string(window_size_) + "," +
                    std::to_string(input_dim_) + "]");
            }
        }
        if (!context_->setInputShape(input_tensor_.c_str(), expected))
        {
            throw std::invalid_argument("TensorRT rejected configured input shape");
        }
        const auto output_shape = context_->getTensorShape(output_tensor_.c_str());
        std::int64_t output_elements = 1;
        for (int axis = 0; axis < output_shape.nbDims; ++axis)
        {
            if (output_shape.d[axis] < 0)
            {
                throw std::invalid_argument(
                    "TensorRT output shape remains dynamic: " +
                    dims_string(output_shape));
            }
            output_elements *= output_shape.d[axis];
        }
        if (output_elements != 1)
        {
            throw std::invalid_argument(
                "TensorRT output must contain one logit, got shape " +
                dims_string(output_shape));
        }
    }

    void release_cuda() noexcept
    {
        if (stream_ != nullptr)
        {
            cudaStreamDestroy(stream_);
            stream_ = nullptr;
        }
        if (output_device_ != nullptr)
        {
            cudaFree(output_device_);
            output_device_ = nullptr;
        }
        if (input_device_ != nullptr)
        {
            cudaFree(input_device_);
            input_device_ = nullptr;
        }
    }

    TensorRTLogger logger_;
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    std::string input_tensor_;
    std::string output_tensor_;
    std::size_t window_size_;
    std::size_t input_dim_;
    std::size_t input_bytes_ = 0;
    void *input_device_ = nullptr;
    void *output_device_ = nullptr;
    cudaStream_t stream_ = nullptr;
};

} // namespace

std::unique_ptr<LogitEngine> make_tensorrt_logit_engine(
    const std::filesystem::path &engine_path,
    const std::string &input_tensor,
    const std::string &output_tensor,
    std::size_t window_size,
    std::size_t input_dim)
{
    return std::make_unique<TensorRTLogitEngine>(
        engine_path,
        input_tensor,
        output_tensor,
        window_size,
        input_dim);
}

bool tensorrt_backend_available() noexcept
{
    return true;
}

} // namespace dkfz::live
