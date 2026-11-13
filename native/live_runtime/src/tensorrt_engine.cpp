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
#include <cstring>
#include <cstdlib>
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
        std::size_t input_dim, bool profile)
        : input_tensor_(std::move(input_tensor)),
          output_tensor_(std::move(output_tensor)),
          window_size_(window_size),
          input_dim_(input_dim), profile_(profile)
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
            check_cuda(cudaMalloc(&output_device_, output_elements_ * sizeof(float)), "cudaMalloc output");
            check_cuda(cudaMallocHost(&input_host_, input_bytes_), "cudaMallocHost input");
            check_cuda(cudaMallocHost(&output_host_, output_elements_ * sizeof(float)), "cudaMallocHost output");
            int least_priority = 0, greatest_priority = 0;
            check_cuda(cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority), "stream priority range");
            check_cuda(cudaStreamCreateWithPriority(&stream_, cudaStreamNonBlocking, greatest_priority), "cudaStreamCreateWithPriority");
            if (profile_)
                for (auto &event : events_) check_cuda(cudaEventCreate(&event), "cudaEventCreate");
            if (!context_->setTensorAddress(input_tensor_.c_str(), input_device_) ||
                !context_->setTensorAddress(output_tensor_.c_str(), output_device_))
            {
                throw std::runtime_error("TensorRT rejected tensor device addresses");
            }
            // Warm up the fixed shape before acquisition. This does not advance
            // feature or recurrent state: each window is an independent call.
            std::memset(input_host_, 0, input_bytes_);
            for (int i = 0; i < 3; ++i) { enqueue(); check_cuda(cudaStreamSynchronize(stream_), "warmup"); }
            // Some TensorRT recurrent/loop implementations cannot be captured.
            // Failure to capture is an optimization miss; retain normal enqueue.
            const auto *disable_graph = std::getenv("DKFZ_DISABLE_CUDA_GRAPH");
            const bool allow_graph = disable_graph == nullptr || std::string(disable_graph) != "1";
            const auto begin = allow_graph
                ? cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal)
                : cudaErrorStreamCaptureUnsupported;
            if (begin == cudaSuccess)
            {
                bool captured = false;
                try { enqueue(true); captured = true; } catch (const std::exception &) {}
                const auto end = cudaStreamEndCapture(stream_, &graph_);
                if (captured && end == cudaSuccess && graph_ != nullptr)
                {
                    if (cudaGraphInstantiate(&graph_exec_, graph_, 0) != cudaSuccess)
                        graph_exec_ = nullptr;
                }
                static_cast<void>(cudaGetLastError());
            }
            else static_cast<void>(cudaGetLastError());
            DiagnosticLogger diagnostics(std::cerr);
            diagnostics.emit(DiagnosticSeverity::info, "tensorrt_execution", "tensorrt",
                "Initialized pinned-buffer TensorRT execution", {
                    {"cuda_graph", graph_exec_ != nullptr}, {"gpu_stage_timing", profile_},
                    {"stream_priority", static_cast<std::int64_t>(greatest_priority)}});
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
        return infer_logits(window).front();
    }

    std::vector<float> infer_logits(const std::vector<float> &window) override
    {
        if (window.size() != window_size_ * input_dim_)
        {
            throw std::invalid_argument(
                "TensorRT input has " + std::to_string(window.size()) +
                " values; expected " +
                std::to_string(window_size_ * input_dim_));
        }
        std::memcpy(input_host_, window.data(), input_bytes_);
        if (graph_exec_) check_cuda(cudaGraphLaunch(graph_exec_, stream_), "cudaGraphLaunch");
        else enqueue();
        check_cuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
        timing_.cuda_graph = graph_exec_ != nullptr;
        if (profile_)
        {
            float elapsed = 0;
            auto duration = [&](int a, int b) {
                check_cuda(cudaEventElapsedTime(&elapsed, events_[a], events_[b]), "cudaEventElapsedTime");
                return static_cast<std::uint64_t>(elapsed * 1e6);
            };
            timing_.upload_ns = duration(0, 1);
            timing_.compute_ns = duration(1, 2);
            timing_.download_ns = duration(2, 3);
        }
        const auto *output = static_cast<const float *>(output_host_);
        return std::vector<float>(output, output + output_elements_);
    }

    EngineTiming last_timing() const noexcept override { return timing_; }

  private:
    void enqueue(bool capture = false)
    {
        if (profile_) check_cuda(cudaEventRecordWithFlags(events_[0], stream_, capture ? cudaEventRecordExternal : cudaEventRecordDefault), "upload start");
        check_cuda(cudaMemcpyAsync(input_device_, input_host_, input_bytes_, cudaMemcpyHostToDevice, stream_), "upload");
        if (profile_) check_cuda(cudaEventRecordWithFlags(events_[1], stream_, capture ? cudaEventRecordExternal : cudaEventRecordDefault), "upload end");
        if (!context_->enqueueV3(stream_)) throw std::runtime_error("TensorRT enqueueV3 returned false");
        if (profile_) check_cuda(cudaEventRecordWithFlags(events_[2], stream_, capture ? cudaEventRecordExternal : cudaEventRecordDefault), "compute end");
        check_cuda(cudaMemcpyAsync(output_host_, output_device_, output_elements_ * sizeof(float), cudaMemcpyDeviceToHost, stream_), "download");
        if (profile_) check_cuda(cudaEventRecordWithFlags(events_[3], stream_, capture ? cudaEventRecordExternal : cudaEventRecordDefault), "download end");
    }

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
        if (output_elements != 1 && output_elements != 2)
        {
            throw std::invalid_argument(
                "TensorRT output must contain one or two logits, got shape " +
                dims_string(output_shape));
        }
        output_elements_ = static_cast<std::size_t>(output_elements);
    }

    void release_cuda() noexcept
    {
        if (stream_) cudaStreamSynchronize(stream_);
        if (graph_exec_) { cudaGraphExecDestroy(graph_exec_); graph_exec_ = nullptr; }
        if (graph_) { cudaGraphDestroy(graph_); graph_ = nullptr; }
        for (auto &event : events_) if (event) { cudaEventDestroy(event); event = nullptr; }
        if (input_host_) { cudaFreeHost(input_host_); input_host_ = nullptr; }
        if (output_host_) { cudaFreeHost(output_host_); output_host_ = nullptr; }
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
    std::size_t output_elements_ = 1;
    void *input_device_ = nullptr;
    void *output_device_ = nullptr;
    void *input_host_ = nullptr, *output_host_ = nullptr;
    cudaStream_t stream_ = nullptr;
    cudaGraph_t graph_ = nullptr;
    cudaGraphExec_t graph_exec_ = nullptr;
    cudaEvent_t events_[4] = {};
    bool profile_ = false;
    EngineTiming timing_;
};

} // namespace

std::unique_ptr<LogitEngine> make_tensorrt_logit_engine(
    const std::filesystem::path &engine_path,
    const std::string &input_tensor,
    const std::string &output_tensor,
    std::size_t window_size,
    std::size_t input_dim, bool profile)
{
    return std::make_unique<TensorRTLogitEngine>(
        engine_path,
        input_tensor,
        output_tensor,
        window_size,
        input_dim, profile);
}

bool tensorrt_backend_available() noexcept
{
    return true;
}

} // namespace dkfz::live
