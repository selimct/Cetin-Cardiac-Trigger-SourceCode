# Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Developed at the German Cancer Research Center (DKFZ).

cmake_minimum_required(VERSION 3.16)

project(dkfz_live_native LANGUAGES CXX)

# Development tests are maintained upstream.
set(BUILD_TESTING OFF CACHE BOOL "Tests are not included in this distribution" FORCE)
find_package(Threads REQUIRED)

set(
    DKFZ_ENABLE_TENSORRT
    "AUTO"
    CACHE STRING
    "Build the TensorRT C++ backend: AUTO, ON, or OFF"
)
set_property(
    CACHE DKFZ_ENABLE_TENSORRT
    PROPERTY STRINGS AUTO ON OFF
)

add_library(
    dkfz_live_core
    STATIC
        live_core/src/live_core.cpp
)
target_compile_features(dkfz_live_core PUBLIC cxx_std_17)
target_include_directories(
    dkfz_live_core
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/live_core/include"
)
target_compile_options(
    dkfz_live_core
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -Wconversion
        -Wshadow
)
set_target_properties(dkfz_live_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(
    dkfz_uart_core
    STATIC
        uart/src/uart_stream.cpp
)
target_compile_features(dkfz_uart_core PUBLIC cxx_std_17)
target_include_directories(
    dkfz_uart_core
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/uart/include"
)
target_compile_options(
    dkfz_uart_core
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -Wconversion
        -Wshadow
)

set(_dkfz_tensorrt_requested FALSE)
if(DKFZ_ENABLE_TENSORRT STREQUAL "ON" OR DKFZ_ENABLE_TENSORRT STREQUAL "AUTO")
    find_path(
        TENSORRT_INCLUDE_DIR
        NvInfer.h
        PATH_SUFFIXES aarch64-linux-gnu x86_64-linux-gnu
    )
    find_library(
        TENSORRT_LIBRARY
        NAMES nvinfer
        PATH_SUFFIXES aarch64-linux-gnu x86_64-linux-gnu
    )
    find_package(CUDAToolkit QUIET)
    if(TENSORRT_INCLUDE_DIR AND TENSORRT_LIBRARY AND CUDAToolkit_FOUND)
        set(_dkfz_tensorrt_requested TRUE)
    elseif(DKFZ_ENABLE_TENSORRT STREQUAL "ON")
        message(FATAL_ERROR
            "DKFZ_ENABLE_TENSORRT=ON, but NvInfer.h, libnvinfer, or the "
            "CUDA toolkit was not found"
        )
    endif()
elseif(NOT DKFZ_ENABLE_TENSORRT STREQUAL "OFF")
    message(FATAL_ERROR "DKFZ_ENABLE_TENSORRT must be AUTO, ON, or OFF")
endif()

set(_dkfz_inference_engine_source
    live_runtime/src/tensorrt_engine_stub.cpp
)
if(_dkfz_tensorrt_requested)
    set(_dkfz_inference_engine_source
        live_runtime/src/tensorrt_engine.cpp
    )
endif()

add_library(
    dkfz_live_runtime_core
    STATIC
        live_runtime/src/deployment_manifest.cpp
        live_runtime/src/diagnostics.cpp
        live_runtime/src/live_inference_pipeline.cpp
        live_runtime/src/threshold_override.cpp
        live_runtime/src/runtime_health.cpp
        live_runtime/src/systemd_notify.cpp
        live_runtime/src/thread_scheduling.cpp
        ${_dkfz_inference_engine_source}
)
target_compile_features(dkfz_live_runtime_core PUBLIC cxx_std_17)
target_include_directories(
    dkfz_live_runtime_core
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/live_runtime/include"
)
target_compile_options(
    dkfz_live_runtime_core
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -Wconversion
        -Wshadow
)
target_link_libraries(dkfz_live_runtime_core PUBLIC dkfz_live_core)
if(_dkfz_tensorrt_requested)
    target_include_directories(
        dkfz_live_runtime_core
        PRIVATE "${TENSORRT_INCLUDE_DIR}"
    )
    target_link_libraries(
        dkfz_live_runtime_core
        PRIVATE "${TENSORRT_LIBRARY}" CUDA::cudart
    )
    target_compile_definitions(dkfz_live_runtime_core PUBLIC DKFZ_HAS_TENSORRT=1)
    message(STATUS "DKFZ native TensorRT backend: enabled")
else()
    message(STATUS "DKFZ native TensorRT backend: disabled (stub runtime)")
endif()

add_library(
    dkfz_integrated_runtime_core
    STATIC
        live_runtime/src/input_validity.cpp
        live_runtime/src/integrated_runtime.cpp
        live_runtime/src/runtime_output.cpp
)
target_compile_features(dkfz_integrated_runtime_core PUBLIC cxx_std_17)
target_include_directories(
    dkfz_integrated_runtime_core
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/live_runtime/include"
)
target_compile_options(
    dkfz_integrated_runtime_core
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -Wconversion
        -Wshadow
)
target_link_libraries(
    dkfz_integrated_runtime_core
    PUBLIC
        dkfz_live_runtime_core
        dkfz_uart_core
        Threads::Threads
)

add_executable(
    native_live_runtime
    live_runtime/native_live_runtime.cpp
)
target_compile_features(native_live_runtime PRIVATE cxx_std_17)
target_compile_options(
    native_live_runtime
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -Wconversion
        -Wshadow
)
target_link_libraries(native_live_runtime PRIVATE dkfz_live_runtime_core)
set_target_properties(
    native_live_runtime
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/inference"
)

add_executable(
    native_uart_live_runtime
    live_runtime/native_uart_live_runtime.cpp
)
target_compile_features(native_uart_live_runtime PRIVATE cxx_std_17)
target_compile_options(
    native_uart_live_runtime
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -Wconversion
        -Wshadow
)
target_link_libraries(
    native_uart_live_runtime
    PRIVATE dkfz_integrated_runtime_core
)
set_target_properties(
    native_uart_live_runtime
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/inference"
)

add_executable(uart_receiver_stream uart_receiver_stream.cpp)
target_compile_features(uart_receiver_stream PRIVATE cxx_std_17)
target_compile_options(
    uart_receiver_stream
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -Wconversion
        -Wshadow
)
target_link_libraries(uart_receiver_stream PRIVATE dkfz_uart_core)

set_target_properties(
    uart_receiver_stream
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/uart"
)

install(
    TARGETS
        uart_receiver_stream
        native_live_runtime
        native_uart_live_runtime
    RUNTIME DESTINATION bin
)

# Deployment-native administration API; no Python or TensorRT dependency.
add_executable(native_live_recovery_control live_runtime/native_live_recovery_control.cpp)
target_compile_features(native_live_recovery_control PRIVATE cxx_std_17)
target_include_directories(native_live_recovery_control PRIVATE live_runtime/include)
target_compile_options(native_live_recovery_control PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(native_live_recovery_control PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/inference")
# The GUI's existing runtime rebuild targets native_uart_live_runtime alone.
# Keep the on-demand control executable in sync through that same build.
add_dependencies(native_uart_live_runtime native_live_recovery_control)

install(TARGETS native_live_recovery_control RUNTIME DESTINATION bin)
