# Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Developed at the German Cancer Research Center (DKFZ).

"""Small TensorRT runtime wrapper for the live ECG LSTM engine."""

from __future__ import annotations

from pathlib import Path

from cuda.bindings import runtime as cudart
import numpy as np
import tensorrt as trt


def _check_cuda(result, action: str):
    """Raise RuntimeError if a cuda-python call failed."""
    if isinstance(result, tuple):
        err = result[0]
        values = result[1:]
    else:
        err = result
        values = ()

    if err != cudart.cudaError_t.cudaSuccess:
        name = cudart.cudaGetErrorName(err)
        if isinstance(name, tuple):
            name = name[1]
        if isinstance(name, bytes):
            name = name.decode("utf-8", errors="replace")
        raise RuntimeError(f"{action} failed: {name}")

    if len(values) == 1:
        return values[0]
    return values


class _DeviceBuffer:
    """Own one CUDA device allocation."""

    def __init__(self, nbytes: int):
        self.nbytes = int(nbytes)
        self.ptr = _check_cuda(cudart.cudaMalloc(self.nbytes), "cudaMalloc")

    def free(self) -> None:
        if self.ptr:
            _check_cuda(cudart.cudaFree(self.ptr), "cudaFree")
            self.ptr = 0

    def __del__(self):
        try:
            self.free()
        except Exception:
            pass


class TensorRTLogitEngine:
    """
    Run a TensorRT engine that maps one live feature window to one raw logit.

    The expected model contract is ``x: float32[1, 25, input_dim] -> logit``.
    Dynamic batch engines also work, but sequence length and feature dimension
    should stay fixed for this live pipeline.
    """

    def __init__(
        self,
        engine_path: str | Path,
        input_name: str = "x",
        output_name: str = "logit",
        input_shape: tuple[int, int, int] = (1, 25, 17),
    ):
        self.engine_path = Path(engine_path)
        self.input_name = input_name
        self.output_name = output_name
        self.input_shape = tuple(int(x) for x in input_shape)

        self.logger = trt.Logger(trt.Logger.WARNING)

        with self.engine_path.open("rb") as f:
            engine_bytes = f.read()
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(engine_bytes)
        if self.engine is None:
            raise RuntimeError(f"Could not deserialize TensorRT engine: {self.engine_path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Could not create TensorRT execution context.")

        self._validate_io_names()
        self._validate_input_contract()
        self._declared_input_shape = tuple(
            int(dim) for dim in self.engine.get_tensor_shape(self.input_name)
        )
        self.stream = _check_cuda(
            cudart.cudaStreamCreate(),
            "cudaStreamCreate",
        )
        self._input_buffer: _DeviceBuffer | None = None
        self._output_buffer: _DeviceBuffer | None = None
        self._output_host: np.ndarray | None = None

    def _validate_io_names(self) -> None:
        names = [
            self.engine.get_tensor_name(i)
            for i in range(self.engine.num_io_tensors)
        ]
        missing = [
            name for name in (self.input_name, self.output_name) if name not in names
        ]
        if missing:
            raise ValueError(
                f"TensorRT engine missing expected tensor(s) {missing}; available={names}"
            )

        input_mode = self.engine.get_tensor_mode(self.input_name)
        output_mode = self.engine.get_tensor_mode(self.output_name)
        if input_mode != trt.TensorIOMode.INPUT:
            raise ValueError(f"{self.input_name!r} is not an input tensor.")
        if output_mode != trt.TensorIOMode.OUTPUT:
            raise ValueError(f"{self.output_name!r} is not an output tensor.")

    def _validate_input_contract(self) -> None:
        """Validate that the serialized engine can consume the configured shape."""
        input_dtype = self.engine.get_tensor_dtype(self.input_name)
        expected_dtype = getattr(trt, "float32", None)
        if expected_dtype is None:
            expected_dtype = trt.DataType.FLOAT
        if input_dtype != expected_dtype:
            raise ValueError(
                f"TensorRT engine input {self.input_name!r} has dtype {input_dtype}; "
                "this live predictor expects float32 input."
            )

        engine_shape = tuple(
            int(dim) for dim in self.engine.get_tensor_shape(self.input_name)
        )
        self._validate_declared_input_shape(engine_shape)

    def _validate_declared_input_shape(self, engine_shape: tuple[int, ...]) -> None:
        expected = self.input_shape
        if len(engine_shape) != len(expected):
            raise ValueError(
                f"TensorRT engine input {self.input_name!r} has rank "
                f"{len(engine_shape)} with shape {engine_shape}; expected rank "
                f"{len(expected)} shape {expected}."
            )

        dynamic_axes = [axis for axis, dim in enumerate(engine_shape) if dim < 0]
        for axis, (actual, expected_dim) in enumerate(zip(engine_shape, expected)):
            if actual >= 0 and actual != expected_dim:
                raise ValueError(
                    f"TensorRT engine input shape {engine_shape} does not match "
                    f"configured shape {expected}: axis {axis} is {actual}, "
                    f"expected {expected_dim}."
                )

        if not dynamic_axes:
            return

        if not hasattr(self.engine, "get_tensor_profile_shape"):
            raise ValueError(
                f"TensorRT engine input shape {engine_shape} is dynamic, but this "
                "TensorRT runtime does not expose profile shape validation."
            )

        profile_index = 0
        min_shape, _opt_shape, max_shape = self.engine.get_tensor_profile_shape(
            self.input_name,
            profile_index,
        )
        min_shape = tuple(int(dim) for dim in min_shape)
        max_shape = tuple(int(dim) for dim in max_shape)
        if len(min_shape) != len(expected) or len(max_shape) != len(expected):
            raise ValueError(
                f"TensorRT profile {profile_index} for input {self.input_name!r} has "
                f"min/max shapes {min_shape}/{max_shape}; expected rank "
                f"{len(expected)}."
            )

        for axis, expected_dim in enumerate(expected):
            if axis == 0:
                if not min_shape[axis] <= expected_dim <= max_shape[axis]:
                    raise ValueError(
                        f"TensorRT profile {profile_index} batch range "
                        f"{min_shape[axis]}..{max_shape[axis]} does not allow "
                        f"configured batch {expected_dim}."
                    )
                continue

            if min_shape[axis] != expected_dim or max_shape[axis] != expected_dim:
                raise ValueError(
                    f"TensorRT profile {profile_index} allows dynamic sequence/feature "
                    f"shape {min_shape}..{max_shape}; this live predictor requires "
                    f"fixed axis {axis}={expected_dim}."
                )

    def _ensure_buffers(self, x: np.ndarray) -> None:
        if hasattr(self.context, "set_input_shape"):
            ok = self.context.set_input_shape(self.input_name, tuple(x.shape))
            if ok is False:
                raise ValueError(
                    f"TensorRT context rejected input shape {tuple(x.shape)} for "
                    f"{self.input_name!r}."
                )

        output_shape = tuple(int(v) for v in self.context.get_tensor_shape(self.output_name))
        if any(v < 0 for v in output_shape):
            raise RuntimeError(f"Output shape is still dynamic: {output_shape}")

        input_nbytes = int(x.nbytes)
        output_dtype = trt.nptype(self.engine.get_tensor_dtype(self.output_name))
        output_host = np.empty(output_shape, dtype=output_dtype)
        output_nbytes = int(output_host.nbytes)

        if self._input_buffer is None or self._input_buffer.nbytes != input_nbytes:
            if self._input_buffer is not None:
                self._input_buffer.free()
            self._input_buffer = _DeviceBuffer(input_nbytes)

        if self._output_buffer is None or self._output_buffer.nbytes != output_nbytes:
            if self._output_buffer is not None:
                self._output_buffer.free()
            self._output_buffer = _DeviceBuffer(output_nbytes)
            self._output_host = output_host
        elif self._output_host is None or self._output_host.shape != output_shape:
            self._output_host = output_host

        self.context.set_tensor_address(self.input_name, int(self._input_buffer.ptr))
        self.context.set_tensor_address(self.output_name, int(self._output_buffer.ptr))

    def infer(self, window: np.ndarray) -> np.ndarray:
        """Run the engine and return the output tensor as a NumPy array."""
        x = np.ascontiguousarray(window, dtype=np.float32)
        if x.ndim == 2:
            x = x[None, :, :]
        if x.shape[1:] != self.input_shape[1:]:
            raise ValueError(
                f"Expected window shape (*, {self.input_shape[1]}, {self.input_shape[2]}), "
                f"got {x.shape}."
            )
        declared_batch = self._declared_input_shape[0]
        if declared_batch >= 0 and x.shape[0] != declared_batch:
            raise ValueError(
                f"TensorRT engine has static batch {declared_batch}, but got "
                f"batch {x.shape[0]}. Rebuild the engine with a dynamic batch "
                "profile for batched offline inference."
            )

        self._ensure_buffers(x)
        assert self._input_buffer is not None
        assert self._output_buffer is not None
        assert self._output_host is not None

        h2d = cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
        d2h = cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
        _check_cuda(
            cudart.cudaMemcpyAsync(
                self._input_buffer.ptr,
                x.ctypes.data,
                int(x.nbytes),
                h2d,
                self.stream,
            ),
            "cudaMemcpyAsync H2D",
        )
        ok = self.context.execute_async_v3(stream_handle=self.stream)
        if not ok:
            raise RuntimeError("TensorRT execute_async_v3 returned False.")
        _check_cuda(
            cudart.cudaMemcpyAsync(
                self._output_host.ctypes.data,
                self._output_buffer.ptr,
                int(self._output_host.nbytes),
                d2h,
                self.stream,
            ),
            "cudaMemcpyAsync D2H",
        )
        _check_cuda(
            cudart.cudaStreamSynchronize(self.stream),
            "cudaStreamSynchronize",
        )
        return np.array(self._output_host, copy=True)

    def infer_logit(self, window: np.ndarray) -> float:
        """Run inference and return the first scalar logit."""
        output = self.infer(window).reshape(-1)
        if output.size != 1:
            raise ValueError(f"Expected one scalar logit, got output shape {output.shape}.")
        return float(output[0])

    def infer_logits(self, windows: np.ndarray) -> np.ndarray:
        """Run one or more windows and return flattened logits."""
        output = self.infer(windows).reshape(-1)
        expected = 1 if windows.ndim == 2 else int(windows.shape[0])
        if output.size != expected:
            raise ValueError(
                f"Expected {expected} logits for input batch, got output shape "
                f"{output.shape}."
            )
        return output.astype(np.float32, copy=False)

    def close(self) -> None:
        """Release owned CUDA allocations."""
        if self._input_buffer is not None:
            self._input_buffer.free()
            self._input_buffer = None
        if self._output_buffer is not None:
            self._output_buffer.free()
            self._output_buffer = None
        if getattr(self, "stream", None):
            _check_cuda(cudart.cudaStreamDestroy(self.stream), "cudaStreamDestroy")
            self.stream = 0

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
