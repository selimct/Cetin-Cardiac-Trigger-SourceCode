# Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Developed at the German Cancer Research Center (DKFZ).

from __future__ import annotations

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup


setup(
    ext_modules=[
        Pybind11Extension(
            "dkfz.native._live_core",
            [
                "native/python/live_core_bindings.cpp",
                "native/live_core/src/live_core.cpp",
            ],
            include_dirs=["native/live_core/include"],
            depends=["native/live_core/include/dkfz_live/live_core.hpp"],
            cxx_std=17,
            extra_compile_args=["-Wall", "-Wextra", "-Wpedantic"],
        )
    ],
    cmdclass={"build_ext": build_ext},
)
