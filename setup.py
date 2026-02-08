import os
import subprocess
import sys
import numpy as np
from setuptools import setup, Extension
from Cython.Build import cythonize

import glob

all_c_files = glob.glob("src/rlemasklib/c/*.c")
single_translation_unit = "src/rlemasklib/c/single_translation_unit.c"

use_single_translation_unit = True
if use_single_translation_unit:
    c_files = [single_translation_unit]
else:
    c_files = [f for f in all_c_files if f != single_translation_unit]

# Platform-specific compiler arguments
# MSVC uses different flag syntax than GCC/Clang
if sys.platform == "win32":
    extra_compile_args = [
        "/O2",
        "/wd4505",
    ]  # /O2=optimize, /wd4505=disable unused function warning
    extra_link_args = []
else:
    extra_compile_args = ["-Wno-cpp", "-Wno-unused-function", "-std=c99", "-O3"]
    extra_link_args = []

include_dirs = [np.get_include(), "src/rlemasklib/c"]
library_dirs = []

# Support custom libdeflate install location via LIBDEFLATE_DIR env var
libdeflate_dir = os.environ.get("LIBDEFLATE_DIR")
if libdeflate_dir:
    include_dirs.append(os.path.join(libdeflate_dir, "include"))
    library_dirs.append(os.path.join(libdeflate_dir, "lib"))

# Check for libdeflate (required for PNG-to-RLE)
try:
    deflate_cflags = subprocess.check_output(
        ["pkg-config", "--cflags", "libdeflate"], stderr=subprocess.DEVNULL
    ).decode().strip().split()
    deflate_libs = subprocess.check_output(
        ["pkg-config", "--libs", "libdeflate"], stderr=subprocess.DEVNULL
    ).decode().strip().split()
    extra_compile_args.extend(deflate_cflags)
    extra_link_args.extend(deflate_libs)
except (subprocess.CalledProcessError, FileNotFoundError):
    if sys.platform == "win32":
        extra_link_args.append("deflate.lib")
    else:
        extra_link_args.append("-ldeflate")

ext_modules = [
    Extension(
        "rlemasklib.rlemasklib_cython",
        sources=["src/rlemasklib/rlemasklib_cython.pyx"] + c_files,
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    Extension(
        "rlemasklib.oop_cython",
        sources=["src/rlemasklib/oop_cython.pyx"] + c_files,
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
]

setup(ext_modules=cythonize(ext_modules))
