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
    extra_compile_args = ["-Wno-cpp", "-Wno-unused-function", "-Wno-int-in-bool-context", "-std=c99", "-O3"]
    extra_link_args = []

include_dirs = [np.get_include(), "src/rlemasklib/c"]
library_dirs = []

# Support custom libdeflate install location via LIBDEFLATE_DIR env var
libdeflate_dir = os.environ.get("LIBDEFLATE_DIR")
if libdeflate_dir:
    include_dirs.append(os.path.join(libdeflate_dir, "include"))
    library_dirs.append(os.path.join(libdeflate_dir, "lib"))


def _env_flag_dirs(var, prefix):
    """Extract directories from flags like -I/path or -L/path in an env var."""
    return [tok[len(prefix):] for tok in os.environ.get(var, "").split()
            if tok.startswith(prefix) and len(tok) > len(prefix)]


def _probe_libdeflate(probe_include_dirs, probe_library_dirs):
    """Try to compile and link a minimal program against libdeflate.

    Returns True if it links, False if libdeflate is missing, None if the probe
    itself cannot run (e.g. no working compiler) and nothing can be concluded.
    """
    import tempfile
    from distutils.ccompiler import new_compiler
    from distutils.sysconfig import customize_compiler

    baseline = "int main(void) { return 0; }\n"
    program = (
        "#include <libdeflate.h>\n"
        "int main(void) {\n"
        "    libdeflate_free_decompressor(libdeflate_alloc_decompressor());\n"
        "    return 0;\n"
        "}\n"
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        def compiles_and_links(name, code, include_dirs, library_dirs, libraries):
            src = os.path.join(tmpdir, name + ".c")
            with open(src, "w") as f:
                f.write(code)
            cc = new_compiler()
            customize_compiler(cc)
            objs = cc.compile([src], output_dir=tmpdir, include_dirs=include_dirs)
            cc.link_executable(
                objs, os.path.join(tmpdir, name),
                library_dirs=library_dirs, libraries=libraries,
            )

        try:
            compiles_and_links("baseline", baseline, [], [], [])
        except Exception:
            return None
        # Baseline proved the toolchain works, so a failure here means libdeflate
        # is missing. (Compiler error classes vary across distutils/setuptools
        # versions, so no narrower except clause is portable.)
        try:
            compiles_and_links(
                "deflate_probe", program, probe_include_dirs, probe_library_dirs,
                ["deflate"],
            )
        except Exception:
            return False
    return True


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
        # Use static lib on Windows to avoid DLL shipping
        extra_link_args.append("deflatestatic.lib")
    else:
        probe_include_dirs = (
            include_dirs
            + _env_flag_dirs("CPPFLAGS", "-I")
            + _env_flag_dirs("CFLAGS", "-I")
        )
        probe_library_dirs = library_dirs + _env_flag_dirs("LDFLAGS", "-L")
        if _probe_libdeflate(probe_include_dirs, probe_library_dirs) is False:
            raise SystemExit(
                "error: libdeflate was not found (and pkg-config could not locate it).\n"
                "rlemasklib requires libdeflate for PNG decoding. Install it, e.g.:\n"
                "    Debian/Ubuntu:  apt install libdeflate-dev\n"
                "    Fedora:         dnf install libdeflate-devel\n"
                "    macOS:          brew install libdeflate\n"
                "    conda:          conda install libdeflate\n"
                "or set LIBDEFLATE_DIR to its install prefix, or pass its location\n"
                "via CFLAGS=-I<include dir> and LDFLAGS=-L<lib dir>."
            )
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
