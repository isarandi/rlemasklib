import numpy as np
from setuptools import setup, Extension
from Cython.Build import cythonize

import glob

all_c_files = glob.glob("src/rlemasklib/c/*.c")
single_translation_unit = 'src/rlemasklib/c/single_translation_unit.c'

use_single_translation_unit = True
if use_single_translation_unit:
    c_files = [single_translation_unit]
else:
    c_files = [f for f in all_c_files if f != single_translation_unit]

ext_modules = [
    Extension(
        'rlemasklib.rlemasklib_cython',
        sources=['src/rlemasklib/rlemasklib_cython.pyx'] + c_files,
        include_dirs=[np.get_include(), 'src/rlemasklib/c'],
        extra_compile_args=['-Wno-cpp', '-Wno-unused-function', '-std=c99', '-O3'],
    ),
    Extension(
        'rlemasklib.oop_cython',
        sources=['src/rlemasklib/oop_cython.pyx'] + c_files,
        include_dirs=[np.get_include(), 'src/rlemasklib/c'],
        extra_compile_args=['-Wno-cpp', '-Wno-unused-function', '-std=c99', '-O3'],
    ),
]

setup(ext_modules=cythonize(ext_modules))
