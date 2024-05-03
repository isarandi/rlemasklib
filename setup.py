import numpy as np
from setuptools import setup, Extension

# To compile and install locally run "python setup.py build_ext --inplace"
# To install library to Python site-packages run "python setup.py build_ext install"
ext_modules = [
    Extension(
        'rlemasklib.rlemasklib_cython',
        sources=['rlemasklib/rlemasklib_cython.pyx'],
        include_dirs=[np.get_include(), 'rlemasklib'],
        extra_compile_args=['-Wno-cpp', '-Wno-unused-function', '-std=c99'],
    )
]

setup(
    name='rlemasklib',
    version='0.2.0',
    author='István Sárándi',
    author_email='istvan.sarandi@uni-tuebingen.de',
    packages=['rlemasklib'],
    install_requires=[
        'setuptools>=18.0',
        'numpy',
        'cython>=0.27.3',
    ],
    ext_modules=ext_modules
)