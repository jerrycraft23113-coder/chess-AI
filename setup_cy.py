"""Build script for Cython-accelerated eval module.

Usage:
    python setup_cy.py build_ext --inplace
"""
from setuptools import setup, Extension
from Cython.Build import cythonize
import sys

extra_compile_args = ["/O2"] if sys.platform == "win32" else ["-O3", "-march=native"]

setup(
    ext_modules=cythonize(
        Extension("eval_cy", ["eval_cy.pyx"], extra_compile_args=extra_compile_args),
        language_level=3,
    ),
)
