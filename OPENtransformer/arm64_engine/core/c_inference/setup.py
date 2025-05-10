#!/usr/bin/env python3
"""
Setup script for the Python interface module for the C wrapper.
"""

from setuptools import setup, find_packages

setup(
    name="py_diffusion_interface",
    version="0.1.0",
    description="Python interface for C-based diffusion inference",
    author="OPENtransformer Team",
    packages=find_packages(),
    install_requires=[
        "torch",
        "diffusers",
        "transformers",
        "numpy",
        "Pillow",
    ],
    python_requires=">=3.6",
) 