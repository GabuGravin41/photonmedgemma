"""
PhotoMedGemma — Photonic Compiler for MedGemma
===============================================
Setup script for pip install.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="photomedgemma",
    version="0.1.0",
    description="Static compilation of MedGemma onto photonic chip substrates",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="PhotoMedGemma Contributors",
    license="Apache-2.0",
    python_requires=">=3.10",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "full": [
            "torch>=2.1.0",
            "transformers>=4.40.0",
            "huggingface_hub>=0.20.0",
            "safetensors>=0.4.0",
        ],
        "layout": [
            "gdsfactory>=7.0.0",
        ],
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "jupyterlab>=4.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "photomedgemma-analyze=scripts.analyze_model:main",
            "photomedgemma-compile=scripts.compile_model:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
