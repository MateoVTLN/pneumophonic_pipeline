"""
Setup script for pneumophonic_analysis package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

setup(
    name="pneumophonic_analysis",
    version="1.0.0",
    author="Based on work by Bianca Zocco",
    author_email="",
    description="Pipeline for integrated respiratory-phonatory analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
        "librosa>=0.9.0",
        "soundfile>=0.10.0",
        "noisereduce>=2.0.0",
        "praat-parselmouth>=0.4.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "openpyxl>=3.0.0",
        "tqdm>=4.62.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=2.12.0",
            "black>=21.0",
            "flake8>=3.9.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=0.5.0",
        ],
        "notebook": [
            "jupyter>=1.0.0",
            "ipywidgets>=7.6.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "pneumophonic-analyze=pneumophonic_analysis.pipeline:main",
        ],
    },
)
