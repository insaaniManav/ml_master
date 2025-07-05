#!/usr/bin/env python3
"""
Setup script for ML-Master: AI-for-AI Framework
"""

from setuptools import setup, find_packages
import os

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ml-master",
    version="1.0.0",
    author="ML-Master Team",
    author_email="ml-master@example.com",
    description="AI-for-AI Framework with Adaptive Memory and Steerable Reasoning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ml-master/ml-master",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.17.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ml-master=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json"],
    },
    keywords="machine-learning ai-for-ai artificial-intelligence mcts reasoning exploration",
    project_urls={
        "Bug Reports": "https://github.com/ml-master/ml-master/issues",
        "Source": "https://github.com/ml-master/ml-master",
        "Documentation": "https://ml-master.readthedocs.io/",
    },
) 