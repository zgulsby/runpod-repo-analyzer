"""Setup script for runpod-repo-analyzer package."""

from setuptools import setup, find_packages

setup(
    name="runpod-repo-analyzer",
    version="0.1.0",
    description="A tool to analyze repositories and generate RunPod-required files",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "pyyaml>=6.0",
        "toml>=0.10.2",
        "runpod>=1.0.0",
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "tensorflow>=2.13.0",
        "numpy>=1.24.0",
        "gitpython>=3.1.0",
        "fastapi>=0.104.0",
        "pydantic>=2.4.2",
        "uvicorn>=0.24.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-cov>=4.1.0",
            "black>=23.10.1",
            "isort>=5.12.0",
            "flake8>=6.1.0",
            "mypy>=1.6.1",
        ]
    },
    entry_points={
        "console_scripts": [
            "runpod-analyze=runpod_analyzer.main:main"
        ]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
) 