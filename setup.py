"""Setup script for multimodal data pipeline package."""

from pathlib import Path

from setuptools import find_packages, setup

# Read version from __version__.py
def get_version():
    """Get version from __version__.py."""
    version_file = Path(__file__).parent / "pipeline" / "__version__.py"
    if version_file.exists():
        with open(version_file) as f:
            for line in f:
                if line.startswith("__version__"):
                    return line.split("=")[1].strip().strip('"').strip("'")
    return "0.1.0"

# Read README for long description
def get_long_description():
    """Get long description from README.md."""
    readme_file = Path(__file__).parent / "README.md"
    if readme_file.exists():
        return readme_file.read_text(encoding="utf-8")
    return ""

setup(
    name="multimodal-data-pipeline",
    version=get_version(),
    description="Production-ready multimodal data curation pipeline for robotics foundation models",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    author="NVIDIA Demo",
    author_email="demo@nvidia.com",
    url="https://github.com/your-org/nvidia-demo",
    packages=find_packages(exclude=["tests", "tests.*", "demos", "demos.*"]),
    python_requires=">=3.9,<3.12",
    install_requires=[
        "ray[data]>=2.10.0",
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "pyarrow>=12.0.0",
        "opencv-python>=4.8.0",
        "pillow>=10.0.0",
        "transformers>=4.30.0",
        "sentencepiece>=0.1.99",
        "scikit-learn>=1.3.0",
        "numba>=0.57.0",
        "cupy-cuda12x>=12.0.0",
        "pydantic>=2.0.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "matplotlib>=3.7.0",
        "plotly>=5.14.0",
        "h5py>=3.9.0",
        "imageio>=2.31.0",
        "imageio-ffmpeg>=0.4.9",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-benchmark>=4.0.0",
            "pytest-xdist>=3.0.0",
            "ruff>=0.1.0",
            "mypy>=1.0.0",
            "types-pyyaml>=6.0.0",
            "types-requests>=2.31.0",
            "pre-commit>=3.0.0",
            "bandit>=1.7.0",
            "safety>=2.0.0",
        ],
        "gpu": [
            "cudf-cu12>=23.12.0",
            "cuml-cu12>=23.12.0",
        ],
        "monitoring": [
            "prometheus-client>=0.18.0",
            "psutil>=5.9.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "pipeline=pipeline.cli:main",
            "pipeline-health=pipeline.cli:health_check",
        ],
    },
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
    include_package_data=True,
    zip_safe=False,
)

