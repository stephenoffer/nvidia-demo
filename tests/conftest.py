"""Pytest configuration and shared fixtures."""

import os
import tempfile
from pathlib import Path
from typing import Generator

import pytest
import ray
from ray.data import Dataset

from pipeline.config import PipelineConfig


@pytest.fixture(scope="session")
def ray_cluster() -> Generator[None, None, None]:
    """Initialize Ray cluster for testing."""
    if not ray.is_initialized():
        ray.init(num_cpus=2, num_gpus=0, object_store_memory=100_000_000)
    yield
    if ray.is_initialized():
        ray.shutdown()


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_config(temp_dir: Path) -> PipelineConfig:
    """Create sample pipeline configuration."""
    return PipelineConfig(
        input_paths=[str(temp_dir / "input")],
        output_path=str(temp_dir / "output"),
        num_gpus=0,
        num_cpus=2,
        batch_size=10,
        enable_gpu_dedup=False,
        streaming=False,
    )


@pytest.fixture
def sample_dataset(ray_cluster: None) -> Dataset:
    """Create sample Ray Dataset for testing."""
    sample_data = [
        {"id": i, "value": float(i), "text": f"sample_{i}"}
        for i in range(100)
    ]
    return ray.data.from_items(sample_data)


@pytest.fixture
def mock_gpu_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock GPU availability for testing."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    try:
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    except ImportError:
        pass


@pytest.fixture(autouse=True)
def reset_ray_state():
    """Reset Ray state between tests."""
    yield
    if ray.is_initialized():
        try:
            ray.shutdown()
        except Exception:
            pass

