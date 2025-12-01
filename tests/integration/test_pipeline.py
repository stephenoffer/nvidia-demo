"""Integration tests for full pipeline execution."""

import tempfile
from pathlib import Path

import pytest
import ray
from ray.data import Dataset

from pipeline.config import PipelineConfig
from pipeline.core.orchestrator import MultimodalPipeline


@pytest.fixture(scope="module")
def ray_cluster():
    """Initialize Ray cluster for integration tests."""
    if not ray.is_initialized():
        ray.init(num_cpus=2, num_gpus=0, object_store_memory=100_000_000)
    yield
    if ray.is_initialized():
        ray.shutdown()


class TestPipelineIntegration:
    """Integration tests for full pipeline."""

    def test_pipeline_initialization(self, ray_cluster, temp_dir: Path):
        """Test pipeline can be initialized."""
        config = PipelineConfig(
            input_paths=[str(temp_dir / "input")],
            output_path=str(temp_dir / "output"),
            num_gpus=0,
            num_cpus=2,
            batch_size=10,
            enable_gpu_dedup=False,
            streaming=False,
        )
        
        pipeline = MultimodalPipeline(config)
        assert pipeline is not None
        assert len(pipeline.stages) > 0

    def test_pipeline_with_sample_data(self, ray_cluster, temp_dir: Path):
        """Test pipeline with sample data."""
        input_dir = temp_dir / "input"
        input_dir.mkdir()
        
        sample_file = input_dir / "sample.jsonl"
        with open(sample_file, "w") as f:
            for i in range(10):
                f.write(f'{{"id": {i}, "value": {i * 10}}}\n')
        
        config = PipelineConfig(
            input_paths=[str(input_dir)],
            output_path=str(temp_dir / "output"),
            num_gpus=0,
            num_cpus=2,
            batch_size=5,
            enable_gpu_dedup=False,
            streaming=False,
        )
        
        pipeline = MultimodalPipeline(config)
        assert pipeline is not None

