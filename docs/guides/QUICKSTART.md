# Quick Start Guide

## Installation

```bash
# Clone repository
git clone <repository-url>
cd nvidia-demo

# Install dependencies
pip install -e ".[dev]"
```

## Basic Usage

### Simple Pipeline

```python
from pipeline.core import MultimodalPipeline
from pipeline.config import PipelineConfig

# Create configuration
config = PipelineConfig(
    input_paths=["data/input/"],
    output_path="data/output/",
    num_gpus=0,  # Set to 0 for CPU-only
    batch_size=100,
)

# Create and run pipeline
pipeline = MultimodalPipeline(config)
results = pipeline.run()

print(f"Pipeline completed: {results}")
```

### Using Declarative API

```python
from pipeline.api.declarative import Pipeline

# Simple declarative API
pipeline = Pipeline(
    sources=[
        {"type": "video", "path": "data/videos/"},
        {"type": "text", "path": "data/text/"},
    ],
    output="data/output/",
    enable_gpu=False,
)

results = pipeline.run()
```

### With GPU Acceleration

```python
from pipeline.core import MultimodalPipeline
from pipeline.config import PipelineConfig

config = PipelineConfig(
    input_paths=["data/input/"],
    output_path="data/output/",
    num_gpus=4,  # Enable GPU acceleration
    batch_size=1000,
    enable_gpu_dedup=True,
    streaming=True,
)

pipeline = MultimodalPipeline(config)
results = pipeline.run()
```

## Adding Custom Stages

```python
from pipeline.stages.base import PipelineStage
from ray.data import Dataset

class CustomStage(PipelineStage):
    def process(self, dataset: Dataset) -> Dataset:
        def transform_batch(batch):
            # Your transformation logic
            return batch
        
        return dataset.map_batches(transform_batch, batch_size=self.batch_size)

# Add to pipeline
pipeline = MultimodalPipeline(config)
pipeline.add_stage(CustomStage())
results = pipeline.run()
```

## Running Tests

```bash
# Run all tests
pytest

# Run unit tests only
pytest tests/unit/

# Run with coverage
pytest --cov=pipeline --cov-report=html

# Run benchmarks
pytest tests/benchmarks/ --benchmark-only
```

## Next Steps

- See [API Documentation](../api/README.md) for detailed API reference
- See [Deployment Guide](../deployment/README.md) for production deployment
- Check `examples/` directory for more examples

