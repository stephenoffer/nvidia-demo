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

**Quick Start (Simplest):**

```python
from pipeline.api import Pipeline

# One-liner pipeline creation
pipeline = Pipeline.quick_start(
    input_paths="data/videos/",
    output_path="data/output/",
    enable_gpu=False
)

results = pipeline.run()
```

**Full Declarative API:**

```python
from pipeline.api import Pipeline

# Simple declarative API
pipeline = Pipeline(
    sources=[
        {"type": "video", "path": "data/videos/"},
        {"type": "text", "path": "data/text/"},
    ],
    output="data/output/",
    enable_gpu=False,
)

# Pythonic features
print(f"Sources: {len(pipeline)}")  # len() support
for source in pipeline:  # Iteration
    print(source)
if "data/videos/" in pipeline:  # Membership check
    print("Video source found")

results = pipeline.run()
```

**Fluent Builder API:**

```python
from pipeline.api import PipelineBuilder

# Method chaining for complex pipelines
pipeline = (
    PipelineBuilder()
    .add_source("data/videos/")
    .add_source("data/text/")
    .enable_gpu(num_gpus=4)
    .set_batch_size(32)
    .add_profiler(columns=["image", "text"])
    .set_output("data/output/")
    .build()
)

results = pipeline.run()
```

**DataFrame API (Pythonic):**

```python
from pipeline.api import PipelineDataFrame

# Create DataFrame
df = PipelineDataFrame.from_paths(["data/input/"])

# Use standard Python built-ins
print(f"Rows: {len(df)}")  # len() support
print(f"Shape: {df.shape}")  # (rows, columns)
print(f"Columns: {df.columns}")  # Column names

# Pythonic indexing
first_10 = df[0:10]  # Slicing (like Pandas)
column = df["episode_id"]  # Column access
value = df.episode_id  # Attribute-style access

# Operator overloading
df1 = PipelineDataFrame.from_paths(["data/data1/"])
df2 = PipelineDataFrame.from_paths(["data/data2/"])
combined = df1 + df2  # Concatenate (like pd.concat)

# Lazy transformations
result = (
    df
    .filter(lambda x: x["quality"] > 0.8)
    .map(lambda x: {**x, "processed": True})
    .groupby("episode_id")
    .agg({"sensor_data": "mean"})
    .collect()  # Trigger execution
)
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

