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

**Simple Function API:**

```python
from pipeline.api import pipeline

# Simple function API - easiest way
p = pipeline(
    sources="data/videos/",
    output="data/output/",
    num_gpus=0
)

results = p.run()
```

**Declarative API:**

```python
from pipeline.api import Pipeline

# Declarative API
p = Pipeline.create(
    sources=[
        {"type": "video", "path": "data/videos/"},
        {"type": "text", "path": "data/text/"},
    ],
    output="data/output/",
    num_gpus=0,
)

# Pythonic features
print(f"Sources: {len(p)}")  # len() support
for source in p:  # Iteration
    print(source)
if "data/videos/" in p:  # Membership check
    print("Video source found")

results = p.run()
```

**Fluent Builder API:**

```python
from pipeline.api import PipelineBuilder

# Method chaining with short names
p = (
    PipelineBuilder()
    .source("video", "data/videos/")
    .source("text", "data/text/")
    .gpu(num_gpus=4)
    .batch(32)
    .profile(profile_columns=["image", "text"])
    .output("data/output/")
    .build()
)

results = p.run()
```

**DataFrame API (Pythonic):**

```python
from pipeline.api import read

# Read data into DataFrame
df = read("data/input/")

# Use standard Python built-ins
print(f"Rows: {len(df)}")  # len() support
print(f"Shape: {df.shape}")  # (rows, columns)
print(f"Columns: {df.columns}")  # Column names

# Pythonic indexing
first_10 = df[0:10]  # Slicing (like Pandas)
column = df["episode_id"]  # Column access
value = df.episode_id  # Attribute-style access

# Pandas-style methods
df.drop("unused_col")  # Drop columns
df.to_parquet("output.parquet")  # Write to Parquet
df.assign(status="active")  # Add columns

# Operator overloading
df1 = read("data/data1/")
df2 = read("data/data2/")
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

