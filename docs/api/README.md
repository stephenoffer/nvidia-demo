# API Documentation

## Core Pipeline

### MultimodalPipeline

Main pipeline orchestrator for multimodal data curation.

```python
from pipeline.core import MultimodalPipeline
from pipeline.config import PipelineConfig

config = PipelineConfig(
    input_paths=["s3://bucket/input/"],
    output_path="s3://bucket/output/",
    num_gpus=4,
    batch_size=1000,
)

pipeline = MultimodalPipeline(config)
results = pipeline.run()
```

#### Methods

- `run()`: Execute the complete pipeline
- `add_stage(stage)`: Add custom processing stage
- `enable_visualization()`: Enable visualization features
- `get_metrics()`: Get current pipeline metrics
- `shutdown()`: Shutdown pipeline and cleanup resources

#### Pythonic Features

The `Pipeline` class supports standard Python built-ins and operators:

```python
from pipeline.api import pipeline

p1 = pipeline(sources="s3://bucket/data1/", output="s3://output/")
p2 = pipeline(sources="s3://bucket/data2/", output="s3://output/")

# Standard Python built-ins
print(f"Number of sources: {len(p1)}")  # len() support
for source in p1:  # Iteration
    print(source)
if p1:  # Boolean check
    print("Pipeline is configured")
if "s3://bucket/data1/" in p1:  # Membership check
    print("Source found")

# Operator overloading
combined = p1 + p2  # Combine sources
```

## DataFrame API

### PipelineDataFrame

A Pythonic DataFrame API inspired by Spark, Polars, and Pandas, with full support for standard Python built-ins and operators.

#### Basic Usage

```python
from pipeline.api import read

# Read data into DataFrame
df = read("s3://bucket/data/")

# Standard Python built-ins
print(f"Rows: {len(df)}")  # Number of rows
print(f"Shape: {df.shape}")  # (rows, columns) tuple
print(f"Columns: {df.columns}")  # List of column names
print(f"Empty: {df.empty}")  # Boolean check

# Pythonic indexing and slicing
column = df["episode_id"]  # Column access
first_row = df[0]  # Row indexing
first_10 = df[0:10]  # Slicing (like Pandas)
selected = df[["col1", "col2"]]  # Multiple columns
value = df.episode_id  # Attribute-style access

# Pandas-style methods
df.drop("unused_col")  # Drop columns (Pandas-style)
df.to_parquet("output.parquet")  # Write to Parquet (Pandas-style)
df.assign(status="active")  # Add columns (Pandas-style)

# Operator overloading
df1 = read("s3://bucket/data1/")
df2 = read("s3://bucket/data2/")
combined = df1 + df2  # Concatenate (like pd.concat)
union = df1 | df2  # Alternative union syntax

# Iteration and membership
for row in df:  # Iterate rows
    process_row(row)
if {"episode_id": "123"} in df:  # Check membership
    print("Row found")

# Copy
df_copy = df.copy()  # Create copy
```

#### Transformations

```python
# Lazy transformations with method chaining
result = (
    df
    .filter(lambda x: x["quality"] > 0.8)
    .map(lambda x: {**x, "processed": True})
    .groupby("episode_id")
    .agg({"sensor_data": "mean", "timestamp": "max"})
    .join(other_df, on="episode_id")
    .sort("timestamp")
    .limit(1000)
    .cache()  # Cache intermediate result
    .collect()  # Trigger execution
)
```

#### GPU Acceleration

```python
# GPU-accelerated batch processing
processed = df.map_batches(
    lambda batch: transform_batch(batch),
    batch_size=1000,
    use_gpu=True,
    num_gpus=4,
)
```

#### Pythonic Features Summary

**Built-ins:**
- `len(df)` - Get number of rows (Python convention)
- `iter(df)` / `for row in df` - Iterate over rows
- `bool(df)` / `if df` - Check if non-empty
- `value in df` - Check membership
- `str(df)` - String representation

**Operators:**
- `df1 + df2` - Concatenate DataFrames (like `pd.concat()`)
- `df1 | df2` - Union DataFrames (alternative syntax)
- `df == other` - Check if same dataset object
- `df != other` - Check if different dataset objects

**Indexing:**
- `df["column"]` - Column access (returns list of values)
- `df.column` - Attribute-style column access (like Pandas)
- `df[0]` - Row indexing (returns row dict)
- `df[0:10]` - Row slicing (returns new DataFrame)
- `df[["col1", "col2"]]` - Multiple column selection (returns DataFrame)

**Properties:**
- `df.shape` - Get (rows, columns) tuple
- `df.columns` - Get list of column names
- `df.empty` - Check if empty
- `df.dataset` - Access underlying Ray Dataset

**Methods:**
- `df.copy()` - Create a copy (shallow copy, lazy evaluation)

## GPU Utilities

### GPU ETL Operations

```python
from pipeline.utils.gpu.etl import gpu_join, gpu_filter, gpu_sort

# GPU-accelerated join
result = gpu_join(left_df, right_df, on=["id"], num_gpus=1)

# GPU-accelerated filter
filtered = gpu_filter(df, lambda x: x["value"] > 100, num_gpus=1)

# GPU-accelerated sort
sorted_df = gpu_sort(df, by=["value"], num_gpus=1)
```

### GPU Array Operations

```python
from pipeline.utils.gpu.arrays import gpu_array_stats, gpu_normalize

# Compute statistics on GPU
stats = gpu_array_stats(array, num_gpus=1)

# Normalize array on GPU
normalized = gpu_normalize(array, num_gpus=1)
```

## Ray Utilities

### Ray Initialization

```python
from pipeline.utils.ray.init import initialize_ray, shutdown_ray

# Initialize Ray cluster
result = initialize_ray(
    num_cpus=8,
    num_gpus=4,
    object_store_memory=10_000_000_000,
)

# Shutdown Ray
shutdown_ray(graceful=True)
```

### Ray Monitoring

```python
from pipeline.utils.ray.monitoring import get_cluster_metrics, log_cluster_status

# Get cluster metrics
metrics = get_cluster_metrics()

# Log cluster status
log_cluster_status()
```

## Configuration

### PipelineConfig

Main configuration class for pipeline execution.

```python
from pipeline.config import PipelineConfig

config = PipelineConfig(
    input_paths=["s3://bucket/input/"],
    output_path="s3://bucket/output/",
    num_gpus=4,
    num_cpus=16,
    batch_size=1000,
    enable_gpu_dedup=True,
    streaming=True,
    checkpoint_interval=100,
)
```

## Stages

### VideoProcessor

Process video data through the pipeline.

```python
from pipeline.stages.video import VideoProcessor

processor = VideoProcessor(
    config=config,
    resolution=(1280, 720),
    fps=30,
)
```

### TextProcessor

Process text data through the pipeline.

```python
from pipeline.stages.text import TextProcessor

processor = TextProcessor(
    max_length=512,
    tokenizer_name="bert-base-uncased",
)
```

### SensorProcessor

Process sensor data through the pipeline.

```python
from pipeline.stages.sensor import SensorProcessor

processor = SensorProcessor(
    sample_rate=100,
    normalize=True,
)
```

