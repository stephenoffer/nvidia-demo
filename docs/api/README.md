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

