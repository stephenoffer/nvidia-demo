# Deployment Guide

## Production Deployment

### Prerequisites

- Python 3.9+
- Ray cluster (local or distributed)
- CUDA-capable GPUs (optional, for GPU acceleration)
- Sufficient disk space for data processing

### Installation

```bash
# Install package
pip install -e .

# Install with GPU support
pip install -e ".[gpu]"
```

### Configuration

Create a configuration file for your deployment:

```yaml
# config.yaml
input_paths:
  - "s3://bucket/input/"
output_path: "s3://bucket/output/"
num_gpus: 4
num_cpus: 16
batch_size: 1000
enable_gpu_dedup: true
streaming: true
checkpoint_interval: 100
```

### Running the Pipeline

```python
from pipeline.api.declarative import Pipeline
import yaml

with open("config.yaml") as f:
    config = yaml.safe_load(f)

pipeline = Pipeline(**config)
results = pipeline.run()
```

### Kubernetes Deployment

See `deployment/kubernetes.yaml` for Kubernetes deployment configuration.

### Environment Variables

- `RAY_ADDRESS`: Ray cluster address (optional)
- `CUDA_VISIBLE_DEVICES`: GPU devices to use
- `NCCL_DEBUG`: NCCL debug level
- `RAY_NAMESPACE`: Ray namespace

### Monitoring

The pipeline includes built-in metrics and monitoring:

```python
from pipeline.observability.metrics import PipelineMetrics

metrics = PipelineMetrics(enabled=True)
pipeline = MultimodalPipeline(config)
results = pipeline.run()

# Access metrics
current_metrics = pipeline.get_metrics()
```

### Health Checks

Health check endpoints are available for monitoring:

```python
from pipeline.utils.ray.monitoring import check_cluster_health

health = check_cluster_health()
if not health["healthy"]:
    # Handle unhealthy cluster
    pass
```

### Scaling

For large-scale deployments:

1. **Horizontal Scaling**: Add more Ray workers
2. **GPU Scaling**: Increase `num_gpus` in config
3. **Streaming**: Enable streaming mode for large datasets
4. **Checkpointing**: Enable checkpointing for fault tolerance

### Troubleshooting

#### Common Issues

1. **Out of Memory**: Reduce batch size or enable streaming
2. **GPU Not Available**: Check CUDA installation and visibility
3. **Ray Connection Issues**: Verify Ray cluster is running
4. **Slow Performance**: Enable GPU acceleration and optimize batch sizes

### Production Best Practices

1. **Enable Checkpointing**: Set `checkpoint_interval` for fault tolerance
2. **Use Streaming**: Enable `streaming=True` for large datasets
3. **Monitor Resources**: Use built-in monitoring and metrics
4. **Error Handling**: Implement proper error handling and retries
5. **Logging**: Configure appropriate log levels for production

