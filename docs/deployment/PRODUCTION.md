# Production Deployment Guide

## Production Configuration

### Environment Setup

```bash
# Set environment variables
export RAY_ADDRESS="ray://head-node:10001"
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export NCCL_DEBUG="WARN"
export RAY_NAMESPACE="production"
```

### Configuration File

Create `config/production.yaml`:

```yaml
# Production configuration
input_paths:
  - "s3://production-bucket/input/"
output_path: "s3://production-bucket/output/"
num_gpus: 8
num_cpus: 32
batch_size: 5000
enable_gpu_dedup: true
streaming: true
checkpoint_interval: 50
enable_observability: true

# Ray Data configuration
ray_data_config:
  execution_cpu: 16.0
  execution_gpu: 4.0
  min_block_size: 1000000
  max_block_size: 10000000
  preserve_order: false

# GPU analytics
gpu_analytics_config:
  enabled: true
  target_columns: ["value", "score"]
  metrics: ["mean", "std", "min", "max"]
  normalize: true
  num_gpus: 2

# Deduplication
dedup_method: "semantic"
similarity_threshold: 0.95
```

### Health Monitoring

```python
from pipeline.health import HealthChecker

# Check overall health
health = HealthChecker.get_overall_health(output_path="s3://bucket/output/")
print(f"System healthy: {health['healthy']}")

# Individual checks
ray_health = HealthChecker.check_ray_cluster()
gpu_health = HealthChecker.check_gpu_resources()
disk_health = HealthChecker.check_disk_space("s3://bucket/output/", required_gb=100)
```

### Monitoring Integration

#### Prometheus Metrics

```python
from pipeline.observability.metrics import PipelineMetrics

metrics = PipelineMetrics(enabled=True)
pipeline = MultimodalPipeline(config)
results = pipeline.run()

# Export metrics for Prometheus
metrics_data = metrics.get_current_metrics()
```

#### Logging Configuration

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)
```

### Scaling Recommendations

#### Small Scale (< 1 TB)
- `num_gpus`: 1-2
- `num_cpus`: 4-8
- `batch_size`: 1000
- `streaming`: false

#### Medium Scale (1-10 TB)
- `num_gpus`: 4-8
- `num_cpus`: 16-32
- `batch_size`: 5000
- `streaming`: true

#### Large Scale (10+ TB)
- `num_gpus`: 8-16
- `num_cpus`: 32-64
- `batch_size`: 10000
- `streaming`: true
- `checkpoint_interval`: 100

### Fault Tolerance

1. **Enable Checkpointing**: Set `checkpoint_interval` for recovery
2. **Use Streaming**: Enable streaming for large datasets
3. **Monitor Resources**: Use health checks regularly
4. **Error Handling**: Implement retry logic for external services

### Security Best Practices

1. **Input Validation**: Always validate input paths
2. **Access Control**: Use IAM roles for S3 access
3. **Secrets Management**: Store credentials securely
4. **Network Security**: Use VPC for Ray cluster

### Performance Optimization

1. **Batch Size Tuning**: Optimize batch sizes for your data
2. **GPU Utilization**: Monitor GPU usage and adjust allocation
3. **Memory Management**: Enable object spilling for large datasets
4. **Network Optimization**: Use high-bandwidth connections for S3

### Troubleshooting

#### Common Production Issues

1. **Out of Memory**
   - Reduce batch size
   - Enable streaming
   - Increase object store memory

2. **Slow Performance**
   - Check GPU utilization
   - Optimize batch sizes
   - Enable GPU acceleration

3. **Ray Connection Issues**
   - Verify Ray cluster is running
   - Check network connectivity
   - Verify Ray address configuration

4. **GPU Errors**
   - Check CUDA installation
   - Verify GPU visibility
   - Check GPU memory usage

