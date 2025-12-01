# Troubleshooting Guide

## Quick Diagnostics

### Health Check Commands

```bash
# Check overall health
pipeline health

# Check Ray cluster
python -c "from pipeline.health import HealthChecker; print(HealthChecker.check_ray_cluster())"

# Check GPU resources
python -c "from pipeline.health import HealthChecker; print(HealthChecker.check_gpu_resources())"
```

### Common Error Messages

#### "Ray cluster not initialized"
**Cause**: Ray not started or connection failed
**Fix**: 
```python
from pipeline.utils.ray.init import initialize_ray
initialize_ray()
```

#### "CUDA not available"
**Cause**: GPU not accessible or CUDA not installed
**Fix**: Check GPU availability, verify CUDA installation

#### "Out of memory"
**Cause**: Insufficient memory for operation
**Fix**: Reduce batch size, enable streaming, increase memory limits

#### "Checkpoint not found"
**Cause**: Checkpoint directory doesn't exist or checkpoint deleted
**Fix**: Verify checkpoint path, check disk space

## Debug Mode

Enable debug logging:
```bash
export LOG_LEVEL=DEBUG
pipeline health
```

## Performance Issues

### Slow Processing
1. Check GPU utilization: `nvidia-smi`
2. Check Ray cluster resources: Ray dashboard
3. Review batch sizes
4. Check network latency for cloud storage

### High Memory Usage
1. Enable streaming mode
2. Reduce batch sizes
3. Check for memory leaks
4. Review checkpoint intervals

## Network Issues

### S3 Connection Failures
```bash
# Test S3 connectivity
aws s3 ls s3://bucket/

# Check credentials
kubectl get secret aws-credentials -n pipeline-production -o jsonpath='{.data}'
```

### Ray Cluster Connection
```bash
# Check Ray address
echo $RAY_ADDRESS

# Test connection
python -c "import ray; ray.init(address='ray://ray-head:10001')"
```

