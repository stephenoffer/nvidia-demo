# Operations Runbook

## Table of Contents
1. [Common Issues](#common-issues)
2. [Troubleshooting](#troubleshooting)
3. [Recovery Procedures](#recovery-procedures)
4. [Maintenance Tasks](#maintenance-tasks)

## Common Issues

### Pipeline Not Starting

**Symptoms:**
- Pods in CrashLoopBackOff
- Health checks failing
- Ray cluster not initializing

**Diagnosis:**
```bash
# Check pod logs
kubectl logs -n pipeline-production deployment/multimodal-pipeline

# Check Ray cluster status
kubectl exec -n pipeline-production deployment/multimodal-pipeline -- python -c "import ray; print(ray.is_initialized())"

# Check health endpoint
curl http://pipeline-service:8080/health
```

**Resolution:**
1. Check GPU availability: `kubectl get nodes -l accelerator=nvidia-gpu`
2. Verify secrets: `kubectl get secret aws-credentials -n pipeline-production`
3. Check resource quotas: `kubectl describe resourcequota -n pipeline-production`
4. Review configuration: `kubectl get configmap multimodal-pipeline-config -n pipeline-production -o yaml`

### High Memory Usage

**Symptoms:**
- Pods being OOMKilled
- High memory utilization metrics
- Slow processing

**Diagnosis:**
```bash
# Check memory usage
kubectl top pods -n pipeline-production

# Check Ray object store
kubectl exec -n pipeline-production deployment/multimodal-pipeline -- python -c "import ray; print(ray.available_resources())"
```

**Resolution:**
1. Reduce batch size in config
2. Enable streaming mode
3. Increase memory limits
4. Check for memory leaks in logs

### GPU Errors

**Symptoms:**
- CUDA errors in logs
- GPU utilization at 0%
- NCCL errors

**Diagnosis:**
```bash
# Check GPU status
kubectl exec -n pipeline-production deployment/multimodal-pipeline -- nvidia-smi

# Check CUDA availability
kubectl exec -n pipeline-production deployment/multimodal-pipeline -- python -c "import torch; print(torch.cuda.is_available())"
```

**Resolution:**
1. Verify GPU node labels: `kubectl get nodes -l accelerator=nvidia-gpu`
2. Check GPU driver: `kubectl describe node <gpu-node>`
3. Review NCCL configuration
4. Check GPU memory: `nvidia-smi`

## Troubleshooting

### Check Pipeline Health

```bash
# Via CLI
pipeline health --output-path s3://bucket/output/

# Via HTTP
curl http://pipeline-service:8080/health | jq

# Via Kubernetes
kubectl exec -n pipeline-production deployment/multimodal-pipeline -- curl localhost:8080/health
```

### View Logs

```bash
# Recent logs
kubectl logs -n pipeline-production deployment/multimodal-pipeline --tail=100

# Follow logs
kubectl logs -n pipeline-production deployment/multimodal-pipeline -f

# Logs from all pods
kubectl logs -n pipeline-production -l app=multimodal-pipeline --tail=100
```

### Check Metrics

```bash
# Prometheus metrics
curl http://pipeline-service:9090/metrics

# Ray dashboard
kubectl port-forward -n pipeline-production service/ray-head-service 8265:8265
# Then open http://localhost:8265
```

## Recovery Procedures

### Restart Pipeline

```bash
# Rolling restart
kubectl rollout restart deployment/multimodal-pipeline -n pipeline-production

# Force restart (if needed)
kubectl delete pod -n pipeline-production -l app=multimodal-pipeline
```

### Recover from Checkpoint

```bash
# List available checkpoints
kubectl exec -n pipeline-production deployment/multimodal-pipeline -- \
  python -c "from pipeline.utils.checkpoint_recovery import CheckpointRecovery; \
  cr = CheckpointRecovery('.checkpoints'); print(cr.list_checkpoints())"

# Resume from checkpoint
# Update config to specify checkpoint name
kubectl set env deployment/multimodal-pipeline \
  RESUME_FROM_CHECKPOINT=checkpoint_000123 \
  -n pipeline-production
```

### Scale Down/Up

```bash
# Scale down
kubectl scale deployment multimodal-pipeline --replicas=0 -n pipeline-production

# Scale up
kubectl scale deployment multimodal-pipeline --replicas=3 -n pipeline-production

# Auto-scaling via HPA
kubectl get hpa -n pipeline-production
```

## Maintenance Tasks

### Backup Configuration

```bash
# Backup all configs
kubectl get configmap -n pipeline-production -o yaml > config-backup.yaml
kubectl get secret -n pipeline-production -o yaml > secrets-backup.yaml
```

### Update Image

```bash
# Update image tag
kubectl set image deployment/multimodal-pipeline \
  pipeline=multimodal-pipeline:v0.2.0 \
  -n pipeline-production

# Monitor rollout
kubectl rollout status deployment/multimodal-pipeline -n pipeline-production
```

### Cleanup Old Checkpoints

```bash
# Cleanup via Python
kubectl exec -n pipeline-production deployment/multimodal-pipeline -- \
  python -c "from pipeline.utils.checkpoint import PipelineCheckpoint; \
  pc = PipelineCheckpoint('.checkpoints', max_checkpoints=5); \
  pc._cleanup_old_checkpoints()"
```

