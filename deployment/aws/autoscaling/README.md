# Autoscaling Configuration

This directory contains autoscaling configurations for GPU workloads on AWS EKS.

## Files

- **hpa-gpu.yaml** - Horizontal Pod Autoscaler with GPU metrics support

## Autoscaling Components

### Horizontal Pod Autoscaler (HPA)

Scales pods based on:
- CPU utilization (target: 70%)
- Memory utilization (target: 80%)
- GPU utilization (target: 70%)
- Custom metrics (request rate, queue depth)

### Vertical Pod Autoscaler (VPA)

Automatically adjusts resource requests/limits based on historical usage.

### Ray Autoscaling

Ray cluster autoscaling is configured in the RayCluster resource (see `../kuberay/`).

## Prerequisites

- Metrics Server installed
- Prometheus Adapter installed (for custom metrics)
- NVIDIA GPU metrics exporter (for GPU metrics)

## Deployment

```bash
# Deploy HPA
kubectl apply -f autoscaling/hpa-gpu.yaml
```

## Configuration

### HPA Settings

- **Min Replicas**: 3
- **Max Replicas**: 20
- **Scale Down**: 50% per minute, min 2 pods
- **Scale Up**: 100% per 30 seconds, max 4 pods

### Metrics

The HPA monitors:
- Resource metrics (CPU, memory)
- Custom pod metrics (GPU utilization, request rate, queue depth)

## Custom Metrics

To enable custom metrics, install Prometheus Adapter:

```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install prometheus-adapter prometheus-community/prometheus-adapter \
  --set prometheus.url=http://prometheus.monitoring.svc.cluster.local
```

## GPU Metrics

For GPU metrics, deploy NVIDIA GPU metrics exporter:

```bash
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/master/nvidia-device-plugin.yml
```

## Verification

```bash
# Check HPA status
kubectl get hpa multimodal-pipeline-hpa -n pipeline-production

# Describe HPA for detailed metrics
kubectl describe hpa multimodal-pipeline-hpa -n pipeline-production

# Check VPA status
kubectl get vpa multimodal-pipeline-vpa -n pipeline-production
```

## Troubleshooting

### HPA Not Scaling

```bash
# Check HPA events
kubectl describe hpa multimodal-pipeline-hpa -n pipeline-production

# Check metrics availability
kubectl get --raw /apis/metrics.k8s.io/v1beta1/namespaces/pipeline-production/pods

# Check custom metrics
kubectl get --raw /apis/custom.metrics.k8s.io/v1beta1/namespaces/pipeline-production/pods/*/gpu_utilization_percent
```

### Metrics Not Available

```bash
# Check Metrics Server
kubectl get deployment metrics-server -n kube-system

# Check Prometheus Adapter
kubectl get deployment prometheus-adapter -n monitoring

# Test metrics API
kubectl top pods -n pipeline-production
```

### Scaling Too Aggressive

Adjust HPA behavior in `hpa-gpu.yaml`:
- Increase `stabilizationWindowSeconds` for scale down
- Decrease scale up policies
- Adjust metric targets

