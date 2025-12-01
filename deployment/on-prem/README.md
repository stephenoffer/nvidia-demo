# On-Premises Kubernetes Deployment Guide

Complete guide for deploying the multimodal pipeline on self-managed Kubernetes clusters.

## Prerequisites

1. **Kubernetes Cluster** (1.24+) with:
   - GPU nodes with NVIDIA device plugin
   - Storage provisioner (NFS, CephFS, or local)
   - NGINX Ingress Controller (optional)
   - Ray Operator
   - Metrics Server (for HPA)
2. **Storage**:
   - Local SSD for checkpoints
   - NFS/CephFS for shared storage
   - Sufficient capacity (1TB+ recommended)
3. **Network**:
   - Internal network access
   - Optional: External ingress

## Quick Start

```bash
# 1. Configure kubeconfig
export KUBECONFIG=/path/to/kubeconfig

# 2. Verify cluster access
kubectl cluster-info

# 3. Deploy
./scripts/deploy-on-prem.sh production
```

## Configuration Steps

### 1. Prepare GPU Nodes

```bash
# Label GPU nodes
kubectl label nodes <node-name> accelerator=nvidia-gpu
kubectl label nodes <node-name> node-type=gpu-worker

# Add taints (optional)
kubectl taint nodes <node-name> dedicated=pipeline:NoSchedule
```

### 2. Configure Storage

#### Option A: Local Storage

```bash
# Create directories on nodes
sudo mkdir -p /mnt/ssd/pipeline-checkpoints
sudo mkdir -p /mnt/pipeline/input
sudo mkdir -p /mnt/pipeline/output
sudo mkdir -p /var/log/pipeline

# Set permissions
sudo chown -R 1000:1000 /mnt/ssd/pipeline-checkpoints
sudo chown -R 1000:1000 /mnt/pipeline
sudo chown -R 1000:1000 /var/log/pipeline
```

#### Option B: NFS

```bash
# Install NFS provisioner
helm repo add nfs-subdir-external-provisioner https://kubernetes-sigs.github.io/nfs-subdir-external-provisioner/
helm install nfs-provisioner nfs-subdir-external-provisioner/nfs-subdir-external-provisioner \
  --set nfs.server=nfs-server.example.com \
  --set nfs.path=/exports
```

#### Option C: CephFS

```bash
# Install Ceph CSI
kubectl apply -f https://raw.githubusercontent.com/ceph/ceph-csi/master/deploy/cephfs/kubernetes/csi-cephfsplugin.yaml
```

### 3. Install Ray Operator

```bash
kubectl create -f https://raw.githubusercontent.com/ray-project/ray-operator/master/deploy/ray-operator.yaml
```

### 4. Install NGINX Ingress (Optional)

```bash
helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
helm install ingress-nginx ingress-nginx/ingress-nginx \
  --namespace ingress-nginx \
  --create-namespace
```

### 5. Update Configuration Files

Edit `deployment/on-prem/kubernetes.yaml`:
- Update registry URL
- Configure storage paths
- Set node selectors
- Configure NFS server (if using)

### 6. Deploy

```bash
./scripts/deploy-on-prem.sh production
```

## Features

### Storage Options

- **Local Storage**: HostPath volumes for high performance
- **NFS**: Shared storage for distributed workloads
- **CephFS**: Distributed filesystem

### Monitoring

- **Prometheus**: Metrics collection
- **Grafana**: Dashboards
- **Logging Aggregation**: Fluentd to Elasticsearch/Loki

### Autoscaling

- **HPA**: CPU/memory based autoscaling
- **Ray Autoscaling**: Automatic Ray worker scaling

## Troubleshooting

### Storage Issues

```bash
# Check PVC status
kubectl get pvc -n pipeline-production

# Check PV status
kubectl get pv

# Check storage class
kubectl get storageclass

# Check pod events
kubectl describe pod <pod-name> -n pipeline-production
```

### GPU Issues

```bash
# Verify device plugin
kubectl get daemonset -n kube-system | grep nvidia

# Check node resources
kubectl describe node <gpu-node> | grep nvidia.com/gpu

# Test GPU access
kubectl run gpu-test --image=nvidia/cuda:11.0-base --rm -it --restart=Never -- \
  nvidia-smi
```

### Network Issues

```bash
# Check ingress controller
kubectl get pods -n ingress-nginx

# Check ingress status
kubectl get ingress -n pipeline-production

# Test service connectivity
kubectl run test-pod --image=busybox --rm -it --restart=Never -- \
  wget -O- http://multimodal-pipeline-service:8080/health
```

## Security Best Practices

1. Use Vault for secrets management
2. Enable network policies
3. Use pod security standards
4. Restrict hostPath volumes
5. Use read-only root filesystems
6. Enable audit logging

## Performance Tuning

### Storage

- Use local SSD for checkpoints
- Use NFS/CephFS for shared data
- Configure appropriate storage classes

### Networking

- Use hostNetwork for high-throughput workloads (if needed)
- Configure pod anti-affinity for distribution
- Use node selectors for GPU workloads

### Resource Limits

- Set appropriate CPU/memory limits
- Configure GPU resource requests
- Use resource quotas per namespace

## Backup and Recovery

### Checkpoints

Checkpoints are stored in persistent volumes. Backup strategy:

```bash
# Backup checkpoint PVC
kubectl get pvc pipeline-checkpoints -n pipeline-production -o yaml > checkpoint-backup.yaml

# Restore from backup
kubectl apply -f checkpoint-backup.yaml
```

### Configuration

All configurations are in Git. Use version control for:
- ConfigMaps
- Deployments
- Services
- Ingress

## Maintenance

### Updates

```bash
# Rolling update
kubectl set image deployment/multimodal-pipeline \
  pipeline=registry.example.com/multimodal-pipeline:v2.0.0 \
  -n pipeline-production

# Monitor rollout
kubectl rollout status deployment/multimodal-pipeline -n pipeline-production
```

### Scaling

```bash
# Manual scaling
kubectl scale deployment multimodal-pipeline --replicas=5 -n pipeline-production

# HPA will handle automatic scaling based on metrics
```

