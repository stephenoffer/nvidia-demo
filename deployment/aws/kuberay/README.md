# Kuberay Configuration

This directory contains Kuberay (Kubernetes Ray Operator) configurations for running Ray clusters on EKS.

## Files

- **ray-cluster-eks.yaml** - Kuberay RayCluster resource configuration
- **node-groups.yaml** - Karpenter NodePool configurations for Ray worker nodes

## Overview

Kuberay is the Kubernetes operator for Ray, providing native Kubernetes integration for Ray clusters. This configuration deploys:

- **Ray Head Node**: Manages the Ray cluster and provides the dashboard
- **GPU Worker Nodes**: For GPU-intensive workloads
- **CPU Worker Nodes**: For CPU-intensive workloads (using Spot instances)

## Prerequisites

- EKS cluster created (see `../cluster/`)
- Kuberay operator installed
- Karpenter installed (for node autoscaling)
- NVIDIA GPU device plugin installed (for GPU nodes)

## Installation

### 1. Install Kuberay Operator

```bash
# Using Helm (recommended)
helm repo add kuberay https://ray-project.github.io/kuberay-helm/
helm install kuberay-operator kuberay/kuberay-operator --namespace ray-system --create-namespace

# Or using kubectl
kubectl apply -f https://raw.githubusercontent.com/ray-project/kuberay/master/helm-chart/kuberay-operator/crds/cluster.yaml
```

### 2. Deploy Ray Cluster

```bash
kubectl apply -f kuberay/ray-cluster-eks.yaml
```

### 3. Configure Node Groups (Karpenter)

```bash
kubectl apply -f kuberay/node-groups.yaml
```

## Verification

```bash
# Check Ray cluster status
kubectl get raycluster -n pipeline-production

# Check Ray head pod
kubectl get pods -n pipeline-production -l ray.io/node-type=head

# Check Ray worker pods
kubectl get pods -n pipeline-production -l ray.io/node-type=worker

# Access Ray dashboard
kubectl port-forward -n pipeline-production svc/ray-head-service 8265:8265
# Then open http://localhost:8265
```

## Configuration

### Ray Cluster Settings

Edit `ray-cluster-eks.yaml` to customize:
- Ray version
- Resource requests/limits
- Autoscaling parameters
- Environment variables

### Node Groups

Edit `node-groups.yaml` to customize:
- Instance types
- Capacity types (on-demand vs spot)
- Node limits
- Disruption policies

## Troubleshooting

### Ray Head Not Starting

```bash
# Check logs
kubectl logs -n pipeline-production -l ray.io/node-type=head

# Check events
kubectl describe raycluster multimodal-pipeline-cluster -n pipeline-production
```

### Workers Not Scaling

```bash
# Check Karpenter logs
kubectl logs -n karpenter -l app.kubernetes.io/name=karpenter

# Check node pool status
kubectl get nodepool -n karpenter
```

### GPU Not Available

```bash
# Verify GPU device plugin
kubectl get daemonset -n kube-system | grep nvidia

# Check node labels
kubectl get nodes --show-labels | grep gpu
```

