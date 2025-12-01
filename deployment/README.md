# Deployment Guide

This directory contains deployment configurations for AWS EKS and on-premises Kubernetes clusters.

## Directory Structure

```
deployment/
├── aws/                    # AWS EKS specific configurations
│   ├── cluster/            # EKS cluster setup
│   │   ├── eks-cluster-setup.sh
│   │   ├── eks-cluster.yaml
│   │   └── eks-addons.tf
│   ├── kuberay/            # Kuberay/Ray cluster configurations
│   │   ├── ray-cluster-eks.yaml
│   │   └── node-groups.yaml
│   ├── networking/         # Networking configurations
│   │   ├── vpc-endpoints.yaml
│   │   ├── vpc-endpoints.tf
│   │   ├── alb-ingress.yaml
│   │   └── service-mesh-istio.yaml
│   ├── security/           # Security configurations
│   │   ├── eks-security-hardening.yaml
│   │   ├── secrets-manager-integration.yaml
│   │   └── iam-role-policy.json
│   ├── monitoring/         # Monitoring configurations
│   │   ├── eks-monitoring.yaml
│   │   ├── cloudwatch-logs.yaml
│   │   └── cloudwatch-dashboard.json
│   ├── autoscaling/        # Autoscaling configurations
│   │   └── hpa-gpu.yaml
│   ├── certificates/       # Certificate management
│   │   └── cert-manager.yaml
│   └── backup/             # Backup and restore
│       ├── backup-restore.yaml
│       └── disaster-recovery.md
├── on-prem/               # On-premises Kubernetes configurations
│   ├── kubernetes.yaml    # Main on-prem deployment
│   ├── ray-cluster.yaml   # Ray cluster for on-prem
│   ├── ingress-nginx.yaml # NGINX ingress
│   ├── logging-aggregation.yaml  # Logging aggregation (Fluentd)
│   ├── vault-integration.yaml  # HashiCorp Vault integration
│   └── hpa.yaml          # Standard HPA
├── helm/                  # Helm charts
├── kustomize/             # Kustomize overlays
└── [shared configs]       # Shared configurations
```

## AWS EKS Deployment

### Prerequisites

1. AWS CLI configured with appropriate credentials
2. kubectl installed
3. EKS cluster created and accessible
4. AWS Load Balancer Controller installed (for ALB ingress)
5. Ray Operator installed
6. External Secrets Operator installed (optional, for Secrets Manager)

### Quick Start

```bash
# Set environment variables
export CLUSTER_NAME=your-eks-cluster
export AWS_REGION=us-east-1

# Deploy
./scripts/deploy-aws.sh production us-east-1
```

### Features

- **IRSA (IAM Roles for Service Accounts)**: No need for AWS credentials in secrets
- **EBS Storage**: GP3 SSD storage class for checkpoints
- **EFS Storage**: Shared storage for distributed workloads
- **CloudWatch Logs**: Centralized logging
- **ALB Ingress**: Application Load Balancer for external access
- **Secrets Manager**: Secure credential management
- **GPU Autoscaling**: GPU-aware HPA with custom metrics

### Configuration

1. Update `deployment/aws/cluster/eks-cluster.yaml`:
   - Replace `ACCOUNT_ID` with your AWS account ID
   - Update ECR registry URL
   - Configure S3 bucket names
   - Set EFS filesystem ID (if using EFS)

2. Create IAM role and policy:
   ```bash
   aws iam create-role --role-name multimodal-pipeline-role \
     --assume-role-policy-document file://deployment/aws/security/iam-role-policy.json
   ```

3. Configure IRSA:
   ```bash
   eksctl create iamserviceaccount \
     --cluster=your-cluster \
     --name=multimodal-pipeline-sa \
     --namespace=pipeline-production \
     --role-name=multimodal-pipeline-role \
     --attach-policy-arn=arn:aws:iam::ACCOUNT:policy/multimodal-pipeline-policy
   ```

## On-Premises Kubernetes Deployment

### Prerequisites

1. Self-managed Kubernetes cluster (v1.24+)
2. kubectl configured with cluster access
3. NGINX Ingress Controller (optional)
4. Ray Operator installed
5. Storage provisioner (NFS, CephFS, or local storage)
6. Logging aggregation (Elasticsearch, Loki, etc.) - optional
7. Vault (optional, for secrets management)

### Quick Start

```bash
# Set kubeconfig if needed
export KUBECONFIG=/path/to/kubeconfig

# Deploy
./scripts/deploy-on-prem.sh production
```

### Features

- **Local Storage**: HostPath volumes for data
- **NFS/CephFS**: Shared storage support
- **NGINX Ingress**: Ingress controller for external access
- **Logging Aggregation**: Fluentd to Elasticsearch/Loki
- **Vault Integration**: Secure secrets management
- **Standard HPA**: CPU/memory based autoscaling

### Configuration

1. Update `deployment/on-prem/kubernetes.yaml`:
   - Update registry URL
   - Configure storage paths
   - Set node selectors and tolerations
   - Configure NFS server (if using)

2. Label GPU nodes:
   ```bash
   kubectl label nodes <node-name> accelerator=nvidia-gpu
   kubectl label nodes <node-name> node-type=gpu-worker
   ```

3. Configure storage:
   - For local storage: Create directories on nodes
   - For NFS: Update NFS server address
   - For CephFS: Configure Ceph cluster

## Helm Deployment

Both AWS and on-prem deployments can use Helm charts:

```bash
# AWS
helm install pipeline deployment/helm/ \
  --namespace pipeline-production \
  --create-namespace \
  --set image.repository=ECR_REGISTRY/multimodal-pipeline \
  --set aws.enabled=true

# On-prem
helm install pipeline deployment/helm/ \
  --namespace pipeline-production \
  --create-namespace \
  --set image.repository=registry.example.com/multimodal-pipeline \
  --set aws.enabled=false
```

## Kustomize Deployment

Use Kustomize for environment-specific overlays:

```bash
# Production
kubectl apply -k deployment/kustomize/production/

# Staging
kubectl apply -k deployment/kustomize/staging/
```

## Monitoring

Both deployments include Prometheus metrics and Grafana dashboards:

- Prometheus ServiceMonitor configured
- Grafana datasource configured
- Custom metrics for GPU utilization
- Health check endpoints

## Troubleshooting

### AWS EKS

- **IRSA not working**: Verify service account annotation and IAM role trust policy
- **S3 access denied**: Check IAM policy permissions
- **ALB not created**: Verify AWS Load Balancer Controller is installed
- **CloudWatch logs missing**: Check Fluent Bit DaemonSet and IAM permissions

### On-Premises

- **Storage issues**: Verify storage classes and PV/PVC status
- **Ingress not working**: Check NGINX Ingress Controller installation
- **Logs not aggregated**: Verify Fluentd DaemonSet and Elasticsearch connectivity
- **GPU not available**: Check node labels and device plugin installation

## Security

- **Secrets**: Use IRSA (AWS) or Vault (on-prem) instead of hardcoded credentials
- **Network Policies**: Applied to restrict pod-to-pod communication
- **Pod Security Standards**: Enforced via admission controllers
- **RBAC**: Service accounts with minimal required permissions

## Backup and Disaster Recovery

- **Checkpoints**: Stored in persistent volumes
- **ConfigMaps**: Version controlled in Git
- **Secrets**: Managed by external secret managers
- **State**: Ray cluster state can be checkpointed

## Scaling

- **Horizontal**: HPA configured for both CPU and GPU metrics
- **Vertical**: Resource requests/limits configured
- **Ray**: Autoscaling enabled for Ray workers

