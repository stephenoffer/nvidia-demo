# AWS EKS Deployment Guide

Complete guide for deploying the multimodal pipeline on AWS EKS using Kuberay.

## Directory Structure

```
aws/
├── cluster/          # EKS cluster setup and configuration
│   ├── eks-cluster-setup.sh
│   ├── eks-cluster.yaml
│   └── eks-addons.tf
├── kuberay/          # Kuberay/Ray cluster configurations
│   ├── ray-cluster-eks.yaml
│   └── node-groups.yaml
├── networking/       # Networking configurations
│   ├── vpc-endpoints.yaml
│   ├── vpc-endpoints.tf
│   ├── alb-ingress.yaml
│   └── service-mesh-istio.yaml
├── security/         # Security configurations
│   ├── eks-security-hardening.yaml
│   ├── secrets-manager-integration.yaml
│   └── iam-role-policy.json
├── monitoring/       # Monitoring configurations
│   ├── eks-monitoring.yaml
│   ├── cloudwatch-logs.yaml
│   └── cloudwatch-dashboard.json
├── autoscaling/      # Autoscaling configurations
│   └── hpa-gpu.yaml
├── certificates/     # Certificate management
│   └── cert-manager.yaml
└── backup/           # Backup and restore
    ├── backup-restore.yaml
    └── disaster-recovery.md
```

## Quick Start

### 1. Prerequisites

- AWS CLI configured
- eksctl installed
- kubectl installed
- Helm installed
- Terraform installed (optional)

### 2. Create EKS Cluster

```bash
cd cluster
./eks-cluster-setup.sh multimodal-pipeline-cluster us-east-1 production
```

### 3. Configure IRSA

```bash
# Create IAM policy
aws iam create-policy \
  --policy-name multimodal-pipeline-policy \
  --policy-document file://security/iam-role-policy.json

# Create service account with IRSA
eksctl create iamserviceaccount \
  --cluster=multimodal-pipeline-cluster \
  --name=multimodal-pipeline-sa \
  --namespace=pipeline-production \
  --attach-policy-arn=arn:aws:iam::ACCOUNT_ID:policy/multimodal-pipeline-policy \
  --approve
```

### 4. Deploy Pipeline

```bash
# Deploy main pipeline
kubectl apply -f cluster/eks-cluster.yaml

# Deploy Ray cluster (Kuberay)
kubectl apply -f kuberay/ray-cluster-eks.yaml

# Deploy node groups (Karpenter)
kubectl apply -f kuberay/node-groups.yaml
```

### 5. Configure Networking

```bash
# Create VPC endpoints
cd networking
terraform init
terraform apply

# Deploy ALB ingress
kubectl apply -f networking/alb-ingress.yaml
```

### 6. Enable Monitoring

```bash
# Deploy CloudWatch logging
kubectl apply -f monitoring/cloudwatch-logs.yaml

# Deploy Prometheus monitoring
kubectl apply -f monitoring/eks-monitoring.yaml
```

## Deployment Order

1. **Cluster Setup** (`cluster/`)
   - Create EKS cluster
   - Install add-ons
   - Configure storage

2. **Security** (`security/`)
   - Configure IRSA
   - Apply security hardening
   - Set up secrets management

3. **Kuberay** (`kuberay/`)
   - Deploy Ray cluster
   - Configure node groups

4. **Networking** (`networking/`)
   - Create VPC endpoints
   - Configure ALB ingress
   - Set up service mesh (optional)

5. **Monitoring** (`monitoring/`)
   - Deploy logging
   - Configure metrics
   - Set up dashboards

6. **Autoscaling** (`autoscaling/`)
   - Configure HPA
   - Set up VPA (optional)

7. **Certificates** (`certificates/`)
   - Request ACM certificates
   - Configure cert-manager (optional)

8. **Backup** (`backup/`)
   - Install Velero
   - Configure backup schedules

## Features

### Kuberay Integration

- Native Kubernetes operator for Ray
- Automatic worker scaling
- GPU and CPU worker groups
- Spot instance support for cost optimization

### IRSA (IAM Roles for Service Accounts)

No need to store AWS credentials in Kubernetes secrets. Service accounts automatically assume IAM roles.

### Storage Options

- **EBS GP3**: Fast SSD for checkpoints
- **EFS**: Shared storage for distributed workloads
- **S3**: Object storage for input/output data

### Monitoring

- **CloudWatch Logs**: Centralized logging
- **Prometheus**: Metrics collection
- **CloudWatch Dashboards**: AWS-native visualization

### Autoscaling

- **HPA**: Horizontal Pod Autoscaler with GPU metrics
- **Ray Autoscaling**: Automatic Ray worker scaling via Kuberay
- **Karpenter**: Node autoscaling

## Configuration

### Update Configuration Values

Before deploying, update these values:

1. **ACCOUNT_ID**: Replace in all files with your AWS account ID
2. **ECR_REGISTRY**: Update image registry in `cluster/eks-cluster.yaml`
3. **S3 Buckets**: Update bucket names in ConfigMaps
4. **Certificate ARNs**: Update in `networking/alb-ingress.yaml`
5. **VPC/Subnet IDs**: Update in Terraform files
6. **Hostnames**: Update in ingress configurations

## Troubleshooting

### Cluster Access Issues

```bash
# Update kubeconfig
aws eks update-kubeconfig --region us-east-1 --name multimodal-pipeline-cluster

# Verify access
kubectl cluster-info
```

### Ray Cluster Not Starting

```bash
# Check Ray cluster status
kubectl get raycluster -n pipeline-production

# Check Ray head pod
kubectl logs -n pipeline-production -l ray.io/node-type=head

# Check Kuberay operator logs
kubectl logs -n ray-system -l app.kubernetes.io/name=kuberay-operator
```

### IRSA Issues

```bash
# Verify service account annotation
kubectl get sa multimodal-pipeline-sa -n pipeline-production -o yaml

# Test IAM role
kubectl run -it --rm test-irsa \
  --image=amazon/aws-cli \
  --serviceaccount=multimodal-pipeline-sa \
  --restart=Never \
  -- aws sts get-caller-identity
```

### Storage Issues

```bash
# Check PVC status
kubectl get pvc -n pipeline-production

# Check storage classes
kubectl get storageclass

# Check EBS volumes
aws ec2 describe-volumes --filters "Name=tag:kubernetes.io/cluster/multimodal-pipeline-cluster,Values=owned"
```

## Cost Optimization

- Use Spot Instances for Ray CPU workers
- Enable EBS volume encryption
- Use S3 Intelligent-Tiering for data
- Configure CloudWatch log retention
- Use GP3 instead of GP2 for better price/performance
- Use Karpenter for efficient node autoscaling

## Security Best Practices

1. Use IRSA instead of access keys
2. Enable S3 bucket encryption
3. Use network policies to restrict traffic
4. Enable pod security standards
5. Rotate secrets regularly
6. Use AWS Secrets Manager for sensitive data
7. Enable VPC endpoints to reduce data transfer costs
8. Use mTLS with Istio (optional)

## Additional Resources

- [Kuberay Documentation](https://ray-project.github.io/kuberay/)
- [AWS EKS Best Practices](https://aws.github.io/aws-eks-best-practices/)
- [Karpenter Documentation](https://karpenter.sh/)
- [Velero Documentation](https://velero.io/docs/)

## Support

For issues or questions:
1. Check the README in each subdirectory
2. Review troubleshooting sections
3. Check AWS EKS documentation
4. Review Kuberay GitHub issues
