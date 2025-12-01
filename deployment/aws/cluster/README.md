# EKS Cluster Configuration

This directory contains the core EKS cluster setup and configuration files.

## Files

- **eks-cluster-setup.sh** - Automated script to create EKS cluster with all required components
- **eks-cluster.yaml** - Main Kubernetes deployment configuration for the pipeline
- **eks-addons.tf** - Terraform configuration for EKS add-ons (EBS CSI, EFS CSI, VPC CNI, etc.)

## Usage

### 1. Create EKS Cluster

```bash
# Using the automated script
./cluster/eks-cluster-setup.sh multimodal-pipeline-cluster us-east-1 production

# Or manually with eksctl
eksctl create cluster --name multimodal-pipeline-cluster --region us-east-1
```

### 2. Install Add-ons

```bash
# Using Terraform
cd cluster
terraform init
terraform apply

# Or using eksctl
eksctl create addon --name aws-ebs-csi-driver --cluster multimodal-pipeline-cluster
```

### 3. Deploy Pipeline

```bash
kubectl apply -f cluster/eks-cluster.yaml
```

## Prerequisites

- AWS CLI configured
- eksctl installed
- kubectl installed
- Terraform installed (for add-ons)
- Appropriate AWS IAM permissions

