#!/bin/bash
# Deploy to AWS EKS
# Usage: ./scripts/deploy-aws.sh [environment] [region]

set -euo pipefail

ENVIRONMENT="${1:-production}"
REGION="${2:-us-east-1}"
CLUSTER_NAME="${CLUSTER_NAME:-multimodal-pipeline-cluster}"

echo "Deploying to AWS EKS..."
echo "Environment: $ENVIRONMENT"
echo "Region: $REGION"
echo "Cluster: $CLUSTER_NAME"

# Check prerequisites
command -v kubectl >/dev/null 2>&1 || { echo "kubectl required but not installed. Aborting." >&2; exit 1; }
command -v aws >/dev/null 2>&1 || { echo "aws CLI required but not installed. Aborting." >&2; exit 1; }
command -v helm >/dev/null 2>&1 || { echo "helm required but not installed. Aborting." >&2; exit 1; }

# Update kubeconfig
echo "Updating kubeconfig..."
aws eks update-kubeconfig --region "$REGION" --name "$CLUSTER_NAME"

# Verify cluster access
echo "Verifying cluster access..."
kubectl cluster-info || { echo "Failed to access cluster. Aborting." >&2; exit 1; }

# Create namespace if it doesn't exist
kubectl create namespace pipeline-$ENVIRONMENT --dry-run=client -o yaml | kubectl apply -f -

# Apply AWS-specific configurations
echo "Applying AWS-specific configurations..."
kubectl apply -f deployment/aws/cluster/eks-cluster.yaml -n pipeline-$ENVIRONMENT

# Apply Ray cluster (Kuberay)
echo "Applying Ray cluster configuration..."
kubectl apply -f deployment/aws/kuberay/ray-cluster-eks.yaml -n pipeline-$ENVIRONMENT

# Apply node groups (Karpenter)
echo "Applying node group configurations..."
kubectl apply -f deployment/aws/kuberay/node-groups.yaml

# Apply ALB ingress if enabled
if [ "${ENABLE_ALB_INGRESS:-false}" = "true" ]; then
    echo "Applying ALB ingress..."
    kubectl apply -f deployment/aws/networking/alb-ingress.yaml -n pipeline-$ENVIRONMENT
fi

# Apply CloudWatch logging if enabled
if [ "${ENABLE_CLOUDWATCH_LOGS:-true}" = "true" ]; then
    echo "Applying CloudWatch logging..."
    kubectl apply -f deployment/aws/monitoring/cloudwatch-logs.yaml
fi

# Apply HPA
echo "Applying Horizontal Pod Autoscaler..."
kubectl apply -f deployment/aws/autoscaling/hpa-gpu.yaml -n pipeline-$ENVIRONMENT

# Wait for deployments
echo "Waiting for deployments to be ready..."
kubectl wait --for=condition=available --timeout=600s deployment/multimodal-pipeline -n pipeline-$ENVIRONMENT || true

# Show status
echo "Deployment status:"
kubectl get pods -n pipeline-$ENVIRONMENT
kubectl get services -n pipeline-$ENVIRONMENT
kubectl get hpa -n pipeline-$ENVIRONMENT

echo "Deployment complete!"

