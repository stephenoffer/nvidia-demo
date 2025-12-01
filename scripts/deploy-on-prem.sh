#!/bin/bash
# Deploy to On-Premises Kubernetes
# Usage: ./scripts/deploy-on-prem.sh [environment]

set -euo pipefail

ENVIRONMENT="${1:-production}"
KUBECONFIG="${KUBECONFIG:-$HOME/.kube/config}"

echo "Deploying to On-Premises Kubernetes..."
echo "Environment: $ENVIRONMENT"
echo "Kubeconfig: $KUBECONFIG"

# Check prerequisites
command -v kubectl >/dev/null 2>&1 || { echo "kubectl required but not installed. Aborting." >&2; exit 1; }
command -v helm >/dev/null 2>&1 || { echo "helm required but not installed. Aborting." >&2; exit 1; }

# Verify cluster access
echo "Verifying cluster access..."
kubectl cluster-info --kubeconfig="$KUBECONFIG" || { echo "Failed to access cluster. Aborting." >&2; exit 1; }

# Create namespace if it doesn't exist
kubectl create namespace pipeline-$ENVIRONMENT --kubeconfig="$KUBECONFIG" --dry-run=client -o yaml | kubectl apply --kubeconfig="$KUBECONFIG" -f -

# Apply on-prem configurations
echo "Applying on-premises configurations..."
kubectl apply -f deployment/on-prem/kubernetes.yaml --kubeconfig="$KUBECONFIG" -n pipeline-$ENVIRONMENT

# Apply Ray cluster
echo "Applying Ray cluster configuration..."
kubectl apply -f deployment/on-prem/ray-cluster.yaml --kubeconfig="$KUBECONFIG" -n pipeline-$ENVIRONMENT

# Apply NGINX ingress if enabled
if [ "${ENABLE_NGINX_INGRESS:-true}" = "true" ]; then
    echo "Applying NGINX ingress..."
    kubectl apply -f deployment/on-prem/ingress-nginx.yaml --kubeconfig="$KUBECONFIG" -n pipeline-$ENVIRONMENT
fi

# Apply logging aggregation if enabled
if [ "${ENABLE_LOGGING_AGGREGATION:-true}" = "true" ]; then
    echo "Applying logging aggregation..."
    kubectl apply -f deployment/on-prem/logging-aggregation.yaml --kubeconfig="$KUBECONFIG"
fi

# Apply Vault integration if enabled
if [ "${ENABLE_VAULT:-false}" = "true" ]; then
    echo "Applying Vault integration..."
    kubectl apply -f deployment/on-prem/vault-integration.yaml --kubeconfig="$KUBECONFIG" -n pipeline-$ENVIRONMENT
fi

# Apply HPA
echo "Applying Horizontal Pod Autoscaler..."
kubectl apply -f deployment/on-prem/hpa.yaml --kubeconfig="$KUBECONFIG" -n pipeline-$ENVIRONMENT

# Wait for deployments
echo "Waiting for deployments to be ready..."
kubectl wait --for=condition=available --timeout=600s deployment/multimodal-pipeline --kubeconfig="$KUBECONFIG" -n pipeline-$ENVIRONMENT || true

# Show status
echo "Deployment status:"
kubectl get pods --kubeconfig="$KUBECONFIG" -n pipeline-$ENVIRONMENT
kubectl get services --kubeconfig="$KUBECONFIG" -n pipeline-$ENVIRONMENT
kubectl get hpa --kubeconfig="$KUBECONFIG" -n pipeline-$ENVIRONMENT

echo "Deployment complete!"

