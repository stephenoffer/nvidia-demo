#!/bin/bash
# Infrastructure setup script
# Creates necessary Kubernetes resources, secrets, and configurations

set -euo pipefail

ENVIRONMENT="${1:-staging}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

NAMESPACE="pipeline-${ENVIRONMENT}"

log_info "Setting up infrastructure for $ENVIRONMENT environment"

# Check prerequisites
command -v kubectl >/dev/null 2>&1 || { log_error "kubectl is required"; exit 1; }

# Create namespace
log_info "Creating namespace $NAMESPACE"
kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -

# Create secrets if they don't exist
if ! kubectl get secret aws-credentials -n "$NAMESPACE" >/dev/null 2>&1; then
    log_warn "AWS credentials secret not found. Creating placeholder..."
    kubectl create secret generic aws-credentials \
        --from-literal=access-key-id="${AWS_ACCESS_KEY_ID:-placeholder}" \
        --from-literal=secret-access-key="${AWS_SECRET_ACCESS_KEY:-placeholder}" \
        -n "$NAMESPACE" \
        --dry-run=client -o yaml | kubectl apply -f -
fi

# Apply network policy
log_info "Applying network policy"
kubectl apply -f "${PROJECT_ROOT}/deployment/network-policy.yaml" -n "$NAMESPACE" || true

# Apply Prometheus ServiceMonitor
log_info "Applying Prometheus ServiceMonitor"
kubectl apply -f "${PROJECT_ROOT}/deployment/prometheus-service-monitor.yaml" || true

# Apply Prometheus alerts
log_info "Applying Prometheus alerting rules"
kubectl apply -f "${PROJECT_ROOT}/deployment/prometheus-alerts.yaml" || true

log_info "Infrastructure setup completed!"

