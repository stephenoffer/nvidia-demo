#!/bin/bash
# Production deployment script for Kubernetes
# Usage: ./scripts/deploy.sh [environment] [version]

set -euo pipefail

ENVIRONMENT="${1:-staging}"
VERSION="${2:-latest}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Validate environment
if [[ ! "$ENVIRONMENT" =~ ^(staging|production)$ ]]; then
    log_error "Invalid environment: $ENVIRONMENT. Must be 'staging' or 'production'"
    exit 1
fi

log_info "Deploying to $ENVIRONMENT environment with version $VERSION"

# Check prerequisites
command -v kubectl >/dev/null 2>&1 || { log_error "kubectl is required but not installed"; exit 1; }
command -v docker >/dev/null 2>&1 || { log_error "docker is required but not installed"; exit 1; }

# Set namespace
NAMESPACE="pipeline-${ENVIRONMENT}"

# Check if namespace exists
if ! kubectl get namespace "$NAMESPACE" >/dev/null 2>&1; then
    log_warn "Namespace $NAMESPACE does not exist. Creating..."
    kubectl create namespace "$NAMESPACE"
fi

# Set kubectl context
kubectl config set-context --current --namespace="$NAMESPACE"

# Build and push Docker image if needed
if [[ "$VERSION" != "latest" ]]; then
    log_info "Building Docker image with tag $VERSION"
    docker build -t multimodal-pipeline:"$VERSION" "$PROJECT_ROOT"
    
    # Push to registry if REGISTRY_URL is set
    if [[ -n "${REGISTRY_URL:-}" ]]; then
        log_info "Pushing image to registry"
        docker tag multimodal-pipeline:"$VERSION" "${REGISTRY_URL}/multimodal-pipeline:${VERSION}"
        docker push "${REGISTRY_URL}/multimodal-pipeline:${VERSION}"
    fi
fi

# Apply Kubernetes manifests
log_info "Applying Kubernetes manifests"

# Apply base configuration
kubectl apply -f "${PROJECT_ROOT}/deployment/kubernetes-production.yaml"

# Update image version if specified
if [[ "$VERSION" != "latest" ]]; then
    kubectl set image deployment/multimodal-pipeline \
        pipeline=multimodal-pipeline:"$VERSION" \
        -n "$NAMESPACE"
fi

# Wait for rollout
log_info "Waiting for deployment rollout..."
kubectl rollout status deployment/multimodal-pipeline -n "$NAMESPACE" --timeout=10m

# Verify deployment
log_info "Verifying deployment..."
kubectl get pods -n "$NAMESPACE" -l app=multimodal-pipeline

# Run smoke tests
log_info "Running smoke tests..."
"${SCRIPT_DIR}/smoke-tests.sh" "$ENVIRONMENT"

log_info "Deployment completed successfully!"

