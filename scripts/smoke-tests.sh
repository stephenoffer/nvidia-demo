#!/bin/bash
# Smoke tests for pipeline deployment
# Usage: ./scripts/smoke-tests.sh [environment]

set -euo pipefail

ENVIRONMENT="${1:-staging}"
NAMESPACE="pipeline-${ENVIRONMENT}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Get service URL
SERVICE_NAME="multimodal-pipeline-service"
SERVICE_PORT="8080"

# Check if pods are running
log_info "Checking pod status..."
PODS=$(kubectl get pods -n "$NAMESPACE" -l app=multimodal-pipeline --field-selector=status.phase=Running -o jsonpath='{.items[*].metadata.name}')

if [[ -z "$PODS" ]]; then
    log_error "No running pods found"
    exit 1
fi

log_info "Found running pods: $PODS"

# Port forward to access service
log_info "Setting up port forwarding..."
kubectl port-forward -n "$NAMESPACE" service/$SERVICE_NAME 8080:$SERVICE_PORT >/dev/null 2>&1 &
PF_PID=$!

# Wait for port forward to be ready
sleep 5

# Cleanup function
cleanup() {
    kill $PF_PID 2>/dev/null || true
}
trap cleanup EXIT

# Test health endpoint
log_info "Testing /health endpoint..."
HEALTH_RESPONSE=$(curl -s http://localhost:8080/health || echo "FAILED")

if [[ "$HEALTH_RESPONSE" == "FAILED" ]]; then
    log_error "Health check failed"
    exit 1
fi

# Check if health response indicates healthy
if echo "$HEALTH_RESPONSE" | grep -q '"healthy":true'; then
    log_info "Health check passed"
else
    log_error "Health check indicates unhealthy status"
    echo "$HEALTH_RESPONSE"
    exit 1
fi

# Test readiness endpoint
log_info "Testing /ready endpoint..."
READY_RESPONSE=$(curl -s http://localhost:8080/ready || echo "FAILED")

if [[ "$READY_RESPONSE" == "FAILED" ]]; then
    log_error "Readiness check failed"
    exit 1
fi

if echo "$READY_RESPONSE" | grep -q '"ready":true'; then
    log_info "Readiness check passed"
else
    log_error "Readiness check indicates not ready"
    echo "$READY_RESPONSE"
    exit 1
fi

# Test liveness endpoint
log_info "Testing /live endpoint..."
LIVE_RESPONSE=$(curl -s http://localhost:8080/live || echo "FAILED")

if [[ "$LIVE_RESPONSE" == "FAILED" ]]; then
    log_error "Liveness check failed"
    exit 1
fi

log_info "All smoke tests passed!"

