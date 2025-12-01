#!/bin/bash
# Restore Kubernetes Resources from Backup
# Usage: ./scripts/restore-k8s.sh <backup-file.tar.gz> [namespace]

set -euo pipefail

BACKUP_FILE="${1:-}"
NAMESPACE="${2:-pipeline-production}"

if [ -z "$BACKUP_FILE" ]; then
    echo "Usage: $0 <backup-file.tar.gz> [namespace]"
    exit 1
fi

if [ ! -f "$BACKUP_FILE" ]; then
    echo "Error: Backup file not found: $BACKUP_FILE"
    exit 1
fi

echo "Restoring Kubernetes resources..."
echo "Backup file: $BACKUP_FILE"
echo "Namespace: $NAMESPACE"

# Extract backup
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

tar -xzf "$BACKUP_FILE" -C "$TEMP_DIR"
BACKUP_DIR=$(find "$TEMP_DIR" -type d -mindepth 1 | head -1)

# Create namespace if it doesn't exist
kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -

# Restore resources (in order)
echo "Restoring ConfigMaps..."
kubectl apply -f "$BACKUP_DIR/configmaps.yaml" -n "$NAMESPACE" || true

echo "Restoring Secrets..."
kubectl apply -f "$BACKUP_DIR/secrets.yaml" -n "$NAMESPACE" || true

echo "Restoring PVCs..."
kubectl apply -f "$BACKUP_DIR/pvcs.yaml" -n "$NAMESPACE" || true

echo "Restoring Deployments..."
kubectl apply -f "$BACKUP_DIR/deployments.yaml" -n "$NAMESPACE" || true

echo "Restoring Services..."
kubectl apply -f "$BACKUP_DIR/services.yaml" -n "$NAMESPACE" || true

echo "Restoring Ingress..."
kubectl apply -f "$BACKUP_DIR/ingress.yaml" -n "$NAMESPACE" || true

echo "Restoring HPA..."
kubectl apply -f "$BACKUP_DIR/hpa.yaml" -n "$NAMESPACE" || true

echo "Restoring Network Policies..."
kubectl apply -f "$BACKUP_DIR/networkpolicies.yaml" -n "$NAMESPACE" || true

echo "Restore complete!"
echo "Verify with: kubectl get pods -n $NAMESPACE"

