#!/bin/bash
# Backup Kubernetes Resources
# Creates backups of all pipeline resources

set -euo pipefail

NAMESPACE="${1:-pipeline-production}"
BACKUP_DIR="${BACKUP_DIR:-./backups/$(date +%Y%m%d-%H%M%S)}"

echo "Backing up Kubernetes resources..."
echo "Namespace: $NAMESPACE"
echo "Backup directory: $BACKUP_DIR"

mkdir -p "$BACKUP_DIR"

# Backup all resources in namespace
kubectl get all -n "$NAMESPACE" -o yaml > "$BACKUP_DIR/all-resources.yaml"

# Backup ConfigMaps
kubectl get configmap -n "$NAMESPACE" -o yaml > "$BACKUP_DIR/configmaps.yaml"

# Backup Secrets (encrypted)
kubectl get secret -n "$NAMESPACE" -o yaml > "$BACKUP_DIR/secrets.yaml"

# Backup PVCs
kubectl get pvc -n "$NAMESPACE" -o yaml > "$BACKUP_DIR/pvcs.yaml"

# Backup Deployments
kubectl get deployment -n "$NAMESPACE" -o yaml > "$BACKUP_DIR/deployments.yaml"

# Backup Services
kubectl get service -n "$NAMESPACE" -o yaml > "$BACKUP_DIR/services.yaml"

# Backup Ingress
kubectl get ingress -n "$NAMESPACE" -o yaml > "$BACKUP_DIR/ingress.yaml"

# Backup HPA
kubectl get hpa -n "$NAMESPACE" -o yaml > "$BACKUP_DIR/hpa.yaml"

# Backup Network Policies
kubectl get networkpolicy -n "$NAMESPACE" -o yaml > "$BACKUP_DIR/networkpolicies.yaml"

# Create archive
tar -czf "$BACKUP_DIR.tar.gz" -C "$(dirname "$BACKUP_DIR")" "$(basename "$BACKUP_DIR")"
rm -rf "$BACKUP_DIR"

echo "Backup complete: $BACKUP_DIR.tar.gz"

