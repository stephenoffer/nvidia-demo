#!/bin/bash
# Backup script for pipeline data and configurations

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BACKUP_DIR="${BACKUP_DIR:-$PROJECT_ROOT/backups}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

# Create backup directory
mkdir -p "$BACKUP_DIR/$TIMESTAMP"

log_info "Starting backup to $BACKUP_DIR/$TIMESTAMP"

# Backup Kubernetes resources
if command -v kubectl >/dev/null 2>&1; then
    log_info "Backing up Kubernetes resources..."
    NAMESPACE="${NAMESPACE:-pipeline-production}"
    
    # ConfigMaps
    kubectl get configmap -n "$NAMESPACE" -o yaml > "$BACKUP_DIR/$TIMESTAMP/configmaps.yaml" 2>/dev/null || true
    
    # Secrets (without data values for security)
    kubectl get secret -n "$NAMESPACE" -o yaml | \
        sed 's/data:.*/data: <redacted>/' > "$BACKUP_DIR/$TIMESTAMP/secrets.yaml" 2>/dev/null || true
    
    # Deployments
    kubectl get deployment -n "$NAMESPACE" -o yaml > "$BACKUP_DIR/$TIMESTAMP/deployments.yaml" 2>/dev/null || true
    
    # Services
    kubectl get service -n "$NAMESPACE" -o yaml > "$BACKUP_DIR/$TIMESTAMP/services.yaml" 2>/dev/null || true
fi

# Backup configuration files
log_info "Backing up configuration files..."
cp -r "$PROJECT_ROOT/deployment" "$BACKUP_DIR/$TIMESTAMP/" 2>/dev/null || true
cp "$PROJECT_ROOT/pyproject.toml" "$BACKUP_DIR/$TIMESTAMP/" 2>/dev/null || true
cp "$PROJECT_ROOT/requirements.txt" "$BACKUP_DIR/$TIMESTAMP/" 2>/dev/null || true

# Create backup archive
log_info "Creating backup archive..."
cd "$BACKUP_DIR"
tar -czf "backup_${TIMESTAMP}.tar.gz" "$TIMESTAMP"
rm -rf "$TIMESTAMP"

log_info "Backup completed: backup_${TIMESTAMP}.tar.gz"

# Cleanup old backups (keep last 7 days)
find "$BACKUP_DIR" -name "backup_*.tar.gz" -mtime +7 -delete 2>/dev/null || true

log_info "Backup process completed"

