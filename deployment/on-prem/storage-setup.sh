#!/bin/bash
# Storage Setup Script for On-Premises Kubernetes
# Configures local storage directories and permissions

set -euo pipefail

echo "Setting up storage for on-premises Kubernetes deployment..."

# Create directories on all nodes
# Run this script on each node or use Ansible/other automation

BASE_DIR="/mnt/pipeline"
SSD_DIR="/mnt/ssd/pipeline-checkpoints"
LOG_DIR="/var/log/pipeline"

# Create base directories
sudo mkdir -p "$BASE_DIR/input"
sudo mkdir -p "$BASE_DIR/output"
sudo mkdir -p "$SSD_DIR"
sudo mkdir -p "$LOG_DIR"

# Set ownership (UID 1000 = pipeline user)
sudo chown -R 1000:1000 "$BASE_DIR"
sudo chown -R 1000:1000 "$SSD_DIR"
sudo chown -R 1000:1000 "$LOG_DIR"

# Set permissions
sudo chmod -R 755 "$BASE_DIR"
sudo chmod -R 755 "$SSD_DIR"
sudo chmod -R 755 "$LOG_DIR"

# Create subdirectories
sudo mkdir -p "$BASE_DIR/input/videos"
sudo mkdir -p "$BASE_DIR/input/text"
sudo mkdir -p "$BASE_DIR/input/sensor"
sudo mkdir -p "$BASE_DIR/output/curated"
sudo mkdir -p "$BASE_DIR/output/checkpoints"

# Set SELinux context (if SELinux is enabled)
if command -v chcon >/dev/null 2>&1; then
    sudo chcon -Rt svirt_sandbox_file_t "$BASE_DIR" || true
    sudo chcon -Rt svirt_sandbox_file_t "$SSD_DIR" || true
    sudo chcon -Rt svirt_sandbox_file_t "$LOG_DIR" || true
fi

echo "Storage setup complete!"
echo "Directories created:"
echo "  - $BASE_DIR/input"
echo "  - $BASE_DIR/output"
echo "  - $SSD_DIR"
echo "  - $LOG_DIR"

