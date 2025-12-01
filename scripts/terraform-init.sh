#!/bin/bash
# Terraform initialization script

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TERRAFORM_DIR="$PROJECT_ROOT/terraform"

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

# Check terraform is installed
if ! command -v terraform >/dev/null 2>&1; then
    log_warn "Terraform not installed. Install from https://www.terraform.io/downloads"
    exit 1
fi

ENVIRONMENT="${1:-staging}"

log_info "Initializing Terraform for $ENVIRONMENT environment..."

cd "$TERRAFORM_DIR"

# Initialize Terraform
terraform init

# Validate configuration
terraform validate

log_info "Terraform initialized successfully!"
log_info "To plan: terraform plan -var='environment=$ENVIRONMENT'"
log_info "To apply: terraform apply -var='environment=$ENVIRONMENT'"

