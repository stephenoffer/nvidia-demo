#!/bin/bash
# Configuration validation script

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

ERRORS=0

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
    ERRORS=$((ERRORS + 1))
}

log_success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

# Validate Python syntax
validate_python() {
    log_info "Validating Python syntax..."
    find "$PROJECT_ROOT/pipeline" -name "*.py" -exec python3 -m py_compile {} \; || {
        log_error "Python syntax validation failed"
        return 1
    }
    log_success "Python syntax valid"
}

# Validate YAML files
validate_yaml() {
    log_info "Validating YAML files..."
    command -v yamllint >/dev/null 2>&1 || {
        log_warn "yamllint not installed, skipping YAML validation"
        return 0
    }
    
    yamllint -d '{extends: default, rules: {line-length: {max: 120}}}' \
        "$PROJECT_ROOT/deployment"/*.yaml \
        "$PROJECT_ROOT/deployment/helm"/*.yaml \
        "$PROJECT_ROOT/deployment/helm/templates"/*.yaml 2>/dev/null || {
        log_error "YAML validation failed"
        return 1
    }
    log_success "YAML files valid"
}

# Validate Kubernetes manifests
validate_kubernetes() {
    log_info "Validating Kubernetes manifests..."
    command -v kubectl >/dev/null 2>&1 || {
        log_warn "kubectl not installed, skipping Kubernetes validation"
        return 0
    }
    
    for file in "$PROJECT_ROOT/deployment"/*.yaml; do
        kubectl apply --dry-run=client -f "$file" >/dev/null 2>&1 || {
            log_error "Kubernetes manifest validation failed: $file"
            return 1
        }
    done
    log_success "Kubernetes manifests valid"
}

# Validate Dockerfile
validate_dockerfile() {
    log_info "Validating Dockerfile..."
    command -v hadolint >/dev/null 2>&1 || {
        log_warn "hadolint not installed, skipping Dockerfile validation"
        return 0
    }
    
    hadolint "$PROJECT_ROOT/Dockerfile" || {
        log_error "Dockerfile validation failed"
        return 1
    }
    log_success "Dockerfile valid"
}

# Main validation
main() {
    echo "Validating project configuration..."
    
    validate_python
    validate_yaml
    validate_kubernetes
    validate_dockerfile
    
    if [ $ERRORS -eq 0 ]; then
        log_success "All validations passed!"
        return 0
    else
        log_error "$ERRORS validation error(s) found"
        return 1
    fi
}

main "$@"

