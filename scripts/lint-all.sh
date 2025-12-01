#!/bin/bash
# Comprehensive linting script

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

ERRORS=0

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
    ERRORS=$((ERRORS + 1))
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

# Activate virtual environment if exists
if [ -d "$PROJECT_ROOT/venv" ]; then
    source "$PROJECT_ROOT/venv/bin/activate"
fi

log_info "Running comprehensive linting..."

# Ruff linting
if command -v ruff >/dev/null 2>&1; then
    log_info "Running ruff linter..."
    ruff check pipeline/ || log_error "Ruff linting failed"
else
    log_warn "ruff not installed, skipping"
fi

# Ruff formatting check
if command -v ruff >/dev/null 2>&1; then
    log_info "Checking code formatting..."
    ruff format --check pipeline/ || log_error "Code formatting check failed"
else
    log_warn "ruff not installed, skipping"
fi

# MyPy type checking
if command -v mypy >/dev/null 2>&1; then
    log_info "Running mypy type checker..."
    mypy pipeline/ --ignore-missing-imports || log_warn "Type checking found issues"
else
    log_warn "mypy not installed, skipping"
fi

# Bandit security scanning
if command -v bandit >/dev/null 2>&1; then
    log_info "Running security scan..."
    bandit -r pipeline/ -ll || log_warn "Security scan found issues"
else
    log_warn "bandit not installed, skipping"
fi

# YAML linting
if command -v yamllint >/dev/null 2>&1; then
    log_info "Linting YAML files..."
    yamllint -d '{extends: default, rules: {line-length: {max: 120}}}' \
        deployment/*.yaml deployment/helm/*.yaml 2>/dev/null || log_warn "YAML linting found issues"
else
    log_warn "yamllint not installed, skipping"
fi

# Shell script linting
if command -v shellcheck >/dev/null 2>&1; then
    log_info "Linting shell scripts..."
    find scripts -name "*.sh" -exec shellcheck {} \; || log_warn "Shell script linting found issues"
else
    log_warn "shellcheck not installed, skipping"
fi

# Summary
echo ""
if [ $ERRORS -eq 0 ]; then
    log_info "All linting checks passed!"
    exit 0
else
    log_error "$ERRORS linting error(s) found"
    exit 1
fi

