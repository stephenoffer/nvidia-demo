#!/bin/bash
# Development environment setup script

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

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

log_info "Setting up development environment..."

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
log_info "Python version: $PYTHON_VERSION"

# Create virtual environment if it doesn't exist
if [ ! -d "$PROJECT_ROOT/venv" ]; then
    log_info "Creating virtual environment..."
    python3 -m venv "$PROJECT_ROOT/venv"
fi

# Activate virtual environment
log_info "Activating virtual environment..."
source "$PROJECT_ROOT/venv/bin/activate"

# Upgrade pip
log_info "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install development dependencies
log_info "Installing development dependencies..."
pip install -e ".[dev]"

# Install pre-commit hooks
log_info "Installing pre-commit hooks..."
pre-commit install

# Check for GPU
if command -v nvidia-smi >/dev/null 2>&1; then
    log_info "GPU detected:"
    nvidia-smi --query-gpu=name --format=csv,noheader
    log_info "Installing GPU dependencies..."
    pip install -e ".[gpu]" || log_warn "Failed to install GPU dependencies"
else
    log_warn "No GPU detected, skipping GPU dependencies"
fi

log_info "Development environment setup complete!"
log_info "To activate the environment, run: source venv/bin/activate"

