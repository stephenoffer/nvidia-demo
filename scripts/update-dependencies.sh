#!/bin/bash
# Update dependencies script

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

# Activate virtual environment if exists
if [ -d "$PROJECT_ROOT/venv" ]; then
    source "$PROJECT_ROOT/venv/bin/activate"
fi

log_info "Updating dependencies..."

# Update pip
log_info "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Update requirements
log_info "Updating requirements.txt..."
pip install --upgrade -r "$PROJECT_ROOT/requirements.txt"
pip freeze > "$PROJECT_ROOT/requirements.txt.new"

# Check for security vulnerabilities
if command -v safety >/dev/null 2>&1; then
    log_info "Checking for security vulnerabilities..."
    safety check --file "$PROJECT_ROOT/requirements.txt.new" || log_warn "Security vulnerabilities found"
fi

log_info "Dependencies updated!"
log_info "Review requirements.txt.new and update requirements.txt if needed"

