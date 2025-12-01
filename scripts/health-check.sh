#!/bin/bash
# Comprehensive health check script

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

HEALTHY=0

check_pass() {
    echo -e "${GREEN}✓${NC} $1"
}

check_fail() {
    echo -e "${RED}✗${NC} $1"
    HEALTHY=1
}

check_warn() {
    echo -e "${YELLOW}⚠${NC} $1"
}

echo "Running comprehensive health checks..."

# Check Python installation
if command -v python3 >/dev/null 2>&1; then
    PYTHON_VERSION=$(python3 --version)
    check_pass "Python installed: $PYTHON_VERSION"
else
    check_fail "Python not installed"
fi

# Check required commands
for cmd in kubectl docker git; do
    if command -v $cmd >/dev/null 2>&1; then
        VERSION=$($cmd --version 2>&1 | head -n1)
        check_pass "$cmd installed: $VERSION"
    else
        check_warn "$cmd not installed (optional)"
    fi
done

# Check virtual environment
if [ -d "$PROJECT_ROOT/venv" ]; then
    check_pass "Virtual environment exists"
else
    check_warn "Virtual environment not found (run ./scripts/setup-dev.sh)"
fi

# Check dependencies
if [ -f "$PROJECT_ROOT/requirements.txt" ]; then
    check_pass "requirements.txt exists"
else
    check_fail "requirements.txt missing"
fi

# Check configuration files
for file in pyproject.toml setup.py Dockerfile; do
    if [ -f "$PROJECT_ROOT/$file" ]; then
        check_pass "$file exists"
    else
        check_fail "$file missing"
    fi
done

# Check Kubernetes configs
if [ -d "$PROJECT_ROOT/deployment" ]; then
    YAML_COUNT=$(find "$PROJECT_ROOT/deployment" -name "*.yaml" | wc -l)
    check_pass "Kubernetes configs found: $YAML_COUNT files"
else
    check_fail "deployment/ directory missing"
fi

# Check CI/CD
if [ -d "$PROJECT_ROOT/.github/workflows" ]; then
    WORKFLOW_COUNT=$(find "$PROJECT_ROOT/.github/workflows" -name "*.yml" | wc -l)
    check_pass "CI/CD workflows found: $WORKFLOW_COUNT files"
else
    check_warn ".github/workflows/ directory missing"
fi

# Check documentation
for doc in README.md CHANGELOG.md CONTRIBUTING.md; do
    if [ -f "$PROJECT_ROOT/$doc" ]; then
        check_pass "$doc exists"
    else
        check_warn "$doc missing"
    fi
done

# Summary
echo ""
if [ $HEALTHY -eq 0 ]; then
    echo -e "${GREEN}All critical checks passed!${NC}"
    exit 0
else
    echo -e "${RED}Some checks failed. Please review above.${NC}"
    exit 1
fi

