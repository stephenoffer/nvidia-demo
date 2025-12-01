#!/bin/bash
# Validate Kubernetes Configuration
# Checks all YAML files for syntax errors and best practices

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Validating Kubernetes configurations..."

ERRORS=0

# Check if kubectl is available
if ! command -v kubectl >/dev/null 2>&1; then
    echo "Warning: kubectl not found, skipping validation"
    exit 0
fi

# Validate AWS configurations (recursively check all subdirectories)
echo "Validating AWS EKS configurations..."
while IFS= read -r -d '' file; do
    echo "  Checking $file..."
    if ! kubectl apply --dry-run=client -f "$file" >/dev/null 2>&1; then
        echo "    ERROR: Invalid YAML in $file"
        ERRORS=$((ERRORS + 1))
    fi
done < <(find "$PROJECT_ROOT/deployment/aws" -name "*.yaml" -type f -print0)

# Validate on-prem configurations
echo "Validating on-premises configurations..."
for file in "$PROJECT_ROOT"/deployment/on-prem/*.yaml; do
    if [ -f "$file" ]; then
        echo "  Checking $file..."
        if ! kubectl apply --dry-run=client -f "$file" >/dev/null 2>&1; then
            echo "    ERROR: Invalid YAML in $file"
            ERRORS=$((ERRORS + 1))
        fi
    fi
done

# Validate Helm charts
if command -v helm >/dev/null 2>&1; then
    echo "Validating Helm charts..."
    if [ -d "$PROJECT_ROOT/deployment/helm" ]; then
        helm lint "$PROJECT_ROOT/deployment/helm" || ERRORS=$((ERRORS + 1))
    fi
fi

if [ $ERRORS -eq 0 ]; then
    echo "✓ All configurations valid"
    exit 0
else
    echo "✗ Found $ERRORS error(s)"
    exit 1
fi

