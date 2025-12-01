#!/bin/bash
# Install Standard Kubernetes Operators
# Replaces custom implementations with standard operators

set -euo pipefail

NAMESPACE="${1:-operators}"

echo "Installing standard Kubernetes operators..."
echo "Namespace: $NAMESPACE"

# Create namespace
kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -

# 1. Prometheus Operator (replaces custom metrics collection)
echo "Installing Prometheus Operator..."
kubectl apply -f https://raw.githubusercontent.com/prometheus-operator/prometheus-operator/main/bundle.yaml

# 2. Grafana Operator (replaces custom dashboard generation)
echo "Installing Grafana Operator..."
kubectl apply -f https://raw.githubusercontent.com/grafana-operator/grafana-operator/main/deploy/manifests/grafana-operator.yaml

# 3. External Secrets Operator (replaces custom secret manager)
echo "Installing External Secrets Operator..."
helm repo add external-secrets https://charts.external-secrets.io
helm install external-secrets external-secrets/external-secrets \
  -n "$NAMESPACE" \
  --create-namespace \
  --wait

# 4. Velero Operator (replaces custom backup code)
echo "Installing Velero Operator..."
helm repo add vmware-tanzu https://vmware-tanzu.github.io/helm-charts
helm install velero vmware-tanzu/velero \
  -n "$NAMESPACE" \
  --create-namespace \
  --wait

# 5. Reloader Operator (replaces custom config reloading)
echo "Installing Reloader Operator..."
kubectl apply -f https://raw.githubusercontent.com/stakater/Reloader/master/deployments/kubernetes/reloader.yaml

# 6. Cert-Manager (replaces custom TLS management)
echo "Installing cert-manager..."
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml
kubectl wait --for=condition=ready pod -l app.kubernetes.io/instance=cert-manager -n cert-manager --timeout=300s

# 7. Fluent Bit Operator (replaces custom logging)
echo "Installing Fluent Bit Operator..."
helm repo add fluent https://fluent.github.io/helm-charts
helm install fluent-bit fluent/fluent-bit \
  -n "$NAMESPACE" \
  --create-namespace \
  --wait

# 8. Loki Operator (replaces custom log aggregation)
echo "Installing Loki Operator..."
kubectl apply -f https://raw.githubusercontent.com/grafana/loki/operator/main/operator/cmd/operator/loki-operator.yaml

# 9. Jaeger Operator (replaces custom tracing)
echo "Installing Jaeger Operator..."
kubectl create namespace jaeger --dry-run=client -o yaml | kubectl apply -f -
kubectl apply -f https://github.com/jaegertracing/jaeger-operator/releases/download/v1.50.0/jaeger-operator.yaml -n jaeger

# 10. OPA Gatekeeper (replaces custom policy enforcement)
echo "Installing OPA Gatekeeper..."
kubectl apply -f https://raw.githubusercontent.com/open-policy-agent/gatekeeper/release-3.14/deploy/gatekeeper.yaml

# 11. Falco (replaces custom security monitoring)
echo "Installing Falco..."
helm repo add falcosecurity https://falcosecurity.github.io/charts
helm install falco falcosecurity/falco \
  -n "$NAMESPACE" \
  --create-namespace \
  --wait

# 12. Karpenter (replaces custom node management for AWS)
if [ "${CLOUD_PROVIDER:-}" = "aws" ]; then
    echo "Installing Karpenter..."
    helm repo add karpenter https://charts.karpenter.sh
    helm install karpenter karpenter/karpenter \
      -n "$NAMESPACE" \
      --create-namespace \
      --wait
fi

# 13. Vertical Pod Autoscaler (replaces custom resource management)
echo "Installing VPA..."
git clone https://github.com/kubernetes/autoscaler.git /tmp/autoscaler || true
kubectl apply -f /tmp/autoscaler/vertical-pod-autoscaler/deploy/vpa-release.yaml || echo "VPA installation skipped"

# 14. Metrics Server (if not already installed)
echo "Checking Metrics Server..."
if ! kubectl get deployment metrics-server -n kube-system >/dev/null 2>&1; then
    kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
fi

# 15. Node Exporter (replaces custom node monitoring)
echo "Installing Node Exporter..."
kubectl apply -f https://raw.githubusercontent.com/prometheus/node_exporter/master/examples/kubernetes/node-exporter-daemonset.yaml

echo "Operator installation complete!"
echo ""
echo "Installed operators:"
kubectl get pods -n "$NAMESPACE"
kubectl get pods -n cert-manager
kubectl get pods -n jaeger

