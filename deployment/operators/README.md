# Kubernetes Operators - Standard Integrations

This directory contains configurations for standard Kubernetes operators that replace custom implementations.

## Operators Installed

### 1. Prometheus Operator
**Replaces**: Custom metrics collection (`pipeline/observability/metrics.py`)
- Standard Prometheus metrics
- ServiceMonitor CRDs
- PodMonitor CRDs
- Alertmanager integration

### 2. Grafana Operator
**Replaces**: Custom dashboard generation (`pipeline/observability/grafana.py`)
- Dashboards as code
- Data source management
- Alerting rules

### 3. External Secrets Operator
**Replaces**: Custom secret manager (`pipeline/utils/secret_manager.py`)
- AWS Secrets Manager integration
- Vault integration
- Sealed Secrets support

### 4. Velero Operator
**Replaces**: Custom backup code (`pipeline/utils/checkpoint_recovery.py`)
- Automated backups
- Cross-region replication
- Disaster recovery

### 5. Reloader Operator
**Replaces**: Custom config watching
- Automatic ConfigMap/Secret reload
- Zero-downtime updates

### 6. Cert-Manager
**Replaces**: Custom TLS management
- Automatic certificate provisioning
- Let's Encrypt integration
- ACM integration (AWS)

### 7. Fluent Bit Operator
**Replaces**: Custom logging (`pipeline/utils/logging_config.py`)
- Log aggregation
- CloudWatch integration
- Loki integration

### 8. Jaeger Operator
**Replaces**: Custom tracing
- Distributed tracing
- OpenTelemetry integration

### 9. OPA Gatekeeper
**Replaces**: Custom policy enforcement
- Admission control
- Policy as code
- Security policies

### 10. Falco
**Replaces**: Custom security monitoring
- Runtime security
- Threat detection
- Compliance monitoring

### 11. Karpenter (AWS)
**Replaces**: Custom node management
- Automatic node provisioning
- Spot instance management
- Cost optimization

### 12. Vertical Pod Autoscaler
**Replaces**: Custom resource management (`pipeline/utils/resource_manager.py`)
- Automatic resource right-sizing
- Resource recommendations

## Installation

```bash
# Install all operators
./deployment/operators/install-operators.sh

# Or install individually
kubectl apply -f deployment/operators/prometheus-operator.yaml
kubectl apply -f deployment/operators/grafana-operator.yaml
# etc.
```

## Benefits

1. **Standardization**: Use industry-standard operators
2. **Maintenance**: Operators are maintained by the community
3. **Features**: Access to advanced features
4. **Integration**: Better integration with Kubernetes ecosystem
5. **Reliability**: Battle-tested operators
6. **Documentation**: Extensive documentation available

## Migration Guide

### From Custom Metrics to Prometheus Operator

**Before**:
```python
# Custom metrics collection
from pipeline.observability.metrics import PipelineMetrics
metrics = PipelineMetrics()
```

**After**:
```yaml
# Use Prometheus client library
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: multimodal-pipeline
spec:
  selector:
    matchLabels:
      app: multimodal-pipeline
```

### From Custom Secrets to External Secrets Operator

**Before**:
```python
# Custom secret manager
from pipeline.utils.secret_manager import SecretManager
manager = SecretManager()
secret = manager.get_secret("API_KEY")
```

**After**:
```yaml
# Use External Secrets Operator
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: pipeline-secrets
spec:
  secretStoreRef:
    name: aws-secrets-manager
  data:
  - secretKey: api-key
    remoteRef:
      key: pipeline/api-credentials
```

### From Custom Backups to Velero

**Before**:
```python
# Custom backup code
from pipeline.utils.checkpoint_recovery import CheckpointRecovery
recovery = CheckpointRecovery(checkpoint_dir)
```

**After**:
```yaml
# Use Velero Operator
apiVersion: velero.io/v1
kind: Schedule
metadata:
  name: pipeline-daily-backup
spec:
  schedule: "0 2 * * *"
  template:
    includedNamespaces:
    - pipeline-production
```

## Next Steps

1. Remove custom implementations
2. Migrate to operator-based solutions
3. Update documentation
4. Train team on operators
5. Monitor operator health

