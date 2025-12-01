# Monitoring Configuration

This directory contains monitoring and observability configurations for AWS EKS.

## Files

- **eks-monitoring.yaml** - Prometheus configuration and CloudWatch exporter
- **cloudwatch-logs.yaml** - Fluent Bit configuration for CloudWatch Logs
- **cloudwatch-dashboard.json** - CloudWatch dashboard definition

## Monitoring Stack

### Components

1. **Prometheus**: Metrics collection and storage
2. **CloudWatch**: AWS-native monitoring and logging
3. **Fluent Bit**: Log aggregation and forwarding
4. **Grafana**: Visualization (optional, not included)

## Deployment

### 1. CloudWatch Logs

```bash
# Create IAM role for Fluent Bit
eksctl create iamserviceaccount \
  --cluster=multimodal-pipeline-cluster \
  --name=fluent-bit \
  --namespace=amazon-cloudwatch \
  --attach-policy-arn=arn:aws:iam::aws:policy/CloudWatchAgentServerPolicy \
  --approve

# Deploy Fluent Bit
kubectl apply -f monitoring/cloudwatch-logs.yaml
```

### 2. Prometheus Monitoring

```bash
# Deploy Prometheus configuration
kubectl apply -f monitoring/eks-monitoring.yaml

# Verify CloudWatch exporter
kubectl get pods -n monitoring -l app=cloudwatch-exporter
```

### 3. CloudWatch Dashboard

```bash
# Import dashboard
aws cloudwatch put-dashboard \
  --dashboard-name multimodal-pipeline-dashboard \
  --dashboard-body file://monitoring/cloudwatch-dashboard.json
```

## Metrics Collected

### Application Metrics

- CPU utilization
- Memory utilization
- GPU utilization
- Request rate
- Error rate
- Latency

### Infrastructure Metrics

- Node count
- Pod count
- Container restarts
- EBS volume I/O
- S3 bucket size
- ALB metrics

### Ray Cluster Metrics

- Ray head/worker status
- Ray job status
- Ray actor count
- Ray object store usage

## Logs

### Log Groups

- `/aws/eks/pipeline-production` - Application logs
- `/aws/eks/cluster-logs` - Cluster-wide logs

### Log Retention

Default retention: 30 days (configurable in CloudWatch)

## Alerts

Prometheus alerts configured for:
- High CPU usage (>80%)
- High memory usage (>80%)
- Pod crash looping
- Ray head down
- Multiple Ray workers down

### Alerting Channels

Configure alertmanager to send alerts to:
- PagerDuty
- Slack
- Email
- SNS

## Dashboards

### CloudWatch Dashboard

Includes widgets for:
- EKS cluster metrics
- ALB metrics
- S3 storage metrics
- EBS volume metrics
- Error logs
- Warning trends

### Access

```bash
# View dashboard in AWS Console
aws cloudwatch get-dashboard --dashboard-name multimodal-pipeline-dashboard

# Or access via AWS Console
# CloudWatch > Dashboards > multimodal-pipeline-dashboard
```

## Troubleshooting

### Fluent Bit Not Shipping Logs

```bash
# Check Fluent Bit pods
kubectl get pods -n amazon-cloudwatch -l k8s-app=fluent-bit

# Check logs
kubectl logs -n amazon-cloudwatch -l k8s-app=fluent-bit

# Verify IAM permissions
kubectl describe sa fluent-bit -n amazon-cloudwatch
```

### Prometheus Not Scraping

```bash
# Check Prometheus configuration
kubectl get configmap prometheus-config -n monitoring -o yaml

# Check Prometheus targets
# Access Prometheus UI and check /targets endpoint
kubectl port-forward -n monitoring svc/prometheus 9090:9090
```

### CloudWatch Metrics Missing

```bash
# Check CloudWatch exporter
kubectl logs -n monitoring -l app=cloudwatch-exporter

# Verify IAM permissions
kubectl describe sa cloudwatch-exporter-sa -n monitoring

# Test metric publishing
kubectl exec -n monitoring -it deployment/cloudwatch-exporter -- \
  aws cloudwatch put-metric-data \
    --namespace Pipeline/Multimodal \
    --metric-name TestMetric \
    --value 1
```

