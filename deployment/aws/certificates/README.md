# Certificate Management

This directory contains TLS certificate management configurations.

## Files

- **cert-manager.yaml** - cert-manager configuration for Let's Encrypt certificates

## Overview

For AWS EKS deployments, there are two approaches to certificate management:

1. **AWS Certificate Manager (ACM)** - Recommended for production (free, auto-renewed)
2. **cert-manager with Let's Encrypt** - Alternative for non-ACM use cases

## AWS Certificate Manager (Recommended)

ACM certificates are free, automatically renewed, and integrate seamlessly with ALB.

### Request Certificate

```bash
aws acm request-certificate \
  --domain-name pipeline.example.com \
  --subject-alternative-names pipeline-api.example.com \
  --validation-method DNS \
  --region us-east-1
```

### Validate Domain

```bash
# Get validation records
aws acm describe-certificate \
  --certificate-arn arn:aws:acm:us-east-1:ACCOUNT_ID:certificate/CERT_ID \
  --query 'Certificate.DomainValidationOptions'

# Add CNAME records to your DNS provider
```

### Use in ALB Ingress

Update `../networking/alb-ingress.yaml` with certificate ARN:

```yaml
alb.ingress.kubernetes.io/certificate-arn: arn:aws:acm:REGION:ACCOUNT_ID:certificate/CERT_ID
```

## cert-manager (Alternative)

cert-manager provides automatic certificate provisioning and renewal using Let's Encrypt.

### Prerequisites

```bash
# Install cert-manager
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml

# Wait for cert-manager to be ready
kubectl wait --for=condition=ready pod -l app.kubernetes.io/instance=cert-manager -n cert-manager --timeout=300s
```

### Deployment

```bash
# Update email in cert-manager.yaml
# Then apply
kubectl apply -f certificates/cert-manager.yaml
```

### Configuration

Before deploying, update:
- Email address in ClusterIssuer
- DNS names in Certificate resources
- DNS01 solver configuration (if using wildcard certificates)

## Certificate Types

### Internal Certificates

For services within the cluster:
- `pipeline-tls-cert-internal`
- Used for service-to-service communication

### External Certificates

For public-facing services:
- `pipeline-tls-cert-external`
- Used for ALB ingress

### Ray Dashboard Certificate

For Ray dashboard access:
- `ray-dashboard-tls-cert`
- Used for secure dashboard access

## Verification

```bash
# Check certificates
kubectl get certificate -n pipeline-production

# Check certificate status
kubectl describe certificate pipeline-tls-cert-external -n pipeline-production

# Check certificate secret
kubectl get secret pipeline-tls-secret-external -n pipeline-production
```

## Troubleshooting

### Certificate Not Issued

```bash
# Check cert-manager logs
kubectl logs -n cert-manager -l app.kubernetes.io/instance=cert-manager

# Check certificate request
kubectl get certificaterequest -n pipeline-production

# Check order status (for ACME)
kubectl get order -n pipeline-production
```

### Certificate Not Renewing

```bash
# Check certificate age
kubectl get certificate -n pipeline-production -o jsonpath='{.items[*].status.notAfter}'

# Verify renewBefore setting
kubectl get certificate pipeline-tls-cert-external -n pipeline-production -o yaml | grep renewBefore
```

### DNS Validation Failing

```bash
# Check DNS records
dig _acme-challenge.pipeline.example.com TXT

# Verify DNS01 solver configuration
kubectl get clusterissuer letsencrypt-prod -o yaml
```

