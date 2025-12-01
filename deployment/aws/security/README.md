# Security Configuration

This directory contains security hardening configurations for AWS EKS deployments.

## Files

- **eks-security-hardening.yaml** - Pod Security Standards, Network Policies, RBAC
- **secrets-manager-integration.yaml** - AWS Secrets Manager integration with External Secrets Operator
- **iam-role-policy.json** - IAM policy document for pipeline service account

## Security Features

### Pod Security Standards

Enforces restricted pod security standards:
- No privileged containers
- No host network/pid/ipc
- Read-only root filesystem (where possible)
- Non-root user execution

### Network Policies

Restricts network traffic:
- Default deny all ingress/egress
- Explicit allow rules for required traffic
- DNS egress allowed for all pods

### RBAC

Minimal RBAC permissions:
- Service accounts with least privilege
- Role-based access control
- No cluster-admin permissions

### IRSA (IAM Roles for Service Accounts)

Uses AWS IAM roles instead of access keys:
- No credentials stored in secrets
- Automatic credential rotation
- Fine-grained permissions

## Deployment

### 1. Create IAM Policy

```bash
# Create IAM policy from JSON
aws iam create-policy \
  --policy-name multimodal-pipeline-policy \
  --policy-document file://security/iam-role-policy.json
```

### 2. Configure IRSA

```bash
# Create service account with IRSA
eksctl create iamserviceaccount \
  --cluster=multimodal-pipeline-cluster \
  --name=multimodal-pipeline-sa \
  --namespace=pipeline-production \
  --attach-policy-arn=arn:aws:iam::ACCOUNT_ID:policy/multimodal-pipeline-policy \
  --approve
```

### 3. Apply Security Hardening

```bash
kubectl apply -f security/eks-security-hardening.yaml
```

### 4. Configure Secrets Manager (Optional)

```bash
# Install External Secrets Operator first
helm repo add external-secrets https://charts.external-secrets.io
helm install external-secrets external-secrets/external-secrets \
  -n external-secrets-system --create-namespace

# Apply secrets manager integration
kubectl apply -f security/secrets-manager-integration.yaml
```

## IAM Policy

The IAM policy (`iam-role-policy.json`) grants permissions for:

- **S3 Access**: Read/write to pipeline data buckets (with encryption requirement)
- **CloudWatch Logs**: Create log groups and streams, write logs
- **ECR**: Pull container images
- **Secrets Manager**: Read secrets
- **CloudWatch Metrics**: Publish custom metrics

### Policy Best Practices

- Least privilege principle
- Resource-specific ARNs
- Encryption enforcement conditions
- Explicit deny for unencrypted S3 uploads

## Network Security

Network policies enforce:
- Pod-to-pod communication restrictions
- Egress filtering
- DNS access control
- Service mesh mTLS (if Istio enabled)

## Compliance

This configuration helps meet:
- **CIS Kubernetes Benchmark**
- **AWS Well-Architected Security Pillar**
- **Pod Security Standards**

## Troubleshooting

### IRSA Not Working

```bash
# Verify service account annotation
kubectl get sa multimodal-pipeline-sa -n pipeline-production -o yaml

# Test IAM role assumption
kubectl run -it --rm test-irsa \
  --image=amazon/aws-cli \
  --serviceaccount=multimodal-pipeline-sa \
  --restart=Never \
  -- aws sts get-caller-identity
```

### Network Policy Blocking Traffic

```bash
# Check network policies
kubectl get networkpolicy -n pipeline-production

# Test connectivity
kubectl run -it --rm debug --image=busybox --restart=Never -- wget -O- http://multimodal-pipeline-service:8080/health
```

### Secrets Not Syncing

```bash
# Check External Secrets Operator
kubectl get pods -n external-secrets-system

# Check ExternalSecret status
kubectl describe externalsecret pipeline-aws-secrets -n pipeline-production
```

