# Networking Configuration

This directory contains networking configurations for AWS EKS, including VPC endpoints, load balancers, and service mesh.

## Files

- **vpc-endpoints.yaml** - Kubernetes NetworkPolicy for VPC endpoint access
- **vpc-endpoints.tf** - Terraform configuration for creating VPC endpoints
- **alb-ingress.yaml** - Application Load Balancer ingress configuration
- **service-mesh-istio.yaml** - Istio service mesh configuration (optional)

## VPC Endpoints

VPC endpoints allow traffic to stay within the VPC, reducing data transfer costs and improving security.

### Required Endpoints

- **S3 Gateway Endpoint**: For S3 access (no cost)
- **CloudWatch Logs Interface Endpoint**: For log shipping
- **ECR Interface Endpoints**: For container image pulls

### Deployment

```bash
# Create VPC endpoints using Terraform
cd networking
terraform init
terraform apply -var="vpc_id=vpc-xxxxx" -var="subnet_ids=[subnet-xxxxx,subnet-yyyyy]"

# Apply network policies
kubectl apply -f networking/vpc-endpoints.yaml
```

## Application Load Balancer

The ALB ingress provides external access to the pipeline service.

### Deployment

```bash
# Ensure AWS Load Balancer Controller is installed
# See ../cluster/eks-cluster-setup.sh

# Deploy ALB ingress
kubectl apply -f networking/alb-ingress.yaml
```

### Configuration

Before deploying, update:
- Certificate ARN in annotations
- Hostname in spec.rules
- Subnet IDs in annotations
- Security group IDs in annotations

## Istio Service Mesh

Istio provides advanced traffic management, security, and observability.

### Prerequisites

```bash
# Install Istio
istioctl install --set profile=default

# Enable Istio injection for namespace
kubectl label namespace pipeline-production istio-injection=enabled
```

### Deployment

```bash
kubectl apply -f networking/service-mesh-istio.yaml
```

## Network Policies

Network policies restrict traffic flow between pods. The VPC endpoints configuration includes a network policy that:

- Allows DNS queries
- Allows HTTPS to VPC endpoints
- Allows Ray head-worker communication
- Restricts other traffic

## Troubleshooting

### VPC Endpoint Not Accessible

```bash
# Check endpoint status
aws ec2 describe-vpc-endpoints --vpc-endpoint-ids vpce-xxxxx

# Check security group rules
aws ec2 describe-security-groups --group-ids sg-xxxxx

# Test connectivity from pod
kubectl run -it --rm debug --image=amazon/aws-cli --restart=Never -- aws s3 ls
```

### ALB Not Created

```bash
# Check AWS Load Balancer Controller logs
kubectl logs -n kube-system -l app.kubernetes.io/name=aws-load-balancer-controller

# Check ingress status
kubectl describe ingress multimodal-pipeline-ingress -n pipeline-production
```

### Istio Traffic Issues

```bash
# Check Istio proxy status
istioctl proxy-status

# Check virtual service
kubectl get virtualservice -n pipeline-production

# Check destination rule
kubectl get destinationrule -n pipeline-production
```

