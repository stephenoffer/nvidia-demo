#!/bin/bash
# EKS Cluster Setup Script
# Creates EKS cluster with all required components
# Usage: ./eks-cluster-setup.sh [cluster-name] [region] [environment]

set -euo pipefail

CLUSTER_NAME="${CLUSTER_NAME:-${1:-multimodal-pipeline-cluster}}"
REGION="${REGION:-${2:-us-east-1}}"
ENVIRONMENT="${ENVIRONMENT:-${3:-production}}"
KUBERNETES_VERSION="${KUBERNETES_VERSION:-1.28}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    local missing_tools=()
    
    command -v eksctl >/dev/null 2>&1 || missing_tools+=("eksctl")
    command -v aws >/dev/null 2>&1 || missing_tools+=("aws CLI")
    command -v kubectl >/dev/null 2>&1 || missing_tools+=("kubectl")
    command -v helm >/dev/null 2>&1 || missing_tools+=("helm")
    
    if [ ${#missing_tools[@]} -ne 0 ]; then
        log_error "Missing required tools: ${missing_tools[*]}"
        log_info "Install eksctl from https://eksctl.io"
        log_info "Install AWS CLI from https://aws.amazon.com/cli/"
        log_info "Install kubectl from https://kubernetes.io/docs/tasks/tools/"
        log_info "Install Helm from https://helm.sh/docs/intro/install/"
        exit 1
    fi
    
    # Check AWS credentials
    if ! aws sts get-caller-identity >/dev/null 2>&1; then
        log_error "AWS credentials not configured. Run 'aws configure'"
        exit 1
    fi
    
    # Check eksctl version
    local eksctl_version
    eksctl_version=$(eksctl version | grep -oP 'v\d+\.\d+\.\d+' | head -1)
    log_info "Using eksctl version: $eksctl_version"
    
    log_info "All prerequisites met"
}

# Validate cluster doesn't already exist
check_cluster_exists() {
    log_info "Checking if cluster already exists..."
    if eksctl get cluster --name="$CLUSTER_NAME" --region="$REGION" >/dev/null 2>&1; then
        log_warn "Cluster $CLUSTER_NAME already exists in region $REGION"
        read -p "Do you want to continue with existing cluster? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Exiting..."
            exit 0
        fi
    fi
}

# Create EKS cluster
create_cluster() {
    log_info "Creating EKS cluster: $CLUSTER_NAME in $REGION"
    
    # Check if SSH key exists
    local ssh_key_path="${SSH_KEY_PATH:-$HOME/.ssh/id_rsa.pub}"
    local ssh_access=""
    if [ -f "$ssh_key_path" ]; then
        ssh_access="--ssh-access --ssh-public-key=$ssh_key_path"
        log_info "Using SSH key: $ssh_key_path"
    else
        log_warn "SSH key not found at $ssh_key_path. Cluster will be created without SSH access."
        ssh_access="--ssh-access=false"
    fi
    
    eksctl create cluster \
        --name="$CLUSTER_NAME" \
        --region="$REGION" \
        --version="$KUBERNETES_VERSION" \
        --with-oidc \
        --nodegroup-name=gpu-nodes \
        --node-type=g5.2xlarge \
        --nodes=2 \
        --nodes-min=1 \
        --nodes-max=10 \
        --managed \
        --node-labels=accelerator=nvidia-gpu,node-type=gpu-worker \
        --node-taints=nvidia.com/gpu=present:NoSchedule \
        $ssh_access \
        --asg-access \
        --full-ecr-access \
        --appmesh-access \
        --alb-ingress-access \
        --tags="Environment=$ENVIRONMENT,ManagedBy=eksctl" || {
        log_error "Failed to create cluster"
        exit 1
    }
    
    log_info "Cluster created successfully"
}

# Create CPU node group
create_cpu_nodegroup() {
    log_info "Creating CPU node group..."
    
    eksctl create nodegroup \
        --cluster="$CLUSTER_NAME" \
        --region="$REGION" \
        --name=cpu-spot-nodes \
        --node-type=c5.2xlarge \
        --nodes=4 \
        --nodes-min=2 \
        --nodes-max=20 \
        --managed \
        --spot \
        --node-labels=workload-type=cpu \
        --asg-access \
        --tags="Environment=$ENVIRONMENT,ManagedBy=eksctl" || {
        log_error "Failed to create CPU node group"
        exit 1
    }
    
    log_info "CPU node group created successfully"
}

# Update kubeconfig
update_kubeconfig() {
    log_info "Updating kubeconfig..."
    aws eks update-kubeconfig --region "$REGION" --name "$CLUSTER_NAME" || {
        log_error "Failed to update kubeconfig"
        exit 1
    }
    
    # Verify cluster access
    if ! kubectl cluster-info >/dev/null 2>&1; then
        log_error "Failed to access cluster"
        exit 1
    fi
    
    log_info "Kubeconfig updated and cluster access verified"
}

# Install EBS CSI Driver
install_ebs_csi_driver() {
    log_info "Installing EBS CSI Driver..."
    
    # Check if addon already exists
    if eksctl get addon --cluster="$CLUSTER_NAME" --region="$REGION" --name=aws-ebs-csi-driver >/dev/null 2>&1; then
        log_warn "EBS CSI Driver addon already exists, skipping installation"
        return
    fi
    
    eksctl create addon \
        --name aws-ebs-csi-driver \
        --cluster "$CLUSTER_NAME" \
        --region "$REGION" \
        --force \
        --wait || {
        log_error "Failed to install EBS CSI Driver"
        exit 1
    }
    
    log_info "EBS CSI Driver installed successfully"
}

# Install EFS CSI Driver
install_efs_csi_driver() {
    log_info "Installing EFS CSI Driver..."
    
    if eksctl get addon --cluster="$CLUSTER_NAME" --region="$REGION" --name=aws-efs-csi-driver >/dev/null 2>&1; then
        log_warn "EFS CSI Driver addon already exists, skipping installation"
        return
    fi
    
    eksctl create addon \
        --name aws-efs-csi-driver \
        --cluster "$CLUSTER_NAME" \
        --region "$REGION" \
        --force \
        --wait || {
        log_error "Failed to install EFS CSI Driver"
        exit 1
    }
    
    log_info "EFS CSI Driver installed successfully"
}

# Install AWS Load Balancer Controller
install_alb_controller() {
    log_info "Installing AWS Load Balancer Controller..."
    
    # Add Helm repository
    helm repo add eks https://aws.github.io/eks-charts || {
        log_error "Failed to add eks Helm repository"
        exit 1
    }
    helm repo update
    
    # Check if already installed
    if helm list -n kube-system | grep -q aws-load-balancer-controller; then
        log_warn "AWS Load Balancer Controller already installed, skipping"
        return
    fi
    
    # Create IAM service account for ALB controller
    eksctl create iamserviceaccount \
        --cluster="$CLUSTER_NAME" \
        --region="$REGION" \
        --namespace=kube-system \
        --name=aws-load-balancer-controller \
        --role-name=eks-alb-controller-role \
        --attach-policy-arn=arn:aws:iam::aws:policy/ElasticLoadBalancingFullAccess \
        --approve \
        --override-existing-serviceaccounts || {
        log_error "Failed to create IAM service account for ALB controller"
        exit 1
    }
    
    # Install ALB controller
    helm install aws-load-balancer-controller eks/aws-load-balancer-controller \
        -n kube-system \
        --set clusterName="$CLUSTER_NAME" \
        --set serviceAccount.create=false \
        --set serviceAccount.name=aws-load-balancer-controller \
        --wait --timeout=5m || {
        log_error "Failed to install AWS Load Balancer Controller"
        exit 1
    }
    
    log_info "AWS Load Balancer Controller installed successfully"
}

# Install Ray Operator
install_ray_operator() {
    log_info "Installing Ray Operator..."
    
    if kubectl get deployment ray-operator -n ray-system >/dev/null 2>&1; then
        log_warn "Ray Operator already installed, skipping"
        return
    fi
    
    kubectl create namespace ray-system --dry-run=client -o yaml | kubectl apply -f -
    
    kubectl apply -f https://raw.githubusercontent.com/ray-project/ray-operator/master/deploy/ray-operator.yaml || {
        log_error "Failed to install Ray Operator"
        exit 1
    }
    
    # Wait for Ray Operator to be ready
    kubectl wait --for=condition=available --timeout=300s deployment/ray-operator -n ray-system || {
        log_error "Ray Operator failed to become ready"
        exit 1
    }
    
    log_info "Ray Operator installed successfully"
}

# Install External Secrets Operator (optional)
install_external_secrets() {
    if [ "${INSTALL_EXTERNAL_SECRETS:-false}" != "true" ]; then
        log_info "Skipping External Secrets Operator installation (set INSTALL_EXTERNAL_SECRETS=true to install)"
        return
    fi
    
    log_info "Installing External Secrets Operator..."
    
    if helm list -n external-secrets-system | grep -q external-secrets; then
        log_warn "External Secrets Operator already installed, skipping"
        return
    fi
    
    helm repo add external-secrets https://charts.external-secrets.io || {
        log_error "Failed to add external-secrets Helm repository"
        exit 1
    }
    helm repo update
    
    helm install external-secrets external-secrets/external-secrets \
        -n external-secrets-system \
        --create-namespace \
        --wait --timeout=5m || {
        log_error "Failed to install External Secrets Operator"
        exit 1
    }
    
    log_info "External Secrets Operator installed successfully"
}

# Install cert-manager (optional)
install_cert_manager() {
    if [ "${INSTALL_CERT_MANAGER:-false}" != "true" ]; then
        log_info "Skipping cert-manager installation (set INSTALL_CERT_MANAGER=true to install)"
        return
    fi
    
    log_info "Installing cert-manager..."
    
    if kubectl get deployment cert-manager -n cert-manager >/dev/null 2>&1; then
        log_warn "cert-manager already installed, skipping"
        return
    fi
    
    kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml || {
        log_error "Failed to install cert-manager"
        exit 1
    }
    
    kubectl wait --for=condition=ready pod -l app.kubernetes.io/instance=cert-manager -n cert-manager --timeout=300s || {
        log_error "cert-manager failed to become ready"
        exit 1
    }
    
    log_info "cert-manager installed successfully"
}

# Main execution
main() {
    log_info "Starting EKS cluster setup"
    log_info "Cluster: $CLUSTER_NAME"
    log_info "Region: $REGION"
    log_info "Environment: $ENVIRONMENT"
    log_info "Kubernetes Version: $KUBERNETES_VERSION"
    
    check_prerequisites
    check_cluster_exists
    
    # Create cluster if it doesn't exist
    if ! eksctl get cluster --name="$CLUSTER_NAME" --region="$REGION" >/dev/null 2>&1; then
        create_cluster
        create_cpu_nodegroup
    else
        log_info "Using existing cluster"
    fi
    
    update_kubeconfig
    install_ebs_csi_driver
    install_efs_csi_driver
    install_alb_controller
    install_ray_operator
    install_external_secrets
    install_cert_manager
    
    log_info "Cluster setup complete!"
    log_info "Cluster name: $CLUSTER_NAME"
    log_info "Region: $REGION"
    echo ""
    log_info "Next steps:"
    echo "  1. Configure IRSA: ./scripts/setup-irsa.sh $CLUSTER_NAME $REGION"
    echo "  2. Deploy pipeline: ./scripts/deploy-aws.sh $ENVIRONMENT $REGION"
    echo "  3. Verify cluster: kubectl get nodes"
}

# Run main function
main "$@"
