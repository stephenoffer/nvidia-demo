# EKS Add-ons Terraform Configuration
# Installs required add-ons for AWS EKS deployment
# Requires: EKS cluster created, OIDC provider configured

terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

# Data sources
data "aws_caller_identity" "current" {}

data "aws_eks_cluster" "main" {
  name = var.cluster_name
}

data "aws_eks_cluster_auth" "main" {
  name = var.cluster_name
}

# EBS CSI Driver Add-on
resource "aws_eks_addon" "ebs_csi_driver" {
  cluster_name             = var.cluster_name
  addon_name               = "aws-ebs-csi-driver"
  addon_version            = var.ebs_csi_driver_version
  service_account_role_arn = aws_iam_role.ebs_csi_driver.arn
  
  tags = merge(
    var.tags,
    {
      Name        = "ebs-csi-driver"
      Addon       = "ebs-csi-driver"
      Environment = var.environment
    }
  )

  depends_on = [
    aws_iam_role_policy_attachment.ebs_csi_driver
  ]
}

# EFS CSI Driver Add-on
resource "aws_eks_addon" "efs_csi_driver" {
  cluster_name             = var.cluster_name
  addon_name               = "aws-efs-csi-driver"
  addon_version            = var.efs_csi_driver_version
  service_account_role_arn = aws_iam_role.efs_csi_driver.arn
  
  tags = merge(
    var.tags,
    {
      Name        = "efs-csi-driver"
      Addon       = "efs-csi-driver"
      Environment = var.environment
    }
  )

  depends_on = [
    aws_iam_role_policy_attachment.efs_csi_driver
  ]
}

# VPC CNI Add-on (usually pre-installed, but can be updated)
resource "aws_eks_addon" "vpc_cni" {
  cluster_name  = var.cluster_name
  addon_name    = "vpc-cni"
  addon_version = var.vpc_cni_version
  
  resolve_conflicts_on_update = "OVERWRITE"
  
  tags = merge(
    var.tags,
    {
      Name        = "vpc-cni"
      Addon       = "vpc-cni"
      Environment = var.environment
    }
  )
}

# CoreDNS Add-on (usually pre-installed)
resource "aws_eks_addon" "coredns" {
  cluster_name  = var.cluster_name
  addon_name    = "coredns"
  addon_version = var.coredns_version
  
  resolve_conflicts_on_update = "OVERWRITE"
  
  tags = merge(
    var.tags,
    {
      Name        = "coredns"
      Addon       = "coredns"
      Environment = var.environment
    }
  )
}

# Kube-proxy Add-on (usually pre-installed)
resource "aws_eks_addon" "kube_proxy" {
  cluster_name  = var.cluster_name
  addon_name    = "kube-proxy"
  addon_version = var.kube_proxy_version
  
  resolve_conflicts_on_update = "OVERWRITE"
  
  tags = merge(
    var.tags,
    {
      Name        = "kube-proxy"
      Addon       = "kube-proxy"
      Environment = var.environment
    }
  )
}

# IAM roles for add-ons
resource "aws_iam_role" "ebs_csi_driver" {
  name = "eks-ebs-csi-driver-role-${var.environment}-${substr(var.cluster_name, -8, -1)}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRoleWithWebIdentity"
        Effect = "Allow"
        Principal = {
          Federated = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:oidc-provider/${replace(data.aws_eks_cluster.main.identity[0].oidc[0].issuer, "https://", "")}"
        }
        Condition = {
          StringEquals = {
            "${replace(data.aws_eks_cluster.main.identity[0].oidc[0].issuer, "https://", "")}:sub" = "system:serviceaccount:kube-system:ebs-csi-controller-sa"
            "${replace(data.aws_eks_cluster.main.identity[0].oidc[0].issuer, "https://", "")}:aud" = "sts.amazonaws.com"
          }
        }
      }
    ]
  })

  tags = merge(
    var.tags,
    {
      Name        = "ebs-csi-driver-role"
      Service     = "ebs-csi-driver"
      Environment = var.environment
    }
  )
}

resource "aws_iam_role_policy_attachment" "ebs_csi_driver" {
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonEBSCSIDriverPolicy"
  role       = aws_iam_role.ebs_csi_driver.name
}

resource "aws_iam_role" "efs_csi_driver" {
  name = "eks-efs-csi-driver-role-${var.environment}-${substr(var.cluster_name, -8, -1)}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRoleWithWebIdentity"
        Effect = "Allow"
        Principal = {
          Federated = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:oidc-provider/${replace(data.aws_eks_cluster.main.identity[0].oidc[0].issuer, "https://", "")}"
        }
        Condition = {
          StringEquals = {
            "${replace(data.aws_eks_cluster.main.identity[0].oidc[0].issuer, "https://", "")}:sub" = "system:serviceaccount:kube-system:efs-csi-controller-sa"
            "${replace(data.aws_eks_cluster.main.identity[0].oidc[0].issuer, "https://", "")}:aud" = "sts.amazonaws.com"
          }
        }
      }
    ]
  })

  tags = merge(
    var.tags,
    {
      Name        = "efs-csi-driver-role"
      Service     = "efs-csi-driver"
      Environment = var.environment
    }
  )
}

resource "aws_iam_role_policy_attachment" "efs_csi_driver" {
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonEFSCSIDriverPolicy"
  role       = aws_iam_role.efs_csi_driver.name
}

# Variables
variable "cluster_name" {
  description = "Name of the EKS cluster"
  type        = string
}

variable "environment" {
  description = "Environment name"
  type        = string
}

variable "ebs_csi_driver_version" {
  description = "Version of the EBS CSI driver add-on"
  type        = string
  default     = "v1.28.0-eksbuild.1"
}

variable "efs_csi_driver_version" {
  description = "Version of the EFS CSI driver add-on"
  type        = string
  default     = "v2.0.7-eksbuild.1"
}

variable "vpc_cni_version" {
  description = "Version of the VPC CNI add-on"
  type        = string
  default     = "v1.16.0-eksbuild.1"
}

variable "coredns_version" {
  description = "Version of the CoreDNS add-on"
  type        = string
  default     = "v1.10.1-eksbuild.1"
}

variable "kube_proxy_version" {
  description = "Version of the kube-proxy add-on"
  type        = string
  default     = "v1.28.1-eksbuild.1"
}

variable "tags" {
  description = "Tags to apply to resources"
  type        = map(string)
  default     = {}
}

# Outputs
output "ebs_csi_driver_addon_arn" {
  description = "ARN of the EBS CSI driver add-on"
  value       = aws_eks_addon.ebs_csi_driver.arn
}

output "efs_csi_driver_addon_arn" {
  description = "ARN of the EFS CSI driver add-on"
  value       = aws_eks_addon.efs_csi_driver.arn
}

output "ebs_csi_driver_role_arn" {
  description = "ARN of the EBS CSI driver IAM role"
  value       = aws_iam_role.ebs_csi_driver.arn
}

output "efs_csi_driver_role_arn" {
  description = "ARN of the EFS CSI driver IAM role"
  value       = aws_iam_role.efs_csi_driver.arn
}
