# Terraform variables

variable "environment" {
  description = "Environment name"
  type        = string
  validation {
    condition     = contains(["staging", "production"], var.environment)
    error_message = "Environment must be 'staging' or 'production'."
  }
}

variable "cluster_name" {
  description = "EKS cluster name"
  type        = string
}

variable "region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "namespace" {
  description = "Kubernetes namespace"
  type        = string
  default     = "pipeline-production"
}

variable "replica_count" {
  description = "Number of pipeline replicas"
  type        = number
  default     = 3
}

variable "gpu_count" {
  description = "Number of GPUs per pod"
  type        = number
  default     = 8
}

