# Terraform outputs

output "s3_bucket_name" {
  description = "S3 bucket name for pipeline data"
  value       = aws_s3_bucket.pipeline_data.id
}

output "s3_bucket_arn" {
  description = "S3 bucket ARN"
  value       = aws_s3_bucket.pipeline_data.arn
}

output "iam_role_arn" {
  description = "IAM role ARN for pipeline"
  value       = aws_iam_role.pipeline.arn
}

output "namespace" {
  description = "Kubernetes namespace"
  value       = var.namespace
}

