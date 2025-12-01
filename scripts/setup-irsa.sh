#!/bin/bash
# Setup IRSA (IAM Roles for Service Accounts) for AWS EKS
# Usage: ./scripts/setup-irsa.sh [cluster-name] [region]

set -euo pipefail

CLUSTER_NAME="${1:-multimodal-pipeline-cluster}"
REGION="${2:-us-east-1}"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

echo "Setting up IRSA for cluster: $CLUSTER_NAME"
echo "Account ID: $ACCOUNT_ID"
echo "Region: $REGION"

# Check prerequisites
command -v eksctl >/dev/null 2>&1 || { echo "eksctl required"; exit 1; }
command -v aws >/dev/null 2>&1 || { echo "aws CLI required"; exit 1; }

# Create IAM policy from JSON file
POLICY_NAME="multimodal-pipeline-policy"
POLICY_ARN=""

if aws iam get-policy --policy-arn "arn:aws:iam::${ACCOUNT_ID}:policy/${POLICY_NAME}" >/dev/null 2>&1; then
    echo "Policy $POLICY_NAME already exists"
    POLICY_ARN="arn:aws:iam::${ACCOUNT_ID}:policy/${POLICY_NAME}"
else
    echo "Creating IAM policy..."
    POLICY_ARN=$(aws iam create-policy \
        --policy-name "$POLICY_NAME" \
        --policy-document file://deployment/aws/security/iam-role-policy.json \
        --query 'Policy.Arn' \
        --output text)
    echo "Policy created: $POLICY_ARN"
fi

# Create service account with IRSA
echo "Creating service account with IRSA..."
eksctl create iamserviceaccount \
    --cluster="$CLUSTER_NAME" \
    --region="$REGION" \
    --name=multimodal-pipeline-sa \
    --namespace=pipeline-production \
    --create-namespace \
    --role-name=multimodal-pipeline-role \
    --attach-policy-arn="$POLICY_ARN" \
    --approve \
    --override-existing-serviceaccounts

echo "IRSA setup complete!"
echo "Service account: multimodal-pipeline-sa"
echo "IAM role: multimodal-pipeline-role"
echo "Policy ARN: $POLICY_ARN"

