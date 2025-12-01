# Backup and Disaster Recovery

This directory contains backup and disaster recovery configurations.

## Files

- **backup-restore.yaml** - Velero backup configuration and schedules
- **disaster-recovery.md** - Disaster recovery procedures and runbooks

## Backup Strategy

### Components Backed Up

1. **Kubernetes Resources**: ConfigMaps, Secrets, Deployments, Services, etc.
2. **Persistent Volumes**: EBS volumes via Velero snapshots
3. **S3 Data**: S3 versioning and cross-region replication
4. **Cluster Configuration**: Infrastructure as Code (Git)

### Backup Tools

- **Velero**: Kubernetes backup and restore
- **EBS Snapshots**: Volume snapshots
- **S3 Versioning**: Object versioning
- **S3 Cross-Region Replication**: Geographic redundancy

## Prerequisites

### Install Velero

```bash
# Download Velero CLI
wget https://github.com/vmware-tanzu/velero/releases/download/v1.11.0/velero-v1.11.0-linux-amd64.tar.gz
tar -xzf velero-v1.11.0-linux-amd64.tar.gz
sudo mv velero-v1.11.0-linux-amd64/velero /usr/local/bin/

# Create S3 bucket for backups
aws s3 mb s3://pipeline-backups-production --region us-east-1

# Create IAM policy for Velero
aws iam create-policy \
  --policy-name VeleroBackupPolicy \
  --policy-document file://backup/velero-iam-policy.json

# Install Velero
velero install \
  --provider aws \
  --plugins velero/velero-plugin-for-aws:v1.7.0 \
  --bucket pipeline-backups-production \
  --secret-file ./credentials-velero \
  --use-volume-snapshots=true \
  --backup-location-config region=us-east-1 \
  --snapshot-location-config region=us-east-1
```

## Deployment

```bash
# Configure IRSA for Velero
eksctl create iamserviceaccount \
  --cluster=multimodal-pipeline-cluster \
  --name=velero \
  --namespace=velero \
  --attach-policy-arn=arn:aws:iam::ACCOUNT_ID:policy/VeleroBackupPolicy \
  --approve

# Deploy backup configuration
kubectl apply -f backup/backup-restore.yaml
```

## Backup Schedules

### Daily Backups

- **Schedule**: Daily at 2 AM UTC
- **Retention**: 30 days
- **Scope**: pipeline-production namespace

### Weekly Backups

- **Schedule**: Weekly on Sunday at 3 AM UTC
- **Retention**: 90 days
- **Scope**: Full namespace backup

### Pre-Upgrade Backups

- **Schedule**: Manual trigger only
- **Retention**: 7 days
- **Scope**: Full namespace backup

## Manual Backups

```bash
# Create manual backup
velero backup create pipeline-backup-manual \
  --include-namespaces pipeline-production \
  --snapshot-volumes

# Check backup status
velero backup describe pipeline-backup-manual

# List backups
velero backup get
```

## Restore Procedures

### Full Restore

```bash
# Restore from backup
velero restore create pipeline-restore \
  --from-backup pipeline-backup-manual \
  --restore-volumes

# Check restore status
velero restore describe pipeline-restore
```

### Selective Restore

```bash
# Restore specific resources
velero restore create selective-restore \
  --from-backup pipeline-backup-manual \
  --include-resources deployments,services,configmaps
```

### Test Restore

```bash
# Restore to test namespace
velero restore create test-restore \
  --from-backup pipeline-backup-manual \
  --namespace-mapping pipeline-production:pipeline-test
```

## Disaster Recovery

See `disaster-recovery.md` for detailed procedures:
- Complete cluster loss
- Data corruption
- Regional outage
- RTO/RPO targets

## Verification

```bash
# Verify backup schedules
kubectl get schedule -n velero

# Check backup status
velero backup get

# Verify S3 backups
aws s3 ls s3://pipeline-backups-production/velero/backups/
```

## Troubleshooting

### Backup Failing

```bash
# Check Velero logs
kubectl logs -n velero -l component=velero

# Check backup logs
velero backup logs pipeline-backup-manual

# Verify S3 access
kubectl exec -n velero deployment/velero -- \
  aws s3 ls s3://pipeline-backups-production/
```

### Restore Failing

```bash
# Check restore logs
velero restore logs pipeline-restore

# Verify namespace exists
kubectl get namespace pipeline-production

# Check resource conflicts
kubectl get events -n pipeline-production
```

### Volume Snapshots Not Created

```bash
# Check snapshot location
velero snapshot-location get

# Verify EBS CSI driver
kubectl get deployment ebs-csi-controller -n kube-system

# Check IAM permissions
kubectl describe sa velero -n velero
```

