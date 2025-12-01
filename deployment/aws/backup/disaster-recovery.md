# Disaster Recovery Plan for AWS EKS Deployment

## Overview

This document outlines disaster recovery procedures for the multimodal pipeline deployed on AWS EKS.

## Backup Strategy

### 1. Kubernetes Resources

**Backup Tool**: Velero
**Frequency**: Daily at 2 AM UTC
**Retention**: 30 days

```bash
# Manual backup
velero backup create pipeline-backup-manual \
  --include-namespaces pipeline-production \
  --snapshot-volumes

# Restore from backup
velero restore create pipeline-restore \
  --from-backup pipeline-backup-manual
```

### 2. S3 Data

**Backup Method**: S3 Versioning + Cross-Region Replication
**Frequency**: Real-time
**Retention**: 90 days

```bash
# Enable versioning
aws s3api put-bucket-versioning \
  --bucket multimodal-pipeline-data-production \
  --versioning-configuration Status=Enabled

# Enable cross-region replication
aws s3api put-bucket-replication \
  --bucket multimodal-pipeline-data-production \
  --replication-configuration file://replication-config.json
```

### 3. EBS Volumes

**Backup Method**: EBS Snapshots
**Frequency**: Daily
**Retention**: 30 days

```bash
# Create snapshot
aws ec2 create-snapshot \
  --volume-id vol-xxxxx \
  --description "Pipeline checkpoint volume snapshot"
```

### 4. EKS Cluster Configuration

**Backup Method**: Git (Infrastructure as Code)
**Frequency**: On every change
**Retention**: Indefinite

All cluster configurations are stored in Git and can be recreated.

## Recovery Procedures

### Scenario 1: Complete Cluster Loss

**RTO**: 2 hours
**RPO**: 24 hours (last backup)

1. **Recreate EKS Cluster**
   ```bash
   ./deployment/aws/eks-cluster-setup.sh
   ```

2. **Restore Kubernetes Resources**
   ```bash
   velero restore create cluster-restore \
     --from-backup pipeline-backup-latest
   ```

3. **Restore EBS Volumes**
   ```bash
   aws ec2 create-volume \
     --snapshot-id snap-xxxxx \
     --availability-zone us-east-1a
   ```

4. **Verify Deployment**
   ```bash
   kubectl get pods -n pipeline-production
   ./scripts/smoke-tests.sh
   ```

### Scenario 2: Data Corruption

**RTO**: 1 hour
**RPO**: 24 hours

1. **Stop Pipeline**
   ```bash
   kubectl scale deployment multimodal-pipeline --replicas=0 -n pipeline-production
   ```

2. **Restore from S3 Version**
   ```bash
   aws s3api list-object-versions \
     --bucket multimodal-pipeline-data-production \
     --prefix output/
   
   aws s3api restore-object \
     --bucket multimodal-pipeline-data-production \
     --key output/file.parquet \
     --version-id VERSION_ID
   ```

3. **Restore Checkpoints**
   ```bash
   # Restore from EBS snapshot or Velero backup
   ```

4. **Resume Pipeline**
   ```bash
   kubectl scale deployment multimodal-pipeline --replicas=3 -n pipeline-production
   ```

### Scenario 3: Regional Outage

**RTO**: 4 hours
**RPO**: 24 hours

1. **Failover to Secondary Region**
   ```bash
   # Update kubeconfig to secondary region
   aws eks update-kubeconfig --region us-west-2 --name backup-cluster
   
   # Deploy to secondary region
   ./scripts/deploy-aws.sh production us-west-2
   ```

2. **Restore from Cross-Region Replication**
   ```bash
   # S3 data automatically replicated
   # Restore Kubernetes resources
   velero restore create dr-restore \
     --from-backup pipeline-backup-latest \
     --restore-volumes
   ```

## Testing

### Backup Verification

```bash
# Weekly backup test
velero backup describe pipeline-backup-latest
velero backup logs pipeline-backup-latest
```

### Disaster Recovery Drill

**Frequency**: Quarterly

1. Create test namespace
2. Restore backup to test namespace
3. Verify all resources
4. Test data access
5. Document issues and improvements

## Monitoring

- **Backup Status**: CloudWatch alarms for backup failures
- **S3 Replication**: S3 replication metrics
- **EBS Snapshots**: CloudWatch events for snapshot completion
- **Velero**: Velero backup status in Prometheus

## Contacts

- **On-Call Engineer**: [Contact Info]
- **AWS Support**: [Support Plan Details]
- **Escalation Path**: [Escalation Procedures]

