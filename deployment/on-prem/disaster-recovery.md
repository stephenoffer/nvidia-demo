# Disaster Recovery Plan for On-Premises Kubernetes Deployment

## Overview

This document outlines disaster recovery procedures for the multimodal pipeline deployed on self-managed Kubernetes.

## Backup Strategy

### 1. Kubernetes Resources

**Backup Tool**: Velero with local/MinIO storage
**Frequency**: Daily at 2 AM
**Retention**: 30 days

```bash
# Manual backup
velero backup create pipeline-backup-manual \
  --include-namespaces pipeline-production \
  --snapshot-volumes \
  --storage-location local-backup

# Restore from backup
velero restore create pipeline-restore \
  --from-backup pipeline-backup-manual
```

### 2. Persistent Volumes

**Backup Method**: Volume snapshots + rsync
**Frequency**: Daily
**Retention**: 30 days

```bash
# Snapshot PVC
kubectl get pvc pipeline-checkpoints -n pipeline-production -o yaml > checkpoint-backup.yaml

# rsync to backup server
rsync -av /mnt/ssd/pipeline-checkpoints/ backup-server:/backups/pipeline-checkpoints/
```

### 3. Configuration Data

**Backup Method**: Git + ConfigMaps export
**Frequency**: On every change
**Retention**: Indefinite

```bash
# Export ConfigMaps
kubectl get configmap -n pipeline-production -o yaml > configmaps-backup.yaml

# Export Secrets (encrypted)
kubectl get secret -n pipeline-production -o yaml > secrets-backup.yaml
```

### 4. Node Data

**Backup Method**: HostPath directory backups
**Frequency**: Daily
**Retention**: 7 days

```bash
# Backup input/output directories
tar -czf pipeline-data-backup-$(date +%Y%m%d).tar.gz /mnt/pipeline/
scp pipeline-data-backup-*.tar.gz backup-server:/backups/
```

## Recovery Procedures

### Scenario 1: Complete Cluster Loss

**RTO**: 4 hours
**RPO**: 24 hours

1. **Rebuild Kubernetes Cluster**
   - Install Kubernetes
   - Configure networking
   - Install required operators

2. **Restore Storage**
   ```bash
   # Restore from backup server
   rsync -av backup-server:/backups/pipeline-checkpoints/ /mnt/ssd/pipeline-checkpoints/
   ```

3. **Restore Kubernetes Resources**
   ```bash
   # Restore from Velero backup
   velero restore create cluster-restore \
     --from-backup pipeline-backup-latest
   ```

4. **Restore ConfigMaps and Secrets**
   ```bash
   kubectl apply -f configmaps-backup.yaml
   kubectl apply -f secrets-backup.yaml
   ```

5. **Verify Deployment**
   ```bash
   kubectl get pods -n pipeline-production
   ./scripts/smoke-tests.sh
   ```

### Scenario 2: Node Failure

**RTO**: 1 hour
**RPO**: 0 (no data loss if using shared storage)

1. **Cordon Failed Node**
   ```bash
   kubectl cordon <failed-node>
   ```

2. **Drain Node**
   ```bash
   kubectl drain <failed-node> --ignore-daemonsets --delete-emptydir-data
   ```

3. **Pods Reschedule Automatically**
   - Kubernetes scheduler moves pods to healthy nodes
   - Verify pod status

4. **Replace Node**
   - Provision new node
   - Join to cluster
   - Label appropriately

### Scenario 3: Data Corruption

**RTO**: 2 hours
**RPO**: 24 hours

1. **Stop Pipeline**
   ```bash
   kubectl scale deployment multimodal-pipeline --replicas=0 -n pipeline-production
   ```

2. **Restore from Backup**
   ```bash
   # Restore checkpoint volume
   rsync -av backup-server:/backups/pipeline-checkpoints/ /mnt/ssd/pipeline-checkpoints/
   
   # Restore data directories
   tar -xzf pipeline-data-backup-YYYYMMDD.tar.gz -C /
   ```

3. **Verify Data Integrity**
   ```bash
   # Run data validation
   kubectl run data-validator --image=pipeline-validator --rm -it -- \
     validate-data /data/checkpoints
   ```

4. **Resume Pipeline**
   ```bash
   kubectl scale deployment multimodal-pipeline --replicas=3 -n pipeline-production
   ```

## Backup Storage

### Primary Backup Location

- **Type**: NFS share or object storage (MinIO)
- **Location**: `/backups/pipeline/`
- **Retention**: 30 days
- **Encryption**: At rest encryption enabled

### Secondary Backup Location

- **Type**: Offsite backup server
- **Location**: Remote location
- **Frequency**: Weekly
- **Retention**: 90 days

## Testing

### Backup Verification

```bash
# Weekly backup test
velero backup describe pipeline-backup-latest
kubectl get pvc -n pipeline-production
```

### Disaster Recovery Drill

**Frequency**: Quarterly

1. Create isolated test cluster
2. Restore backup to test cluster
3. Verify all resources
4. Test data access
5. Document issues

## Monitoring

- **Backup Status**: Prometheus alerts for backup failures
- **Storage Usage**: Monitor backup storage capacity
- **Velero**: Velero backup status metrics
- **Node Health**: Node exporter metrics

## Contacts

- **On-Call Engineer**: [Contact Info]
- **Storage Team**: [Contact Info]
- **Escalation Path**: [Escalation Procedures]

