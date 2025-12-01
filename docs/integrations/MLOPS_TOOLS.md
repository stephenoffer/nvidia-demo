# MLOps Tools Integration Guide

This document describes the integration with standard MLOps tools, replacing custom implementations with industry-standard solutions.

## Overview

We've replaced custom implementations with standard OSS tools:

1. **Data Lineage**: OpenLineage (replaces `DataLineageTracker`)
2. **Experiment Tracking**: MLflow or Weights & Biases (replaces custom metrics tracking)
3. **Data Versioning**: MLflow or DVC (replaces `DataVersionManager`)
4. **Model Versioning**: MLflow Model Registry

## OpenLineage Integration

### Purpose
OpenLineage provides standard data lineage tracking, replacing our custom `DataLineageTracker`.

### Benefits
- Standard format compatible with Airflow, Spark, dbt, etc.
- Better visualization and governance
- Integration with data platforms

### Usage

```python
from pipeline.integrations.openlineage import create_openlineage_tracker

# Initialize tracker
tracker = create_openlineage_tracker(
    url="http://openlineage-backend:5000",
    namespace="multimodal-pipeline",
)

# Track lineage for a pipeline stage
run_id = tracker.track_lineage(
    stage_name="deduplication",
    input_paths=["s3://bucket/input/"],
    output_path="s3://bucket/output/",
    metadata={"method": "semantic", "threshold": 0.95},
)
```

### Configuration

Set environment variable:
```bash
export OPENLINEAGE_URL=http://openlineage-backend:5000
```

## MLflow Integration

### Purpose
MLflow provides experiment tracking, model versioning, and data versioning.

### Benefits
- Industry-standard experiment tracking
- Model registry for versioning
- Artifact storage
- UI for visualization

### Usage

```python
from pipeline.integrations.mlflow import create_mlflow_tracker

# Initialize tracker
tracker = create_mlflow_tracker(
    tracking_uri="http://mlflow-server:5000",
    experiment_name="groot-data-etl",
)

# Start run
tracker.start_run(run_name="pipeline-run-001")

# Log parameters
tracker.log_params({
    "num_gpus": 8,
    "batch_size": 1000,
    "similarity_threshold": 0.95,
})

# Log metrics
tracker.log_metrics({
    "total_samples": 1000000,
    "dedup_rate": 0.15,
    "throughput": 5000.0,
})

# Log data version
tracker.log_data_version(
    input_paths=["s3://bucket/input/"],
    output_path="s3://bucket/output/",
    dataset_hash="abc123",
)

# End run
tracker.end_run()
```

### Integration with PipelineMetrics

```python
from pipeline.observability.metrics import PipelineMetrics

# Enable MLflow in metrics
metrics = PipelineMetrics(
    enabled=True,
    enable_prometheus=True,
    enable_mlflow=True,
    mlflow_tracking_uri="http://mlflow-server:5000",
)
```

## Weights & Biases Integration

### Purpose
W&B provides experiment tracking as an alternative to MLflow.

### Benefits
- Excellent visualization
- Team collaboration features
- Hyperparameter optimization
- Model versioning

### Usage

```python
from pipeline.integrations.wandb import create_wandb_tracker

# Initialize tracker
tracker = create_wandb_tracker(
    project="multimodal-pipeline",
    entity="ml-team",
)

# Start run
tracker.start_run(
    run_name="pipeline-run-001",
    config={"num_gpus": 8, "batch_size": 1000},
)

# Log metrics
tracker.log_metrics({
    "total_samples": 1000000,
    "dedup_rate": 0.15,
})

# End run
tracker.end_run()
```

### Integration with PipelineMetrics

```python
from pipeline.observability.metrics import PipelineMetrics

# Enable W&B in metrics
metrics = PipelineMetrics(
    enabled=True,
    enable_prometheus=True,
    enable_wandb=True,
    wandb_project="multimodal-pipeline",
)
```

## Migration Guide

### From DataLineageTracker to OpenLineage

**Before**:
```python
from pipeline.utils.data_lineage import DataLineageTracker

tracker = DataLineageTracker(enabled=True)
lineage_id = tracker.add_lineage(
    item_id="item_001",
    source_path="s3://bucket/data.parquet",
    source_type="file",
)
```

**After**:
```python
from pipeline.integrations.openlineage import create_openlineage_tracker

tracker = create_openlineage_tracker(enabled=True)
run_id = tracker.track_lineage(
    stage_name="data-loading",
    input_paths=["s3://bucket/data.parquet"],
    output_path="s3://bucket/output/",
)
```

### From DataVersionManager to MLflow

**Before**:
```python
from pipeline.utils.data_versioning import DataVersionManager

manager = DataVersionManager(version_dir="./versions")
version = manager.create_version(
    input_paths=["s3://bucket/input/"],
    output_path="s3://bucket/output/",
    pipeline_config=config,
)
```

**After**:
```python
from pipeline.integrations.mlflow import create_mlflow_tracker

tracker = create_mlflow_tracker()
tracker.start_run()
tracker.log_data_version(
    input_paths=["s3://bucket/input/"],
    output_path="s3://bucket/output/",
    dataset_hash="abc123",
)
```

## Installation

```bash
# OpenLineage
pip install openlineage-python

# MLflow
pip install mlflow

# Weights & Biases
pip install wandb
```

## Configuration

### MLflow Server

Deploy MLflow server:
```bash
mlflow server --host 0.0.0.0 --port 5000
```

Or use Docker:
```bash
docker run -p 5000:5000 ghcr.io/mlflow/mlflow:v2.8.1
```

### OpenLineage Backend

Deploy OpenLineage backend (see OpenLineage documentation).

### W&B

Sign up at https://wandb.ai and configure:
```bash
wandb login
```

## Best Practices

1. **Use MLflow for experiment tracking** - Industry standard, good UI
2. **Use OpenLineage for data lineage** - Standard format, better integration
3. **Use W&B for visualization** - If team prefers W&B UI
4. **Don't use both MLflow and W&B** - Choose one for consistency
5. **Always log data versions** - Critical for reproducibility

## Deprecated Modules

The following modules are deprecated but kept for backward compatibility:

- `pipeline.utils.data_lineage` - Use `pipeline.integrations.openlineage`
- `pipeline.utils.data_versioning` - Use `pipeline.integrations.mlflow`

These modules will be removed in a future version.

