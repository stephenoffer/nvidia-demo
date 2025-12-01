# Deprecated Modules Removal

## Summary

Removed deprecated modules that were replaced with standard OSS tools or consolidated into other modules.

## Removed Modules

### 1. `pipeline/utils/data_lineage.py`
- **Replaced by**: `pipeline.integrations.openlineage.OpenLineageTracker`
- **Reason**: OpenLineage provides standard data lineage format and better integration
- **Migration**: Use `create_openlineage_tracker()` from `pipeline.integrations.openlineage`

### 2. `pipeline/utils/data_versioning.py`
- **Replaced by**: `pipeline.integrations.mlflow.MLflowTracker`
- **Reason**: MLflow provides integrated data versioning with experiment tracking
- **Migration**: Use `create_mlflow_tracker()` and call `log_data_version()`

### 3. `pipeline/utils/checkpoint_recovery.py`
- **Replaced by**: `pipeline.utils.checkpoint.PipelineCheckpoint`
- **Reason**: Functionality consolidated into `PipelineCheckpoint` class
- **Migration**: Use `PipelineCheckpoint` directly, it has all recovery methods

### 4. `pipeline/utils/retry_cloud.py`
- **Replaced by**: `pipeline.utils.retry.retry_cloud_storage`
- **Reason**: Functionality consolidated into `retry.py`
- **Migration**: Import `retry_cloud_storage` from `pipeline.utils.retry`

### 5. `pipeline/utils/input_sanitization.py`
- **Replaced by**: `pipeline.utils.input_validation.InputValidator`
- **Reason**: Functionality consolidated into `InputValidator` class
- **Migration**: Use `InputValidator` which has all sanitization methods

## Updated Imports

### Before → After

```python
# Data Lineage
# Before
from pipeline.utils.data_lineage import DataLineageTracker
tracker = DataLineageTracker(enabled=True)

# After
from pipeline.integrations.openlineage import create_openlineage_tracker
tracker = create_openlineage_tracker(enabled=True)

# Data Versioning
# Before
from pipeline.utils.data_versioning import DataVersionManager
manager = DataVersionManager(version_dir="./versions")

# After
from pipeline.integrations.mlflow import create_mlflow_tracker
tracker = create_mlflow_tracker()
tracker.log_data_version(...)

# Checkpoint Recovery
# Before
from pipeline.utils.checkpoint_recovery import CheckpointRecovery
recovery = CheckpointRecovery(checkpoint_dir="./checkpoints")

# After
from pipeline.utils.checkpoint import PipelineCheckpoint
checkpoint = PipelineCheckpoint(checkpoint_dir="./checkpoints")

# Retry Cloud Storage
# Before
from pipeline.utils.retry_cloud import retry_cloud_storage

# After
from pipeline.utils.retry import retry_cloud_storage

# Input Sanitization
# Before
from pipeline.utils.input_sanitization import InputSanitizer
sanitizer = InputSanitizer(strict=True)

# After
from pipeline.utils.input_validation import InputValidator
validator = InputValidator(strict=True)
```

## Files Updated

- `pipeline/core/orchestrator.py` - Updated to use new modules
- `pipeline/datasources/groot.py` - Updated retry import
- `pipeline/loaders/multimodal.py` - Updated retry import

## Benefits

1. **Cleaner codebase**: Removed ~500+ lines of duplicate/deprecated code
2. **Standard tools**: Using industry-standard OSS tools (OpenLineage, MLflow)
3. **Better integration**: Better integration with ML ecosystem
4. **Less maintenance**: Fewer modules to maintain

## Breaking Changes

⚠️ **Breaking**: These modules are completely removed. Update imports before upgrading.

If you have code using these modules, update imports as shown above.

## Verification

After removal, verify imports work:
```bash
python -c "from pipeline.core.orchestrator import MultimodalPipeline; print('OK')"
```

