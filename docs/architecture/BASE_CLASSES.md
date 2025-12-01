# Base Class Hierarchy

This document describes the base class hierarchy and patterns used throughout the pipeline.

## Stage Base Classes

### PipelineStage (Abstract Base Class)
- **Location**: `pipeline/stages/base.py`
- **Purpose**: Base class for all pipeline processing stages
- **Key Methods**:
  - `process(dataset: Dataset) -> Dataset` (abstract)
  - `_process_batch_with_error_handling()` (helper)

### ProcessorBase
- **Location**: `pipeline/stages/base.py`
- **Purpose**: Base class for item-by-item processing stages
- **Key Methods**:
  - `_process_item(item: dict[str, Any]) -> Optional[dict[str, Any]]` (abstract)
  - `process()` (implements batch processing)

### ValidatorBase
- **Location**: `pipeline/stages/base.py`
- **Purpose**: Base class for validation stages
- **Key Methods**:
  - `_validate_item(item: dict[str, Any]) -> dict[str, Any]` (abstract)
  - `process()` (implements validation logic)

## Datasource Base Classes

### FileBasedDatasource
- **Location**: `pipeline/datasources/base.py`
- **Purpose**: Base class for custom file-based datasources
- **Extends**: `ray.data.datasource.FileBasedDatasource`
- **Key Methods**:
  - `_read_stream(f: pyarrow.NativeFile, path: str) -> Iterator[pyarrow.Table]` (abstract)

## Synthetic Data Generator Base Classes

### SyntheticDataGenerator
- **Location**: `pipeline/synthetic/generator.py`
- **Purpose**: Base class for synthetic data generators
- **Key Methods**:
  - `generate_batch(size: int) -> list[dict[str, Any]]` (abstract)

## Integration Base Classes

### IsaacLabLoader
- **Location**: `pipeline/integrations/isaac_lab.py`
- **Purpose**: Loader for Isaac Lab simulation data
- **Pattern**: Standalone class (no base class needed)

### CosmosDreamsLoader
- **Location**: `pipeline/integrations/cosmos.py`
- **Purpose**: Loader for Cosmos Dreams synthetic data
- **Pattern**: Standalone class

## Naming Conventions

### Stages
- **Processors**: `*Processor` (e.g., `VideoProcessor`, `TextProcessor`)
- **Validators**: `*Validator` (e.g., `CompletenessValidator`, `PhysicsValidator`)
- **Detectors**: `*Detector` (e.g., `AnomalyDetector`, `EpisodeBoundaryDetector`)
- **Analyzers**: `*Analyzer` (e.g., `DistributionAnalyzer`)
- **Stages**: `*Stage` (e.g., `TemporalAlignmentStage`, `GPUAnalyticsStage`)

### Datasources
- **Format**: `*Datasource` (e.g., `GR00TDatasource`, `ROSBagDatasource`)

### Loaders
- **Format**: `*Loader` (e.g., `MultimodalLoader`, `IsaacLabLoader`)

### Generators
- **Format**: `*Generator` (e.g., `SyntheticDataGenerator`, `RoboticsTrajectoryGenerator`)

### Trackers
- **Format**: `*Tracker` (e.g., `MLflowTracker`, `OpenLineageTracker`)

### Managers
- **Format**: `*Manager` (e.g., `PipelineLifecycleManager`, `ResourceManager`)

## Best Practices

1. **Use Abstract Base Classes**: All base classes should inherit from `ABC` and use `@abstractmethod`
2. **Single Responsibility**: Each base class should have a clear, single purpose
3. **Consistent Naming**: Follow the naming conventions above
4. **Error Handling**: Base classes should provide common error handling patterns
5. **Documentation**: All base classes should have clear docstrings explaining their purpose

## Extension Patterns

### Creating a New Stage

```python
from pipeline.stages.base import ProcessorBase

class MyProcessor(ProcessorBase):
    def _process_item(self, item: dict[str, Any]) -> Optional[dict[str, Any]]:
        # Your processing logic
        return item
```

### Creating a New Validator

```python
from pipeline.stages.base import ValidatorBase

class MyValidator(ValidatorBase):
    def _validate_item(self, item: dict[str, Any]) -> dict[str, Any]:
        return {
            'is_valid': True,
            'validation_result': '...',
        }
```

### Creating a New Datasource

```python
from pipeline.datasources.base import FileBasedDatasource

class MyDatasource(FileBasedDatasource):
    def _read_stream(self, f, path):
        # Read file and yield pyarrow.Table blocks
        yield table
```

## Registry Integration

All components can be registered with the component registry:

```python
from pipeline.core.registry import stage_registry

@stage_registry.register('my_stage')
class MyProcessor(ProcessorBase):
    pass
```

