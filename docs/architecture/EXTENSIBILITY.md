# Extensibility Guide

This document describes how to extend the pipeline with custom components.

## Component Registry System

The pipeline uses a registry pattern for dynamic component discovery and registration.

### Registering Custom Stages

```python
from pipeline.core.registry import stage_registry
from pipeline.stages.base import ProcessorBase

@stage_registry.register('my_custom_stage')
class MyCustomStage(ProcessorBase):
    def _process_item(self, item: dict[str, Any]) -> dict[str, Any] | None:
        # Your custom processing logic
        return item

# Or register directly
stage_registry.register('my_stage', MyCustomStage)

# Or with a factory function
def create_my_stage(config):
    return MyCustomStage(config=config)

stage_registry.register('my_stage', factory=create_my_stage)
```

### Registering Custom Datasources

```python
from pipeline.core.registry import datasource_registry
from pipeline.datasources.base import FileBasedDatasource

@datasource_registry.register('my_format')
class MyFormatDatasource(FileBasedDatasource):
    def _read_stream(self, f, path):
        # Your custom reading logic
        yield table
```

### Using Factory Functions

```python
from pipeline.core.factory import create_stage, create_validator, create_datasource

# Create stages
stage = create_stage('temporal_alignment', config=config)
validator = create_validator('completeness', reject_invalid=True)

# Create datasources
datasource = create_datasource('groot', paths=['/path/to/data'])
```

## Plugin System

### Creating a Plugin Package

Create a Python package with the following structure:

```
my_pipeline_plugin/
├── __init__.py
├── stages.py      # Custom stages
├── datasources.py # Custom datasources
└── setup.py       # Plugin registration
```

### Plugin Registration

```python
# my_pipeline_plugin/__init__.py
from pipeline.core.registry import stage_registry, datasource_registry
from .stages import MyCustomStage
from .datasources import MyFormatDatasource

def register_plugin():
    """Register all plugin components."""
    stage_registry.register('my_stage', MyCustomStage)
    datasource_registry.register('my_format', MyFormatDatasource)

# Auto-register on import
register_plugin()
```

### Using Plugins

```python
# Import plugin to register components
import my_pipeline_plugin

# Now you can use registered components
from pipeline.core.factory import create_stage
stage = create_stage('my_stage', config=config)
```

## Base Classes

### PipelineStage

Base class for all pipeline stages. Subclasses must implement `process()`.

```python
from pipeline.stages.base import PipelineStage
from ray.data import Dataset

class MyStage(PipelineStage):
    def process(self, dataset: Dataset) -> Dataset:
        # Your processing logic
        return dataset.map_batches(self._process_batch)
```

### ProcessorBase

Base class for item-by-item processing stages.

```python
from pipeline.stages.base import ProcessorBase

class MyProcessor(ProcessorBase):
    def _process_item(self, item: dict[str, Any]) -> dict[str, Any] | None:
        # Process single item
        # Return None to filter out item
        return item
```

### ValidatorBase

Base class for validation stages.

```python
from pipeline.stages.base import ValidatorBase

class MyValidator(ValidatorBase):
    def _validate_item(self, item: dict[str, Any]) -> dict[str, Any]:
        # Return validation result
        return {
            'is_valid': True,
            'validation_score': 0.95,
        }
```

### FileBasedDatasource

Base class for custom file-based datasources.

```python
from pipeline.datasources.base import FileBasedDatasource

class MyDatasource(FileBasedDatasource):
    def _read_stream(self, f, path):
        # Read file and yield pyarrow.Table blocks
        data = f.readall()
        table = pyarrow.Table.from_pydict({'data': [data]})
        yield table
```

## Best Practices

1. **Follow Naming Conventions**
   - Stages: `*Stage` or `*Processor` suffix
   - Validators: `*Validator` suffix
   - Datasources: `*Datasource` suffix

2. **Use Type Hints**
   - Always use type hints for better IDE support
   - Use `from __future__ import annotations` for forward references

3. **Error Handling**
   - Use the base class error handling patterns
   - Log errors appropriately
   - Return None to filter items (for processors)

4. **Documentation**
   - Add docstrings to all public methods
   - Document expected input/output formats
   - Include usage examples

5. **Testing**
   - Write unit tests for custom components
   - Test error cases
   - Test edge cases

## Examples

See `examples/` directory for complete examples of custom components.

