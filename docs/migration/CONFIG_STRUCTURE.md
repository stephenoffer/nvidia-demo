# Configuration Structure Migration Guide

## Overview

The configuration structure has been reorganized to be more maintainable while preserving backward compatibility.

## Structure

```
pipeline/
├── config.py              # Main PipelineConfig class (backward compatibility)
└── config/
    ├── __init__.py        # Re-exports PipelineConfig and RayDataConfig
    └── ray_data.py        # RayDataConfig for Ray Data settings
```

## Import Paths

### Recommended (Public API)
```python
from pipeline import PipelineConfig, MultimodalPipeline
```

### Also Supported
```python
from pipeline.config import PipelineConfig, RayDataConfig
from pipeline.config.ray_data import RayDataConfig
```

### Deprecated (but still works)
```python
from pipeline.config import PipelineConfig  # Works via config/__init__.py
```

## Why This Structure?

1. **Backward Compatibility**: `pipeline/config.py` maintains existing imports
2. **Future Extensibility**: `pipeline/config/` directory allows adding more config modules
3. **Clear Organization**: Related config classes are grouped together
4. **No Breaking Changes**: All existing imports continue to work

## Future Config Modules

New configuration modules can be added to `pipeline/config/`:

```
pipeline/config/
├── __init__.py
├── ray_data.py
├── gpu.py          # GPU configuration (future)
├── storage.py      # Storage configuration (future)
└── monitoring.py   # Monitoring configuration (future)
```

## Migration

No migration needed! All existing imports continue to work.

If you want to use the new structure:

```python
# Old (still works)
from pipeline.config import PipelineConfig

# New (recommended)
from pipeline import PipelineConfig
```

