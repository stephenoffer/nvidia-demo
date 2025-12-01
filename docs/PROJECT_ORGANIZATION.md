# Project Organization Guide

This document describes the organization and structure of the multimodal data pipeline project.

## Overview

This project follows Python best practices for package organization and structure.

## Naming Conventions

### Project vs Package vs Repository

- **Repository Name**: `nvidia-demo` (Git repository name)
- **Project Name**: `multimodal-data-pipeline` (PyPI package name, used in `pyproject.toml`)
- **Package Name**: `pipeline` (Python import name, directory name)

### Import Structure

```python
# Correct imports
from pipeline import MultimodalPipeline, PipelineConfig
from pipeline.core import PipelineExecutor
from pipeline.stages import TemporalAlignmentStage

# Avoid deprecated imports
from pipeline.core import MultimodalPipeline  # Deprecated, use pipeline.core.orchestrator
```

## Directory Structure

### Root Level

Only essential files should be at the root:
- Configuration: `pyproject.toml`, `setup.py`, `requirements*.txt`
- Documentation: `README.md`, `CHANGELOG.md`, `CONTRIBUTING.md`, `LICENSE`
- Build: `Makefile`, `Dockerfile`, `docker-compose.yml`
- Project structure: `PROJECT_STRUCTURE.md`

### Package Directory (`pipeline/`)

Organized by functionality:
- `api/` - Public API surface
- `core/` - Core pipeline components
- `datasources/` - Data source implementations
- `stages/` - Processing stages
- `integrations/` - External library integrations
- `utils/` - Utility modules

### Documentation (`docs/`)

Organized by topic:
- `api/` - API reference
- `architecture/` - System architecture
- `deployment/` - Deployment guides
- `guides/` - User guides
- `operations/` - Operations documentation

### Examples (`examples/`)

All example scripts and demos:
- Integration examples
- Usage examples
- Performance examples
- Visualization examples

### Tests (`tests/`)

Organized by test type:
- `unit/` - Unit tests
- `integration/` - Integration tests
- `benchmarks/` - Performance benchmarks

## File Naming

### Python Files
- Use `snake_case` for module names
- Use `PascalCase` for class names
- Use `snake_case` for function/variable names

### Configuration Files
- Use `kebab-case` for YAML files: `kubernetes-production.yaml`
- Use `snake_case` for Python config: `pytest.ini`, `setup.py`

### Documentation Files
- Use `UPPERCASE` for main docs: `README.md`, `CHANGELOG.md`
- Use `UPPERCASE` for section docs: `PROJECT_STRUCTURE.md`
- Use `Title Case` for topic docs: `Quick Start Guide.md`

## Import Organization

### Standard Library Order
1. Standard library imports
2. Third-party imports
3. Local application imports

### Example
```python
# Standard library
import logging
from pathlib import Path
from typing import Any, Dict

# Third-party
import ray
from ray.data import Dataset
import pandas as pd

# Local
from pipeline.config import PipelineConfig
from pipeline.core import PipelineExecutor
```

## Module Organization

### Single Responsibility
Each module should have a single, clear purpose:
- `temporal_alignment.py` - Temporal alignment only
- `episode_detector.py` - Episode detection only
- `gpu_dedup.py` - GPU deduplication orchestration

### Module Size
- Aim for <500 lines per module
- Split large modules into focused submodules
- Use subpackages for related functionality

## Configuration Management

### Environment-Specific Configs
- Use `kustomize/` for Kubernetes environment configs
- Use `helm/values.yaml` for Helm chart defaults
- Use `.env.example` as template

### Code Configuration
- `pipeline/config.py` - Main configuration class
- `pipeline/config/` - Configuration submodules
- Environment variables for runtime config

## Testing Organization

### Test Structure
```
tests/
├── unit/              # Fast, isolated tests
├── integration/       # End-to-end tests
├── benchmarks/        # Performance tests
└── fixtures/          # Test data and fixtures
```

### Test Naming
- Test files: `test_*.py` or `*_test.py`
- Test classes: `Test*`
- Test functions: `test_*`

## Documentation Organization

### User-Facing Docs
- `README.md` - Project overview
- `docs/guides/` - How-to guides
- `docs/api/` - API reference

### Developer Docs
- `CONTRIBUTING.md` - Contribution guide
- `PROJECT_STRUCTURE.md` - Structure documentation
- `docs/architecture/` - Architecture docs

### Operations Docs
- `docs/operations/` - Runbooks and troubleshooting
- `docs/deployment/` - Deployment guides

## Best Practices

1. **Keep root clean** - Only essential files at root
2. **Group related files** - Use subdirectories for organization
3. **Consistent naming** - Follow conventions throughout
4. **Clear imports** - Use absolute imports, avoid relative
5. **Document structure** - Keep PROJECT_STRUCTURE.md updated
6. **Version control** - Use .gitignore appropriately
7. **Configuration** - Centralize config management

## Migration Checklist

When reorganizing:
- [ ] Update all imports
- [ ] Update documentation references
- [ ] Update CI/CD paths
- [ ] Update scripts that reference moved files
- [ ] Test imports work correctly
- [ ] Update PROJECT_STRUCTURE.md
- [ ] Verify tests still pass

