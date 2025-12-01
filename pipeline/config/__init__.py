"""Configuration modules for the pipeline.

This package provides configuration classes for the multimodal data pipeline.
The main PipelineConfig class is in the parent config.py file to maintain
backward compatibility with existing imports.

Structure:
- pipeline/config.py: Main PipelineConfig class (backward compatibility)
- pipeline/config/ray_data.py: RayDataConfig for Ray Data settings
- pipeline/config/defaults.py: Default configuration factories
- pipeline/config/factory.py: Configuration builder and factory functions
"""

# Import from subdirectory first (this is safe and doesn't cause circular import)
from pipeline.config.ray_data import RayDataConfig
from pipeline.config.defaults import (
    get_default_gpu_analytics_config,
    get_default_output_config,
    get_default_pipeline_config,
    get_default_ray_data_config,
    get_default_resource_config,
    get_default_sensor_config,
    get_default_text_config,
    get_default_video_config,
)
from pipeline.config.factory import (
    ConfigBuilder,
    create_config,
    create_config_from_dict,
)

# Import PipelineConfig from parent config.py file
# This maintains backward compatibility while allowing the config/ directory
# to exist for future configuration modules
import importlib.util
from pathlib import Path

_config_file_path = Path(__file__).parent.parent / "config.py"
_spec = importlib.util.spec_from_file_location("pipeline._config_loader", _config_file_path)
_config_loader = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_config_loader)

# Extract PipelineConfig class
PipelineConfig = _config_loader.PipelineConfig

__all__ = [
    "PipelineConfig",
    "RayDataConfig",
    "ConfigBuilder",
    "create_config",
    "create_config_from_dict",
    "get_default_pipeline_config",
    "get_default_video_config",
    "get_default_text_config",
    "get_default_sensor_config",
    "get_default_resource_config",
    "get_default_ray_data_config",
    "get_default_gpu_analytics_config",
    "get_default_output_config",
]
