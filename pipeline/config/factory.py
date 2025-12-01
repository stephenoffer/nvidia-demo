"""Configuration factory for creating PipelineConfig instances.

Provides factory functions and builders for creating configuration objects
with consistent defaults and validation. Ensures repeatability across the codebase.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Dict, List, Optional

from pipeline.config.defaults import get_default_pipeline_config
from pipeline.exceptions import ConfigurationError

# Import PipelineConfig directly from parent config.py to avoid circular import
_config_file_path = Path(__file__).parent.parent / "config.py"
_spec = importlib.util.spec_from_file_location("pipeline._config_loader", _config_file_path)
_config_loader = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_config_loader)
PipelineConfig = _config_loader.PipelineConfig


class ConfigBuilder:
    """Builder pattern for PipelineConfig creation.
    
    Provides fluent API for building configurations with validation
    and consistent defaults.
    """
    
    def __init__(self) -> None:
        """Initialize config builder with defaults."""
        self._config_dict = get_default_pipeline_config()
    
    def with_input_paths(self, paths: List[str]) -> "ConfigBuilder":
        """Set input paths.
        
        Args:
            paths: List of input data paths
        
        Returns:
            Self for method chaining
        """
        self._config_dict["input_paths"] = paths
        return self
    
    def with_output_path(self, path: str) -> "ConfigBuilder":
        """Set output path.
        
        Args:
            path: Output path for curated data
        
        Returns:
            Self for method chaining
        """
        self._config_dict["output_path"] = path
        return self
    
    def with_gpus(self, num_gpus: int) -> "ConfigBuilder":
        """Set number of GPUs.
        
        Args:
            num_gpus: Number of GPUs to use
        
        Returns:
            Self for method chaining
        """
        self._config_dict["num_gpus"] = num_gpus
        return self
    
    def with_cpus(self, num_cpus: Optional[int]) -> "ConfigBuilder":
        """Set number of CPUs.
        
        Args:
            num_cpus: Number of CPUs to use (None = auto)
        
        Returns:
            Self for method chaining
        """
        self._config_dict["num_cpus"] = num_cpus
        return self
    
    def with_batch_size(self, batch_size: Optional[int]) -> "ConfigBuilder":
        """Set batch size.
        
        Args:
            batch_size: Batch size for processing (None = use default)
        
        Returns:
            Self for method chaining
        """
        self._config_dict["batch_size"] = batch_size
        return self
    
    def with_streaming(self, streaming: bool = True) -> "ConfigBuilder":
        """Enable or disable streaming mode.
        
        Args:
            streaming: Whether to enable streaming
        
        Returns:
            Self for method chaining
        """
        self._config_dict["streaming"] = streaming
        return self
    
    def with_compute_mode(self, mode: str) -> "ConfigBuilder":
        """Set compute mode.
        
        Args:
            mode: Compute mode ("auto", "cpu", "gpu")
        
        Returns:
            Self for method chaining
        """
        self._config_dict["compute_mode"] = mode
        return self
    
    def update_config(self, updates: Dict[str, Any]) -> "ConfigBuilder":
        """Update configuration with dictionary of values.
        
        Args:
            updates: Dictionary of configuration updates
        
        Returns:
            Self for method chaining
        """
        self._config_dict.update(updates)
        return self
    
    def build(self, validate: bool = True) -> PipelineConfig:
        """Build PipelineConfig instance.
        
        Args:
            validate: Whether to validate configuration before returning
        
        Returns:
            PipelineConfig instance
        
        Raises:
            ConfigurationError: If validation fails
        """
        config = PipelineConfig(**self._config_dict)
        if validate:
            config.validate()
        return config


def create_config(
    input_paths: List[str],
    output_path: str,
    **kwargs: Any,
) -> PipelineConfig:
    """Create PipelineConfig with specified inputs and optional overrides.
    
    Convenience function for creating configurations with consistent defaults.
    
    Args:
        input_paths: List of input data paths
        output_path: Output path for curated data
        **kwargs: Additional configuration overrides
    
    Returns:
        PipelineConfig instance with defaults applied
    
    Example:
        ```python
        config = create_config(
            input_paths=["s3://bucket/input/"],
            output_path="s3://bucket/output/",
            num_gpus=4,
            streaming=True,
        )
        ```
    """
    defaults = get_default_pipeline_config()
    defaults["input_paths"] = input_paths
    defaults["output_path"] = output_path
    defaults.update(kwargs)
    
    config = PipelineConfig(**defaults)
    config.validate()
    return config


def create_config_from_dict(config_dict: Dict[str, Any], validate: bool = True) -> PipelineConfig:
    """Create PipelineConfig from dictionary with defaults applied.
    
    Merges provided dictionary with defaults, ensuring all required fields
    are present and defaults are applied for missing optional fields.
    
    Args:
        config_dict: Configuration dictionary
        validate: Whether to validate configuration before returning
    
    Returns:
        PipelineConfig instance
    
    Raises:
        ConfigurationError: If validation fails and validate=True
    """
    defaults = get_default_pipeline_config()
    
    # Merge provided config with defaults (provided values take precedence)
    merged = {**defaults, **config_dict}
    
    # Handle nested dictionaries (merge instead of replace)
    for key in ["video_config", "text_config", "sensor_config", "resource_config",
                "ray_data_config", "gpu_analytics_config", "output_config"]:
        if key in config_dict and isinstance(config_dict[key], dict):
            merged[key] = {**defaults.get(key, {}), **config_dict[key]}
    
    config = PipelineConfig(**merged)
    if validate:
        config.validate()
    return config

