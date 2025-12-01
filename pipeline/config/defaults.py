"""Configuration defaults factory.

Provides factory functions to create default configuration dictionaries
using constants from pipeline.utils.constants. Ensures all defaults
are centralized and repeatable.
"""

from __future__ import annotations

from typing import Any, Dict

from pipeline.utils.constants import (
    _DEFAULT_CHECKPOINT_INTERVAL,
    _DEFAULT_COMPRESSION,
    _DEFAULT_DEDUP_METHOD,
    _DEFAULT_FRAME_RATE,
    _DEFAULT_LANGUAGE,
    _DEFAULT_MAX_DURATION,
    _DEFAULT_MAX_LENGTH,
    _DEFAULT_MIN_LENGTH,
    _DEFAULT_NUM_ROWS_PER_FILE,
    _DEFAULT_PREFETCH_BUFFER_SIZE,
    _DEFAULT_RESOLUTION,
    _DEFAULT_SAMPLE_RATE,
    _DEFAULT_SIMILARITY_THRESHOLD,
    _DEFAULT_GPUS,
    _DEFAULT_RAY_DASHBOARD_PORT,
)


def get_default_video_config() -> Dict[str, Any]:
    """Get default video processing configuration.
    
    Returns:
        Dictionary with default video config values from constants
    """
    return {
        "extract_frames": True,
        "frame_rate": _DEFAULT_FRAME_RATE,
        "resolution": _DEFAULT_RESOLUTION,
        "max_duration": _DEFAULT_MAX_DURATION,
    }


def get_default_text_config() -> Dict[str, Any]:
    """Get default text processing configuration.
    
    Returns:
        Dictionary with default text config values from constants
    """
    return {
        "min_length": _DEFAULT_MIN_LENGTH,
        "max_length": _DEFAULT_MAX_LENGTH,
        "remove_boilerplate": True,
        "language": _DEFAULT_LANGUAGE,
    }


def get_default_sensor_config() -> Dict[str, Any]:
    """Get default sensor data configuration.
    
    Returns:
        Dictionary with default sensor config values from constants
    """
    return {
        "sample_rate": _DEFAULT_SAMPLE_RATE,
        "normalize": True,
        "remove_outliers": True,
    }


def get_default_resource_config() -> Dict[str, Any]:
    """Get default resource allocation configuration.
    
    Returns:
        Dictionary with default resource config values
    """
    return {
        "enable_autoscaling": True,
        "min_workers": 1,
        "max_workers": 100,
    }


def get_default_ray_data_config() -> Dict[str, Any]:
    """Get default Ray Data performance configuration.
    
    Returns:
        Dictionary with default Ray Data config values
    """
    return {
        "override_num_blocks": None,
        "read_num_cpus": 1.0,
        "read_num_gpus": None,
        "read_memory": None,
        "read_concurrency": None,
        "min_block_size": None,
        "max_block_size": None,
        "shuffle_max_block_size": None,
        "preserve_order": False,
        "locality_with_output": False,
        "execution_cpu": None,
        "execution_gpu": None,
        "execution_object_store_memory": None,
    }


def get_default_gpu_analytics_config() -> Dict[str, Any]:
    """Get default GPU analytics configuration.
    
    Returns:
        Dictionary with default GPU analytics config values from constants
    """
    return {
        "enabled": False,
        "target_columns": [],
        "normalize": True,
        "metrics": ["mean", "std"],
        "num_gpus": _DEFAULT_GPUS,
    }


def get_default_output_config() -> Dict[str, Any]:
    """Get default output configuration.
    
    Returns:
        Dictionary with default output config values from constants
    """
    from pipeline.utils.constants import (
        _DEFAULT_COMPRESSION,
        _DEFAULT_NUM_ROWS_PER_FILE,
    )
    
    return {
        "compression": _DEFAULT_COMPRESSION,
        "compression_level": None,
        "num_rows_per_file": _DEFAULT_NUM_ROWS_PER_FILE,
        "partition_by": None,
        "enable_compression": True,
    }


def get_default_pipeline_config() -> Dict[str, Any]:
    """Get default pipeline configuration dictionary.
    
    Returns:
        Dictionary with all default config values using constants
    """
    from pipeline.utils.constants import (
        _DEFAULT_BATCH_SIZE,
        _DEFAULT_SIMILARITY_THRESHOLD,
    )
    
    return {
        "enable_gpu_dedup": True,
        "num_gpus": _DEFAULT_GPUS,
        "num_cpus": None,
        "batch_size": None,
        "streaming": True,
        "checkpoint_interval": _DEFAULT_CHECKPOINT_INTERVAL,
        "checkpoint_dir": None,
        "dedup_method": _DEFAULT_DEDUP_METHOD,
        "prefetch_buffer_size": _DEFAULT_PREFETCH_BUFFER_SIZE,
        "similarity_threshold": _DEFAULT_SIMILARITY_THRESHOLD,
        "enable_observability": True,
        "enable_prometheus": True,
        "prometheus_port": _DEFAULT_RAY_DASHBOARD_PORT,
        "enable_grafana": True,
        "compute_mode": "auto",
        "random_seed": None,
        "enable_lineage_tracking": True,
        "video_config": get_default_video_config(),
        "text_config": get_default_text_config(),
        "sensor_config": get_default_sensor_config(),
        "resource_config": get_default_resource_config(),
        "ray_data_config": get_default_ray_data_config(),
        "gpu_analytics_config": get_default_gpu_analytics_config(),
        "gpu_memory_pool_size": None,
        "enable_rmm_pool": True,
        "output_config": get_default_output_config(),
        "omniverse_paths": None,
        "cosmos_paths": None,
        "enable_omniverse_export": False,
        "omniverse_export_path": None,
    }

