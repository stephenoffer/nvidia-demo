"""Factory functions for creating pipeline components.

Provides factory functions for creating stages, datasources, and other
components with consistent initialization patterns.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Type

from pipeline.config import PipelineConfig
from pipeline.core.registry import (
    stage_registry,
    datasource_registry,
    loader_registry,
    validator_registry,
)
from pipeline.stages.base import PipelineStage, ProcessorBase, ValidatorBase

logger = logging.getLogger(__name__)


def create_stage(
    stage_type: str,
    config: Optional[PipelineConfig] = None,
    **kwargs: Any,
) -> Optional[PipelineStage]:
    """Create a pipeline stage instance.
    
    Args:
        stage_type: Stage type name
        config: Pipeline configuration (optional)
        **kwargs: Additional stage-specific arguments
    
    Returns:
        Stage instance or None if not found
    """
    # Try registry first
    stage = stage_registry.create(stage_type, config=config, **kwargs)
    if stage is not None:
        return stage
    
    # Fall back to direct imports for built-in stages
    # Lazy import to avoid circular dependencies
    try:
        if stage_type == "temporal_alignment":
            from pipeline.stages.temporal_alignment import TemporalAlignmentStage
            return TemporalAlignmentStage(**kwargs)
        elif stage_type == "episode_detector":
            from pipeline.stages.episode_detector import EpisodeBoundaryDetector
            return EpisodeBoundaryDetector(**kwargs)
        elif stage_type == "video_processor":
            from pipeline.stages.video import VideoProcessor
            return VideoProcessor(config=config, **kwargs)
        elif stage_type == "text_processor":
            from pipeline.stages.text import TextProcessor
            return TextProcessor(**kwargs)
        elif stage_type == "sensor_processor":
            from pipeline.stages.sensor import SensorProcessor
            return SensorProcessor(**kwargs)
    except ImportError as e:
        logger.error(f"Failed to import stage '{stage_type}': {e}")
        return None
    
    return None


def create_validator(
    validator_type: str,
    reject_invalid: bool = False,
    **kwargs: Any,
) -> Optional[ValidatorBase]:
    """Create a validator stage instance.
    
    Args:
        validator_type: Validator type name (e.g., 'completeness', 'physics')
        reject_invalid: Whether to reject invalid items
        **kwargs: Additional validator-specific arguments
    
    Returns:
        Validator instance or None if not found
    """
    # Try registry first
    validator = validator_registry.create(validator_type, reject_invalid=reject_invalid, **kwargs)
    if validator is not None:
        return validator
    
    # Fall back to direct imports
    try:
        if validator_type == "completeness":
            from pipeline.stages.completeness_validator import CompletenessValidator
            return CompletenessValidator(reject_invalid=reject_invalid, **kwargs)
        elif validator_type == "physics":
            from pipeline.stages.physics_validator import PhysicsValidator
            return PhysicsValidator(reject_invalid=reject_invalid, **kwargs)
        elif validator_type == "cross_modal":
            from pipeline.stages.cross_modal_validator import CrossModalValidator
            return CrossModalValidator(reject_invalid=reject_invalid, **kwargs)
    except ImportError as e:
        logger.error(f"Failed to import validator '{validator_type}': {e}")
        return None
    
    return None


def create_datasource(
    datasource_type: str,
    paths: Any,
    **kwargs: Any,
) -> Any:
    """Create a datasource instance.
    
    Args:
        datasource_type: Datasource type name (e.g., 'groot', 'ros2bag')
        paths: File path(s) or directory path(s)
        **kwargs: Additional datasource-specific arguments
    
    Returns:
        Datasource instance or None if not found
    """
    # Try registry first
    datasource = datasource_registry.create(datasource_type, paths=paths, **kwargs)
    if datasource is not None:
        return datasource
    
    # Fall back to direct imports
    try:
        if datasource_type == "groot":
            from pipeline.datasources.groot import GR00TDatasource
            return GR00TDatasource(paths=paths, **kwargs)
        elif datasource_type == "ros2bag":
            from pipeline.datasources.ros2bag import ROS2BagDatasource
            return ROS2BagDatasource(paths=paths, **kwargs)
        elif datasource_type == "rosbag":
            from pipeline.datasources.rosbag import ROSBagDatasource
            return ROSBagDatasource(paths=paths, **kwargs)
        elif datasource_type == "hdf5":
            from pipeline.datasources.hdf5 import HDF5Datasource
            return HDF5Datasource(paths=paths, **kwargs)
    except ImportError as e:
        logger.error(f"Failed to import datasource '{datasource_type}': {e}")
        return None
    
    return None


def create_loader(
    loader_type: str,
    config: Optional[PipelineConfig] = None,
    **kwargs: Any,
) -> Any:
    """Create a loader instance.
    
    Args:
        loader_type: Loader type name (e.g., 'multimodal', 'isaac_lab')
        config: Pipeline configuration (optional)
        **kwargs: Additional loader-specific arguments
    
    Returns:
        Loader instance or None if not found
    """
    # Try registry first
    loader = loader_registry.create(loader_type, config=config, **kwargs)
    if loader is not None:
        return loader
    
    # Fall back to direct imports
    try:
        if loader_type == "multimodal":
            from pipeline.loaders.multimodal import MultimodalLoader
            return MultimodalLoader(config=config or PipelineConfig(), **kwargs)
        elif loader_type == "isaac_lab":
            from pipeline.integrations.isaac_lab import IsaacLabLoader
            return IsaacLabLoader(**kwargs)
        elif loader_type == "cosmos":
            from pipeline.integrations.cosmos import CosmosDreamsLoader
            return CosmosDreamsLoader(**kwargs)
    except ImportError as e:
        logger.error(f"Failed to import loader '{loader_type}': {e}")
        return None
    
    return None


__all__ = [
    "create_stage",
    "create_validator",
    "create_datasource",
    "create_loader",
]

