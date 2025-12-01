"""Processing stages for multimodal data.

This module provides a comprehensive set of processing stages organized by category:
- Validators: Data validation stages
- Processors: Data transformation stages
- Analyzers: Data analysis stages
- Filters: Data filtering stages
"""

# Validators
from pipeline.stages.completeness_validator import CompletenessValidator
from pipeline.stages.cross_modal_validator import CrossModalValidator
from pipeline.stages.physics_validator import PhysicsValidator

# Processors
from pipeline.stages.episode_detector import EpisodeBoundaryDetector
from pipeline.stages.instruction_grounding import InstructionGroundingStage
from pipeline.stages.sequence_normalizer import SequenceNormalizer
from pipeline.stages.sensor import SensorProcessor
from pipeline.stages.temporal_alignment import TemporalAlignmentStage
from pipeline.stages.temporal_resampler import TemporalResampler
from pipeline.stages.text import TextProcessor
from pipeline.stages.transition_alignment import TransitionAlignmentStage
from pipeline.stages.video import VideoProcessor

# Analyzers
from pipeline.stages.anomaly_detector import AnomalyDetector
from pipeline.stages.data_sharding import DataShardingStage
from pipeline.stages.distribution_analyzer import DistributionAnalyzer
from pipeline.stages.gpu_analytics import GPUAnalyticsStage
from pipeline.stages.multimodal_features import MultimodalFeatureExtractor
from pipeline.stages.quality_scorer import DataQualityScorer

# Filters
from pipeline.stages.filters import QualityFilter
from pipeline.stages.modality_dropout import ModalityDropoutStage

# Base classes
from pipeline.stages.base import PipelineStage, ProcessorBase, ValidatorBase

__all__ = [
    # Base classes
    "PipelineStage",
    "ProcessorBase",
    "ValidatorBase",
    # Validators
    "CompletenessValidator",
    "CrossModalValidator",
    "PhysicsValidator",
    # Processors
    "EpisodeBoundaryDetector",
    "InstructionGroundingStage",
    "SequenceNormalizer",
    "SensorProcessor",
    "TemporalAlignmentStage",
    "TemporalResampler",
    "TextProcessor",
    "TransitionAlignmentStage",
    "VideoProcessor",
    # Analyzers
    "AnomalyDetector",
    "DataShardingStage",
    "DistributionAnalyzer",
    "GPUAnalyticsStage",
    "MultimodalFeatureExtractor",
    "DataQualityScorer",
    # Filters
    "QualityFilter",
    "ModalityDropoutStage",
]
