"""Processing stages for multimodal data.

This module provides a comprehensive set of processing stages organized by category:
- Validators: Data validation stages
- Processors: Data transformation stages
- Analyzers: Data analysis stages
- Filters: Data filtering stages
- Inference: Model inference stages
"""

# Import from organized subdirectories
from pipeline.stages.validators import (
    CompletenessValidator,
    CrossModalValidator,
    PhysicsValidator,
    SchemaValidator,
)

from pipeline.stages.processors import (
    DataAggregator,
    DataTransformer,
    EpisodeBoundaryDetector,
    FeatureEngineeringStage,
    InstructionGroundingStage,
    ModalityDropoutStage,
    SensorProcessor,
    SequenceNormalizer,
    TemporalAlignmentStage,
    TemporalResampler,
    TextProcessor,
    TransitionAlignmentStage,
    VideoProcessor,
)

from pipeline.stages.analyzers import (
    AnomalyDetector,
    DataProfiler,
    DataShardingStage,
    DistributionAnalyzer,
    DriftDetector,
    GPUAnalyticsStage,
    MultimodalFeatureExtractor,
    DataQualityScorer,
)

from pipeline.stages.filters import QualityFilter

from pipeline.stages.inference import BatchInferenceStage

# Re-export all for backward compatibility
__all__ = [
    # Validators
    "CompletenessValidator",
    "CrossModalValidator",
    "PhysicsValidator",
    "SchemaValidator",
    # Processors
    "DataAggregator",
    "DataTransformer",
    "EpisodeBoundaryDetector",
    "FeatureEngineeringStage",
    "InstructionGroundingStage",
    "ModalityDropoutStage",
    "SensorProcessor",
    "SequenceNormalizer",
    "TemporalAlignmentStage",
    "TemporalResampler",
    "TextProcessor",
    "TransitionAlignmentStage",
    "VideoProcessor",
    # Analyzers
    "AnomalyDetector",
    "DataProfiler",
    "DataShardingStage",
    "DistributionAnalyzer",
    "DriftDetector",
    "GPUAnalyticsStage",
    "MultimodalFeatureExtractor",
    "DataQualityScorer",
    # Filters
    "QualityFilter",
    # Inference
    "BatchInferenceStage",
]

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
    "FeatureEngineeringStage",
    "DataTransformer",
    "DataAggregator",
    # Analyzers
    "AnomalyDetector",
    "DataShardingStage",
    "DistributionAnalyzer",
    "GPUAnalyticsStage",
    "MultimodalFeatureExtractor",
    "DataQualityScorer",
    "DataProfiler",
    "DriftDetector",
    "SchemaValidator",
    "BatchInferenceStage",
    # Filters
    "QualityFilter",
    "ModalityDropoutStage",
]
