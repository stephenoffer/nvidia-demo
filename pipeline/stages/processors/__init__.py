"""Data processing and transformation stages."""

from pipeline.stages.processors.data_transformer import DataAggregator, DataTransformer
from pipeline.stages.processors.episode_detector import EpisodeBoundaryDetector
from pipeline.stages.processors.feature_engineering import FeatureEngineeringStage
from pipeline.stages.processors.instruction_grounding import InstructionGroundingStage
from pipeline.stages.processors.modality_dropout import ModalityDropoutStage
from pipeline.stages.processors.sensor import SensorProcessor
from pipeline.stages.processors.sequence_normalizer import SequenceNormalizer
from pipeline.stages.processors.temporal_alignment import TemporalAlignmentStage
from pipeline.stages.processors.temporal_resampler import TemporalResampler
from pipeline.stages.processors.text import TextProcessor
from pipeline.stages.processors.transition_alignment import TransitionAlignmentStage
from pipeline.stages.processors.video import VideoProcessor

__all__ = [
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
]

