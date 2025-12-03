"""GR00T Model Training - Core Training Functionality.

This module contains the core training functionality for the GR00T
Vision-Language-Action (VLA) foundation model, extracted from the
example training script for reuse across the codebase.
"""

from pipeline.training.groot.config import (
    DEFAULT_ACTION_DIM,
    DEFAULT_ATTENTION_HEADS,
    DEFAULT_ATTENTION_LAYERS,
    DEFAULT_BETA_END,
    DEFAULT_BETA_START,
    DEFAULT_DROPOUT,
    DEFAULT_GRADIENT_CLIP_NORM,
    DEFAULT_HIDDEN_DIM,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_LANGUAGE_DIM,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MAX_SEQ_LEN,
    DEFAULT_NUM_DIFFUSION_STEPS,
    DEFAULT_VISION_DIM,
    DEFAULT_VOCAB_SIZE,
    DEFAULT_WEIGHT_DECAY,
    GrootTrainingConfig,
)
from pipeline.training.groot.data import (
    get_training_dataset,
    preprocess_batch,
    validate_batch,
)
from pipeline.training.groot.losses import compute_diffusion_loss
from pipeline.training.groot.model import (
    CrossModalAttention,
    DiffusionNoiseScheduler,
    DiffusionUNet,
    GrootSystem1,
    GrootSystem2,
    GrootVLA,
    LanguageTransformerEncoder,
    MultiHeadAttention,
    SinusoidalPositionalEncoding,
    TransformerBlock,
    VisionTransformerEncoder,
)
from pipeline.training.groot.schedulers import get_learning_rate_scheduler
from pipeline.training.groot.trainer import evaluate_model, train_epoch, train_func

__all__ = [
    # Config
    "GrootTrainingConfig",
    "DEFAULT_VISION_DIM",
    "DEFAULT_LANGUAGE_DIM",
    "DEFAULT_HIDDEN_DIM",
    "DEFAULT_ACTION_DIM",
    "DEFAULT_VOCAB_SIZE",
    "DEFAULT_MAX_SEQ_LEN",
    "DEFAULT_IMAGE_SIZE",
    "DEFAULT_NUM_DIFFUSION_STEPS",
    "DEFAULT_LEARNING_RATE",
    "DEFAULT_WEIGHT_DECAY",
    "DEFAULT_BETA_START",
    "DEFAULT_BETA_END",
    "DEFAULT_GRADIENT_CLIP_NORM",
    "DEFAULT_DROPOUT",
    "DEFAULT_ATTENTION_HEADS",
    "DEFAULT_ATTENTION_LAYERS",
    # Model
    "GrootVLA",
    "GrootSystem1",
    "GrootSystem2",
    "VisionTransformerEncoder",
    "LanguageTransformerEncoder",
    "CrossModalAttention",
    "DiffusionNoiseScheduler",
    "DiffusionUNet",
    "TransformerBlock",
    "MultiHeadAttention",
    "SinusoidalPositionalEncoding",
    # Data
    "get_training_dataset",
    "preprocess_batch",
    "validate_batch",
    # Losses
    "compute_diffusion_loss",
    # Schedulers
    "get_learning_rate_scheduler",
    # Trainer
    "train_func",
    "train_epoch",
    "evaluate_model",
]

