"""GR00T Training Configuration.

Configuration dataclass and constants for GR00T model training.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Constants - Based on GR00T N1 specifications from study guide
DEFAULT_VISION_DIM = 768  # Vision encoder output dimension (NVIDIA-Eagle based)
DEFAULT_LANGUAGE_DIM = 2048  # SmolLM-1.7B hidden dimension
DEFAULT_HIDDEN_DIM = 2048  # Fusion dimension
DEFAULT_ACTION_DIM = 25  # Typical humanoid robot DOF (20-30 range)
DEFAULT_VOCAB_SIZE = 50000  # ~50,000 tokens as per study guide
DEFAULT_MAX_SEQ_LEN = 2048  # Context length: 2048-4096 tokens (using lower end)
DEFAULT_IMAGE_SIZE = 224  # 224x224 or 336x336 (using 224)
DEFAULT_NUM_DIFFUSION_STEPS = 1000  # 1000 steps for training, 50-100 steps for inference
DEFAULT_LEARNING_RATE = 1e-4  # 1e-4 to 3e-4 range
DEFAULT_WEIGHT_DECAY = 1e-4  # 1e-4 as per study guide (not 0.01)
DEFAULT_BETA_START = 0.0001
DEFAULT_BETA_END = 0.02
DEFAULT_GRADIENT_CLIP_NORM = 1.0
DEFAULT_DROPOUT = 0.1  # 0.1-0.2 range (using 0.1)
DEFAULT_ATTENTION_HEADS = 16  # 16-32 range (using 16)
DEFAULT_ATTENTION_LAYERS = 24  # 24-32 for language model (using 24)


@dataclass
class GrootTrainingConfig:
    """Configuration for GR00T model training."""

    # Model architecture
    vision_dim: int = DEFAULT_VISION_DIM
    language_dim: int = DEFAULT_LANGUAGE_DIM
    hidden_dim: int = DEFAULT_HIDDEN_DIM
    action_dim: int = DEFAULT_ACTION_DIM
    vocab_size: int = DEFAULT_VOCAB_SIZE
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN
    image_size: int = DEFAULT_IMAGE_SIZE
    num_attention_heads: int = DEFAULT_ATTENTION_HEADS
    num_attention_layers: int = DEFAULT_ATTENTION_LAYERS
    dropout: float = DEFAULT_DROPOUT

    # Diffusion model
    num_diffusion_steps: int = DEFAULT_NUM_DIFFUSION_STEPS
    beta_start: float = DEFAULT_BETA_START
    beta_end: float = DEFAULT_BETA_END

    # Training
    batch_size: int = 256  # 256-512 for joint training (per study guide)
    learning_rate: float = DEFAULT_LEARNING_RATE
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    num_epochs: int = 100
    gradient_clip_norm: float = DEFAULT_GRADIENT_CLIP_NORM
    gradient_accumulation_steps: int = 4  # 4-8 steps per study guide
    warmup_steps: int = 1000
    use_mixed_precision: bool = True
    use_gradient_checkpointing: bool = True

    # Data
    curated_data_path: str = ""
    val_data_path: str = ""  # Separate validation dataset path
    train_split: float = 0.9
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    use_dali: bool = True  # Use NVIDIA DALI for GPU-accelerated data loading
    use_streaming: bool = True  # Use Ray Data streaming for large datasets
    use_gpu_object_store: bool = True  # Use GPU object store with RDMA (Ray 2.6+)

    # NVIDIA GPU optimizations
    use_cuda_graphs: bool = False  # CUDA graphs for repetitive kernels (PyTorch 2.0+)
    use_structured_sparsity: bool = False  # 2:4 structured sparsity (A100+)
    use_flash_attention: bool = True  # Flash Attention for memory efficiency
    enable_tensor_parallelism: bool = False  # Tensor parallelism for very large models
    use_zero_infinity: bool = False  # ZeRO-Infinity for massive models

    # Validation
    eval_interval: int = 1000
    save_interval: int = 5000
    early_stopping_patience: int = 10

    # Distributed training
    num_workers_distributed: int = 8
    use_gpu: bool = True
    use_fsdp: bool = False  # Fully Sharded Data Parallel (Ray Train 2.6+)
    fsdp_config: Optional[Dict[str, Any]] = field(
        default_factory=lambda: {
            "sharding_strategy": "FULL_SHARD",
            "cpu_offload": False,
            "mixed_precision": True,
        }
    )
    use_deepspeed: bool = False  # DeepSpeed ZeRO (for very large models)
    deepspeed_config: Optional[Dict[str, Any]] = field(
        default_factory=lambda: {
            "zero_optimization": {
                "stage": 2,
                "offload_optimizer": {"device": "cpu"},
            },
            "gradient_accumulation_steps": 1,
        }
    )

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.vision_dim <= 0:
            raise ValueError("vision_dim must be positive")
        if self.language_dim <= 0:
            raise ValueError("language_dim must be positive")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.action_dim <= 0:
            raise ValueError("action_dim must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0 < self.train_split < 1:
            raise ValueError("train_split must be between 0 and 1")
        if self.num_diffusion_steps <= 0:
            raise ValueError("num_diffusion_steps must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be between 0 and 1")

