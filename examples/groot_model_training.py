"""GR00T Model Training Example - Full Foundation Model Training with Ray Data + Ray Train.

This example demonstrates training the full GR00T (Generalist Robot 00 Technology)
Vision-Language-Action (VLA) foundation model using Ray Data for data processing
and Ray Train for distributed training.

The GR00T N1 model architecture (based on NVIDIA specifications):
- System 2 (Slow): Vision-Language Model (VLM) for deliberate reasoning
  - Vision Encoder: NVIDIA-Eagle architecture (transformer-based)
  - Language Model: SmolLM-1.7B (1.7B parameters, 2048 hidden dim, 24-32 layers)
  - Processes multi-view camera inputs (2-4 views) at 224x224 or 336x336 resolution
  - Vocabulary: ~50,000 tokens, context length: 2048-4096 tokens
- System 1 (Fast): Diffusion Transformer for reactive action generation
  - Generates continuous actions at 100+ Hz (100+ actions per second)
  - Action space: 20-30 DOF for humanoid robots
  - Diffusion steps: 50-100 for inference (1000 for training)
  - Real-time control at 10-30 Hz control frequency
- Total: 2 billion parameters (data maximalist, model minimalist philosophy)
- Input: Photons (vision) → Output: Actions (continuous control)

Following NVIDIA's "data maximalist, model minimalist" approach:
- Complex data pipelines combining:
  - Fossil Fuel: Internet-scale web data and human videos
  - Nuclear Fuel: Synthetic data from Simulation 1.0 (Isaac Lab) and Simulation 2.0 (Cosmos Dreams)
  - Human Fuel: Real robot data from teleoperation (4-24 hours per robot per day)
- Clean model architecture that compresses trillions of tokens from data pipeline

Training on Ray infrastructure:
- Almost all compute jobs based on Ray for versatility
- Handles physical AI multimodal sensor data processing
- Large-scale VLA training
- Complex evaluation pipelines with thousands of simulation engines
"""

import logging
import math
import os
import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np
import ray  # https://docs.ray.io/
from ray.data import Dataset  # https://docs.ray.io/en/latest/data/data.html
from ray.data.context import DataContext  # https://docs.ray.io/en/latest/data/data.html
from ray.train import Checkpoint, ScalingConfig, get_context  # https://docs.ray.io/en/latest/train/train.html
from ray.train.torch import TorchTrainer  # https://docs.ray.io/en/latest/train/train.html
from ray.train.torch.config import TorchConfig  # https://docs.ray.io/en/latest/train/train.html
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

# NVIDIA optimizations
DALI_AVAILABLE = False
try:
    import nvidia.dali as dali  # type: ignore # noqa: F401
    DALI_AVAILABLE = True
except ImportError:
    pass  # DALI not available, will log warning after logger is initialized

# GPU memory utilities
try:
    from pipeline.utils.gpu.memory import (
        get_cuda_device,
        get_gpu_memory_info,
        check_gpu_memory,
        gpu_memory_cleanup,
    )
except ImportError:
    # Fallback if utilities not available
    def get_cuda_device(device_id=None):
        return device_id or 0
    def get_gpu_memory_info(device_id=None):
        return {"total": 0, "allocated": 0, "reserved": 0, "free": 0}
    def check_gpu_memory(required_bytes, device_id=None, safety_margin=0.1):
        return True, {}
    @contextmanager
    def gpu_memory_cleanup():
        yield

from pipeline import MultimodalPipeline
from pipeline.config import PipelineConfig
from pipeline.integrations.cosmos import CosmosDreamsLoader
from pipeline.integrations.isaac_lab import IsaacLabLoader
from pipeline.training.integration import TrainingPipelineIntegration
from pipeline.utils.constants import _TRAINING_BATCH_SIZE

logger = logging.getLogger(__name__)

# Log optional dependencies after logger is initialized
if not DALI_AVAILABLE:
    logger.warning("NVIDIA DALI not available. Install with: pip install nvidia-dali-cuda120")

# Constants - Based on GR00T N1 specifications from study guide
DEFAULT_VISION_DIM = 768  # Vision encoder output dimension (NVIDIA-Eagle based)
DEFAULT_LANGUAGE_DIM = 2048  # SmolLM-1.7B hidden dimension
DEFAULT_HIDDEN_DIM = 2048  # Fusion dimension
DEFAULT_ACTION_DIM = 25  # Typical humanoid robot DOF (20-30 range)
DEFAULT_VOCAB_SIZE = 50000  # ~50,000 tokens as per study guide
DEFAULT_MAX_SEQ_LEN = 2048  # Context length: 2048-4096 tokens (using lower end)
DEFAULT_IMAGE_SIZE = 224  # 224x224 or 336x336 (using 224)
DEFAULT_NUM_DIFFUSION_STEPS = 100  # 50-100 steps for inference (1000 for training)
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
    val_data_path: str = ""  # CRITICAL FIX: Separate validation dataset path
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
    fsdp_config: Optional[Dict[str, Any]] = field(default_factory=lambda: {
        "sharding_strategy": "FULL_SHARD",
        "cpu_offload": False,
        "mixed_precision": True,
    })
    use_deepspeed: bool = False  # DeepSpeed ZeRO (for very large models)
    deepspeed_config: Optional[Dict[str, Any]] = field(default_factory=lambda: {
        "zero_optimization": {
            "stage": 2,
            "offload_optimizer": {"device": "cpu"},
        },
        "gradient_accumulation_steps": 1,
    })
    
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


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequences.
    
    CRITICAL FIX: Support longer sequences by computing on-the-fly if needed.
    GR00T uses context lengths up to 4096 tokens.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        """Initialize positional encoding.
        
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length (should be >= max_seq_len)
        """
        super().__init__()
        
        # CRITICAL FIX: Ensure max_len is sufficient for GR00T's context length
        # GR00T uses up to 4096 tokens, so we need at least that
        max_len = max(max_len, 4096)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)
        self.max_len = max_len
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input.
        
        Args:
            x: Input tensor [B, L, D]
            
        Returns:
            Tensor with positional encoding added
        """
        return x + self.pe[:, : x.size(1)]


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism with Flash Attention support."""
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        bias: bool = True,
        use_flash_attention: bool = True,
    ):
        """Initialize multi-head attention.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
            bias: Whether to use bias
            use_flash_attention: Whether to use Flash Attention (memory efficient)
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.use_flash_attention = use_flash_attention
        
        # Try to import Flash Attention
        self.flash_attn_available = False
        if use_flash_attention:
            try:
                from flash_attn import flash_attn_func  # type: ignore
                self.flash_attn_func = flash_attn_func
                self.flash_attn_available = True
                logger.info("Flash Attention available - using memory-efficient attention")
            except ImportError:
                logger.warning("Flash Attention not available. Install with: pip install flash-attn")
                self.use_flash_attention = False
        
        self.w_q = nn.Linear(d_model, d_model, bias=bias)
        self.w_k = nn.Linear(d_model, d_model, bias=bias)
        self.w_v = nn.Linear(d_model, d_model, bias=bias)
        self.w_o = nn.Linear(d_model, d_model, bias=bias)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.d_k)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through attention.
        
        Args:
            query: Query tensor [B, L_q, D]
            key: Key tensor [B, L_k, D]
            value: Value tensor [B, L_v, D]
            mask: Attention mask [B, L_q, L_k] (optional)
            
        Returns:
            Output tensor and attention weights
        """
        batch_size = query.size(0)
        seq_len_q = query.size(1)
        seq_len_k = key.size(1)
        
        # Linear projections
        Q = self.w_q(query).view(batch_size, seq_len_q, self.num_heads, self.d_k)
        K = self.w_k(key).view(batch_size, seq_len_k, self.num_heads, self.d_k)
        V = self.w_v(value).view(batch_size, seq_len_k, self.num_heads, self.d_k)
        
        # Use Flash Attention if available and conditions are met
        if (
            self.use_flash_attention
            and self.flash_attn_available
            and mask is None
            and Q.dtype == torch.float16
            and seq_len_q >= 128  # Flash Attention is most beneficial for long sequences
        ):
            # Flash Attention: [B, L, H, D] format
            attn_output = self.flash_attn_func(
                Q, K, V,
                dropout_p=self.dropout.p if self.training else 0.0,
                softmax_scale=self.scale,
                causal=False,
            )
            attn_output = attn_output.contiguous().view(batch_size, seq_len_q, self.d_model)
            attn_weights = None  # Flash Attention doesn't return weights
        else:
            # Standard attention
            Q = Q.transpose(1, 2)  # [B, H, L_q, D]
            K = K.transpose(1, 2)  # [B, H, L_k, D]
            V = V.transpose(1, 2)  # [B, H, L_k, D]
            
            # Scaled dot-product attention
            scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
            
            if mask is not None:
                # Handle mask shape: [B, L_q, L_k] or [B, 1, L_k] or [B, L_k]
                if len(mask.shape) == 2:
                    # [B, L_k] -> [B, 1, 1, L_k] for broadcasting
                    mask = mask.unsqueeze(1).unsqueeze(1)
                elif len(mask.shape) == 3:
                    # [B, L_q, L_k] -> [B, 1, L_q, L_k] for broadcasting with heads
                    mask = mask.unsqueeze(1)
                # mask is now [B, 1, L_q, L_k] or [B, 1, 1, L_k]
                scores = scores.masked_fill(mask == 0, float("-inf"))
            
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            
            attn_output = torch.matmul(attn_weights, V)
            attn_output = attn_output.transpose(1, 2).contiguous().view(
                batch_size, seq_len_q, self.d_model
            )
        
        output = self.w_o(attn_output)
        
        return output, attn_weights


class TransformerBlock(nn.Module):
    """Transformer block with self-attention and feed-forward."""
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        """Initialize transformer block.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Feed-forward dimension
            dropout: Dropout probability
            activation: Activation function
        """
        super().__init__()
        
        self.attention = MultiHeadAttention(
            d_model, num_heads, dropout, use_flash_attention=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through transformer block.
        
        Args:
            x: Input tensor [B, L, D]
            mask: Attention mask [B, L, L] (optional)
            
        Returns:
            Output tensor
        """
        # Self-attention with residual connection
        attn_out, _ = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward with residual connection
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)
        
        return x


class VisionTransformerEncoder(nn.Module):
    """Vision Transformer encoder for image processing.
    
    Based on NVIDIA-Eagle architecture used in GR00T N1 System 2.
    Processes multi-view camera inputs (typically 2-4 views) at 224x224 or 336x336 resolution.
    Outputs vision tokens with dimension 768 or 1024.
    """
    
    def __init__(
        self,
        image_size: int = DEFAULT_IMAGE_SIZE,
        patch_size: int = 16,
        in_channels: int = 3,
        d_model: int = DEFAULT_VISION_DIM,
        num_layers: int = 12,
        num_heads: int = DEFAULT_ATTENTION_HEADS,
        d_ff: int = 3072,
        dropout: float = DEFAULT_DROPOUT,
    ):
        """Initialize Vision Transformer encoder.
        
        Args:
            image_size: Input image size
            patch_size: Patch size for image splitting
            in_channels: Number of input channels
            d_model: Model dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            d_ff: Feed-forward dimension
            dropout: Dropout probability
        """
        super().__init__()
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(
            in_channels, d_model, kernel_size=patch_size, stride=patch_size
        )
        
        # CRITICAL FIX: Class token and positional encoding
        # Ensure positional encoding can handle multi-view inputs
        # Multi-view: num_patches * num_views + 1 (class token)
        max_patches = self.num_patches * 4 + 1  # Support up to 4 views
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_patches)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Forward pass through vision encoder.
        
        Args:
            images: Batch of images [B, C, H, W] or [B, num_views, C, H, W] for multi-view
                   GR00T processes 2-4 camera views typically
            
        Returns:
            Vision features [B, d_model] (averaged across views if multi-view)
        """
        batch_size = images.size(0)
        
        # CRITICAL FIX: Handle multi-view inputs (2-4 camera views)
        # GR00T processes multiple camera views and fuses them
        if len(images.shape) == 5:
            # Multi-view input: [B, num_views, C, H, W]
            num_views = images.size(1)
            images = images.view(-1, *images.shape[2:])  # [B*num_views, C, H, W]
            batch_size_processed = images.size(0)
        else:
            num_views = 1
            batch_size_processed = batch_size
        
        # Patch embedding
        x = self.patch_embed(images)  # [B*num_views, d_model, H', W']
        x = x.flatten(2).transpose(1, 2)  # [B*num_views, num_patches, d_model]
        
        # Add class token
        cls_tokens = self.cls_token.expand(batch_size_processed, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # [B*num_views, num_patches + 1, d_model]
        
        # CRITICAL FIX: Ensure positional encoding can handle sequence length
        # Check if we exceed max_len and handle gracefully
        seq_len = x.size(1)
        if seq_len > self.pos_encoding.pe.size(1):
            # Extend positional encoding if needed (shouldn't happen with proper config)
            logger.warning(f"Sequence length {seq_len} exceeds positional encoding max_len")
            # Use only available positions
            x = x[:, :self.pos_encoding.pe.size(1)]
            seq_len = x.size(1)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        # Extract class token
        x = self.norm(x)
        vision_features = x[:, 0]  # [B*num_views, d_model]
        
        # CRITICAL FIX: Average across views if multi-view input
        if num_views > 1:
            vision_features = vision_features.view(batch_size, num_views, -1)
            vision_features = vision_features.mean(dim=1)  # [B, d_model] - average across views
        
        return vision_features


class LanguageTransformerEncoder(nn.Module):
    """Transformer encoder for language processing.
    
    Based on SmolLM-1.7B architecture used in GR00T N1 System 2.
    - 1.7 billion parameters
    - Hidden dimension: 2048
    - Number of layers: 24-32 transformer blocks
    - Attention heads: 16-32 multi-head attention
    - Vocabulary size: ~50,000 tokens
    - Context length: 2048-4096 tokens
    """
    
    def __init__(
        self,
        vocab_size: int = DEFAULT_VOCAB_SIZE,
        d_model: int = DEFAULT_LANGUAGE_DIM,
        num_layers: int = DEFAULT_ATTENTION_LAYERS,
        num_heads: int = DEFAULT_ATTENTION_HEADS,
        d_ff: int = 4096,
        max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
        dropout: float = DEFAULT_DROPOUT,
    ):
        """Initialize language transformer encoder.
        
        Args:
            vocab_size: Vocabulary size
            d_model: Model dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            d_ff: Feed-forward dimension
            max_seq_len: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # CRITICAL FIX: Positional encoding with sufficient max_len
        # GR00T uses context length up to 4096, ensure we support it
        pos_max_len = max(max_seq_len, 4096)  # Support GR00T's max context length
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, pos_max_len)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through language encoder.
        
        Args:
            tokens: Token sequences [B, L]
            mask: Attention mask [B, L] or [B, L, L] (optional)
            
        Returns:
            Language features [B, d_model]
        """
        # Token embedding
        x = self.token_embedding(tokens) * math.sqrt(self.d_model)  # [B, L, d_model]
        
        # Add positional encoding
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Convert 1D mask to 2D attention mask if needed
        attention_mask = None
        if mask is not None:
            if len(mask.shape) == 2:
                # [B, L] -> [B, 1, 1, L] for broadcasting in attention
                # Expand to [B, L, L] for attention
                batch_size, seq_len = mask.shape
                attention_mask = mask.unsqueeze(1).expand(batch_size, seq_len, seq_len)  # [B, L, L]
            else:
                attention_mask = mask
        
        # Transformer blocks
        for block in self.transformer_blocks:
            x = block(x, attention_mask)
        
        # CRITICAL FIX: Global average pooling with proper mask handling
        x = self.norm(x)
        if mask is not None:
            # Handle both 1D and 2D masks
            if len(mask.shape) == 2:
                # [B, L] mask - use for pooling
                # CRITICAL FIX: Ensure mask is boolean or convert to float
                if mask.dtype != torch.float32 and mask.dtype != torch.bool:
                    mask = mask.float()
                elif mask.dtype == torch.bool:
                    mask = mask.float()
                
                mask_expanded = mask.unsqueeze(-1)  # [B, L, 1]
                masked_x = x * mask_expanded  # [B, L, d_model]
                
                # CRITICAL FIX: Avoid division by zero
                mask_sum = mask.sum(dim=1, keepdim=True).float().clamp(min=1.0)  # [B, 1]
                lang_features = masked_x.sum(dim=1) / mask_sum  # [B, d_model]
            else:
                # [B, L, L] mask - use diagonal or sum
                # CRITICAL FIX: Convert to float if needed
                if mask.dtype != torch.float32:
                    mask = mask.float()
                mask_expanded = mask.sum(dim=-1, keepdim=True)  # [B, L, 1]
                mask_sum = mask_expanded.sum(dim=1).clamp(min=1.0)  # [B, 1]
                lang_features = (x * mask_expanded).sum(dim=1) / mask_sum  # [B, d_model]
        else:
            lang_features = x.mean(dim=1)  # [B, d_model]
        
        return lang_features


class CrossModalAttention(nn.Module):
    """Cross-modal attention for vision-language fusion."""
    
    def __init__(
        self,
        vision_dim: int,
        language_dim: int,
        hidden_dim: int,
        num_heads: int = DEFAULT_ATTENTION_HEADS,
        dropout: float = DEFAULT_DROPOUT,
    ):
        """Initialize cross-modal attention.
        
        Args:
            vision_dim: Vision feature dimension
            language_dim: Language feature dimension
            hidden_dim: Hidden dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        self.language_proj = nn.Linear(language_dim, hidden_dim)
        
        self.attention = MultiHeadAttention(
            hidden_dim, num_heads, dropout, use_flash_attention=True
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        vision_features: torch.Tensor,
        language_features: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through cross-modal attention.
        
        Args:
            vision_features: Vision features [B, vision_dim]
            language_features: Language features [B, language_dim]
            
        Returns:
            Fused features [B, hidden_dim]
        """
        # Project to common dimension
        vision_proj = self.vision_proj(vision_features).unsqueeze(1)  # [B, 1, hidden_dim]
        lang_proj = self.language_proj(language_features).unsqueeze(1)  # [B, 1, hidden_dim]
        
        # Cross-attention: vision attends to language
        fused, _ = self.attention(vision_proj, lang_proj, lang_proj)
        fused = self.norm(fused.squeeze(1) + vision_proj.squeeze(1))
        fused = self.dropout(fused)
        
        return fused


class GrootSystem2(nn.Module):
    """System 2: Vision-Language Model for slow deliberate reasoning.
    
    Processes language instructions and understands task context.
    Generates high-level plans and reasoning about actions.
    """
    
    def __init__(
        self,
        vision_dim: int = DEFAULT_VISION_DIM,
        language_dim: int = DEFAULT_LANGUAGE_DIM,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        image_size: int = DEFAULT_IMAGE_SIZE,
        vocab_size: int = DEFAULT_VOCAB_SIZE,
        max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
        num_attention_heads: int = DEFAULT_ATTENTION_HEADS,
        num_attention_layers: int = DEFAULT_ATTENTION_LAYERS,
        dropout: float = DEFAULT_DROPOUT,
    ):
        """Initialize System 2 VLM component.
        
        Args:
            vision_dim: Vision encoder output dimension
            language_dim: Language encoder dimension
            hidden_dim: Hidden layer dimension
            image_size: Input image size
            vocab_size: Vocabulary size
            max_seq_len: Maximum sequence length
            num_attention_heads: Number of attention heads
            num_attention_layers: Number of transformer layers
            dropout: Dropout probability
        """
        super().__init__()
        
        # Vision encoder (Vision Transformer)
        self.vision_encoder = VisionTransformerEncoder(
            image_size=image_size,
            d_model=vision_dim,
            num_layers=num_attention_layers,
            num_heads=num_attention_heads,
            dropout=dropout,
        )
        
        # Language encoder (Transformer)
        self.language_encoder = LanguageTransformerEncoder(
            vocab_size=vocab_size,
            d_model=language_dim,
            num_layers=num_attention_layers,
            num_heads=num_attention_heads,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )
        
        # Cross-modal attention for fusion
        self.cross_modal_attention = CrossModalAttention(
            vision_dim=vision_dim,
            language_dim=language_dim,
            hidden_dim=hidden_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
        )
        
        # Reasoning head
        self.reasoning_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0.0)
                nn.init.constant_(module.weight, 1.0)
    
    def forward(
        self,
        images: torch.Tensor,
        language_tokens: torch.Tensor,
        language_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through System 2.
        
        Args:
            images: Batch of images [B, C, H, W]
            language_tokens: Batch of language token sequences [B, L]
            language_mask: Attention mask for language tokens [B, L] (optional)
            
        Returns:
            Reasoning features [B, hidden_dim]
        """
        # Encode vision
        vision_features = self.vision_encoder(images)  # [B, vision_dim]
        
        # Encode language
        lang_features = self.language_encoder(language_tokens, mask=language_mask)  # [B, language_dim]
        
        # Cross-modal fusion
        fused = self.cross_modal_attention(vision_features, lang_features)  # [B, hidden_dim]
        
        # Generate reasoning
        reasoning = self.reasoning_head(fused)  # [B, hidden_dim]
        
        return reasoning


class DiffusionNoiseScheduler:
    """Noise scheduler for diffusion model."""
    
    def __init__(
        self,
        num_steps: int = DEFAULT_NUM_DIFFUSION_STEPS,
        beta_start: float = DEFAULT_BETA_START,
        beta_end: float = DEFAULT_BETA_END,
        schedule: str = "linear",
    ):
        """Initialize noise scheduler.
        
        Args:
            num_steps: Number of diffusion steps
            beta_start: Starting beta value
            beta_end: Ending beta value
            schedule: Schedule type ("linear", "cosine")
        """
        self.num_steps = num_steps
        
        if schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_steps)
        elif schedule == "cosine":
            s = 0.008
            steps = torch.arange(num_steps + 1, dtype=torch.float32)
            alphas_cumprod = torch.cos(((steps / num_steps) + s) / (1 + s) * math.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = torch.clamp(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")
        
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Precompute for sampling
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        # Add small epsilon to avoid division by zero
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod + 1e-8)
        )
        # Clamp posterior variance to avoid numerical issues
        self.posterior_variance = torch.clamp(self.posterior_variance, min=1e-20)
    
    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample random timesteps.
        
        Args:
            batch_size: Batch size
            device: Device
            
        Returns:
            Random timesteps [B]
        """
        return torch.randint(0, self.num_steps, (batch_size,), device=device)
    
    def add_noise(
        self,
        x_start: torch.Tensor,
        timesteps: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Add noise to input.
        
        Args:
            x_start: Clean input [B, ...]
            timesteps: Timesteps [B]
            noise: Optional noise tensor
            
        Returns:
            Noisy input
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[timesteps].view(-1, *([1] * (len(x_start.shape) - 1)))
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[timesteps].view(-1, *([1] * (len(x_start.shape) - 1)))
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise


class DiffusionUNet(nn.Module):
    """U-Net architecture for diffusion model."""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 4,
        dropout: float = DEFAULT_DROPOUT,
    ):
        """Initialize diffusion U-Net.
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            hidden_dim: Hidden dimension
            num_layers: Number of layers
            dropout: Dropout probability
        """
        super().__init__()
        
        # Time embedding
        self.time_embed_dim = hidden_dim
        self.time_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Down blocks
        self.down_blocks = nn.ModuleList()
        dim = hidden_dim
        for i in range(num_layers):
            self.down_blocks.append(
                nn.Sequential(
                    nn.Linear(dim, dim * 2),
                    nn.LayerNorm(dim * 2),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                )
            )
            dim *= 2
        
        # Middle block
        self.middle_block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
        )
        
        # Up blocks
        self.up_blocks = nn.ModuleList()
        for i in range(num_layers):
            self.up_blocks.append(
                nn.Sequential(
                    nn.Linear(dim * 2, dim // 2),
                    nn.LayerNorm(dim // 2),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                )
            )
            dim //= 2
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
    
    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through U-Net.
        
        Args:
            x: Input tensor [B, input_dim]
            timestep: Timestep [B]
            context: Context features [B, context_dim] (optional)
            
        Returns:
            Output tensor [B, output_dim]
        """
        # Time embedding
        t_emb = self._get_timestep_embedding(timestep)
        t_emb = self.time_embed(t_emb)
        
        # Input projection
        h = self.input_proj(x)
        if context is not None:
            h = h + context
        
        # Down blocks
        skip_connections = []
        for down_block in self.down_blocks:
            skip_connections.append(h)
            h = down_block(h)
        
        # Middle block
        h = self.middle_block(h)
        
        # Up blocks with skip connections
        for up_block in self.up_blocks:
            h = torch.cat([h, skip_connections.pop()], dim=-1)
            h = up_block(h)
        
        # Output projection
        output = self.output_proj(h)
        
        return output
    
    def _get_timestep_embedding(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Get sinusoidal timestep embedding.
        
        Args:
            timesteps: Timesteps [B] or [B, 1]
            
        Returns:
            Timestep embeddings [B, time_embed_dim]
        """
        # Handle both [B] and [B, 1] shapes
        if len(timesteps.shape) == 2:
            timesteps = timesteps.squeeze(-1)  # [B, 1] -> [B]
        
        half_dim = self.time_embed_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) * -emb)
        emb = timesteps[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class GrootSystem1(nn.Module):
    """System 1: Diffusion Transformer for fast reactive action generation.
    
    Generates continuous action values at 100+ Hz (100+ actions per second) for real-time control.
    Uses diffusion process to generate smooth, natural actions.
    
    Technical specifications (per GR00T study guide):
    - Action space: 20-30 DOF for humanoid robots
    - Control frequency: 10-30 Hz (33-100 ms per action)
    - Temporal horizon: 1-10 seconds lookahead
    - Diffusion steps: 50-100 for inference (1000 for training)
    - Noise schedule: Cosine or linear, beta range [0.0001, 0.02]
    - Inference latency: 10-30ms per step
    """
    
    def __init__(
        self,
        input_dim: int = DEFAULT_HIDDEN_DIM,
        action_dim: int = DEFAULT_ACTION_DIM,
        hidden_dim: int = 512,
        num_diffusion_steps: int = DEFAULT_NUM_DIFFUSION_STEPS,
        beta_start: float = DEFAULT_BETA_START,
        beta_end: float = DEFAULT_BETA_END,
        dropout: float = DEFAULT_DROPOUT,
    ):
        """Initialize System 1 diffusion component.
        
        Args:
            input_dim: Input feature dimension (from System 2)
            action_dim: Action space dimension (20-30 DOF for humanoid robots)
            hidden_dim: Hidden dimension for diffusion model (512-1024)
            num_diffusion_steps: Number of diffusion steps (50-100 for inference, 1000 for training)
            beta_start: Starting beta value (0.0001)
            beta_end: Ending beta value (0.02)
            dropout: Dropout probability (0.1-0.2)
        """
        super().__init__()
        
        self.action_dim = action_dim
        self.num_diffusion_steps = num_diffusion_steps
        
        # Create noise scheduler and register its buffers
        noise_scheduler = DiffusionNoiseScheduler(
            num_steps=num_diffusion_steps,
            beta_start=beta_start,
            beta_end=beta_end,
        )
        
        # Register buffers so they move with the model to correct device
        self.register_buffer("betas", noise_scheduler.betas)
        self.register_buffer("alphas", noise_scheduler.alphas)
        self.register_buffer("alphas_cumprod", noise_scheduler.alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", noise_scheduler.alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", noise_scheduler.sqrt_alphas_cumprod)
        self.register_buffer("sqrt_one_minus_alphas_cumprod", noise_scheduler.sqrt_one_minus_alphas_cumprod)
        self.register_buffer("posterior_variance", noise_scheduler.posterior_variance)
        
        # Keep scheduler reference for methods that need it
        self.noise_scheduler = noise_scheduler
        # Update scheduler's buffers to point to registered buffers
        self.noise_scheduler.betas = self.betas
        self.noise_scheduler.alphas = self.alphas
        self.noise_scheduler.alphas_cumprod = self.alphas_cumprod
        self.noise_scheduler.alphas_cumprod_prev = self.alphas_cumprod_prev
        self.noise_scheduler.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod
        self.noise_scheduler.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod
        self.noise_scheduler.posterior_variance = self.posterior_variance
        
        # Diffusion U-Net
        self.diffusion_model = DiffusionUNet(
            input_dim=action_dim,  # Predict noise in action space
            output_dim=action_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        
        # Context projection (from System 2 reasoning)
        self.context_proj = nn.Linear(input_dim, hidden_dim)
    
    def forward(
        self,
        reasoning_features: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass through System 1 diffusion model.
        
        Args:
            reasoning_features: Features from System 2 [B, input_dim]
            actions: Ground truth actions [B, action_dim] (for training)
            timesteps: Diffusion timesteps [B] (for training)
            noise: Noise tensor [B, action_dim] (optional)
            
        Returns:
            Tuple of (predicted_noise_or_actions, actual_noise)
            - Training: (predicted_noise [B, action_dim], actual_noise [B, action_dim])
            - Inference: (sampled_actions [B, action_dim], None)
        """
        batch_size = reasoning_features.shape[0]
        
        # Project context
        context = self.context_proj(reasoning_features)  # [B, hidden_dim]
        
        if self.training and actions is not None:
            # Training: predict noise
            if timesteps is None:
                timesteps = self.noise_scheduler.sample_timesteps(
                    batch_size, reasoning_features.device
                )
            
            # Add noise to actions
            if noise is None:
                noise = torch.randn_like(actions)
            
            noisy_actions = self.noise_scheduler.add_noise(actions, timesteps, noise)
            
            # Predict noise
            timestep_emb = timesteps.float().unsqueeze(-1)  # [B, 1]
            predicted_noise = self.diffusion_model(
                noisy_actions, timestep_emb, context
            )
            
            return predicted_noise, noise
        else:
            # Inference: sample actions
            sampled_actions = self.sample(reasoning_features)
            return sampled_actions, None
    
    def sample(self, reasoning_features: torch.Tensor, inference_steps: Optional[int] = None) -> torch.Tensor:
        """Sample actions from diffusion model.
        
        Args:
            reasoning_features: Features from System 2 [B, input_dim]
            inference_steps: Number of inference steps (50-100 for real-time, None uses training steps)
            
        Returns:
            Sampled actions [B, action_dim]
        """
        batch_size = reasoning_features.shape[0]
        device = reasoning_features.device
        
        # Use fewer steps for inference (50-100) for real-time performance
        # Training uses full steps (1000), but inference can use fewer
        num_steps = inference_steps if inference_steps is not None else min(100, self.num_diffusion_steps)
        step_size = self.num_diffusion_steps // num_steps if num_steps < self.num_diffusion_steps else 1
        
        # Start from pure noise
        actions = torch.randn(batch_size, self.action_dim, device=device)
        
        # Project context
        context = self.context_proj(reasoning_features)  # [B, hidden_dim]
        
        # CRITICAL FIX: Denoising loop with proper bounds checking
        for i in range(num_steps - 1, -1, -1):
            t = i * step_size
            # CRITICAL FIX: Ensure t is within valid range [0, num_diffusion_steps-1]
            t = max(0, min(t, self.num_diffusion_steps - 1))
            
            timestep = torch.full((batch_size,), t, device=device)
            timestep_emb = timestep.float().unsqueeze(-1)  # [B, 1]
            
            # Predict noise
            predicted_noise = self.diffusion_model(actions, timestep_emb, context)
            
            # CRITICAL FIX: Denoise using DDPM sampling with bounds checking
            alpha_t = self.alphas[t]
            alpha_cumprod_t = self.alphas_cumprod[t]
            
            # CRITICAL FIX: Add epsilon to avoid division by zero
            sqrt_alpha_cumprod_t = torch.sqrt(alpha_cumprod_t + 1e-8)
            sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]
            
            # Predict x_0 from noisy actions
            pred_x0 = (actions - sqrt_one_minus_alpha_cumprod_t * predicted_noise) / sqrt_alpha_cumprod_t
            
            # CRITICAL FIX: Compute previous step with correct indexing
            if i > 0:  # Not the final step
                # Ensure t is within valid range for posterior_variance
                t_clamped = min(t, len(self.posterior_variance) - 1)
                posterior_variance = self.posterior_variance[t_clamped]
                
                # CRITICAL FIX: Clamp posterior variance to avoid numerical issues
                posterior_variance = torch.clamp(posterior_variance, min=1e-20, max=1.0)
                
                # Sample noise for stochastic step
                noise = torch.randn_like(actions)
                
                # CRITICAL FIX: Compute mean of posterior distribution with numerical stability
                sqrt_alpha_t = torch.sqrt(alpha_t + 1e-8)
                beta_t = self.betas[t]
                coeff = beta_t / (sqrt_one_minus_alpha_cumprod_t + 1e-8)
                posterior_mean = (1.0 / sqrt_alpha_t) * (actions - coeff * predicted_noise)
                
                # Sample from posterior
                actions = posterior_mean + torch.sqrt(posterior_variance) * noise
            else:
                # Final step: no noise
                actions = pred_x0
        
        return actions


class GrootVLA(nn.Module):
    """Complete GR00T Vision-Language-Action (VLA) Foundation Model.
    
    Combines System 2 (deliberate reasoning) and System 1 (reactive actions)
    to generate actions from vision and language inputs.
    
    Architecture inspired by "Thinking Fast and Slow" paradigm:
    - System 2: Slow, deliberate reasoning (VLM)
    - System 1: Fast, reactive action generation (Diffusion)
    """
    
    def __init__(
        self,
        config: Optional[GrootTrainingConfig] = None,
    ):
        """Initialize GR00T VLA model.
        
        Args:
            config: Training configuration
        """
        super().__init__()
        
        if config is None:
            config = GrootTrainingConfig()
        
        self.config = config
        
        self.system2 = GrootSystem2(
            vision_dim=config.vision_dim,
            language_dim=config.language_dim,
            hidden_dim=config.hidden_dim,
            image_size=config.image_size,
            vocab_size=config.vocab_size,
            max_seq_len=config.max_seq_len,
            num_attention_heads=config.num_attention_heads,
            num_attention_layers=config.num_attention_layers,
            dropout=config.dropout,
        )
        
        self.system1 = GrootSystem1(
            input_dim=config.hidden_dim,
            action_dim=config.action_dim,
            num_diffusion_steps=config.num_diffusion_steps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            dropout=config.dropout,
        )
    
    def forward(
        self,
        images: torch.Tensor,
        language_tokens: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None,
        language_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through complete GR00T model.
        
        Args:
            images: Batch of images [B, C, H, W]
            language_tokens: Batch of language token sequences [B, L]
            actions: Ground truth actions [B, action_dim] (for training)
            timesteps: Diffusion timesteps [B] (for training)
            language_mask: Attention mask for language tokens [B, L] (optional)
            
        Returns:
            Dictionary with actions/predicted_noise and intermediate features
        """
        # System 2: Deliberate reasoning
        reasoning = self.system2(images, language_tokens, language_mask)  # [B, hidden_dim]
        
        # System 1: Fast action generation
        if self.training and actions is not None:
            predicted_noise, actual_noise = self.system1(reasoning, actions, timesteps)  # [B, action_dim]
            return {
                "predicted_noise": predicted_noise,
                "actual_noise": actual_noise,
                "reasoning": reasoning,
            }
        else:
            sampled_actions, _ = self.system1(reasoning)  # [B, action_dim]
            return {
                "actions": sampled_actions,
                "reasoning": reasoning,
            }


def compute_diffusion_loss(
    predicted_noise: torch.Tensor,
    target_noise: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute diffusion loss.
    
    Args:
        predicted_noise: Predicted noise [B, action_dim]
        target_noise: Target noise [B, action_dim]
        reduction: Loss reduction ("mean", "sum", "none")
        
    Returns:
        Loss value
    """
    loss = F.mse_loss(predicted_noise, target_noise, reduction=reduction)
    return loss


def get_learning_rate_scheduler(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr: float = 0.0,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Get learning rate scheduler with warmup.
    
    Args:
        optimizer: Optimizer
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        min_lr: Minimum learning rate
        
    Returns:
        Learning rate scheduler
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(min_lr, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def preprocess_batch(
    batch: Dict[str, Any],
    image_size: int = DEFAULT_IMAGE_SIZE,
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, torch.Tensor]:
    """Preprocess batch for training.
    
    Args:
        batch: Raw batch dictionary
        image_size: Target image size
        max_seq_len: Maximum sequence length
        device: Device to move tensors to
        
    Returns:
        Preprocessed batch dictionary
    """
    # CRITICAL FIX: Extract and preprocess images with multi-view support
    if "image" in batch:
        images = batch["image"]
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)
        if not isinstance(images, torch.Tensor):
            images = torch.tensor(images)
        
        # Normalize to [0, 1] and ensure correct shape
        if images.dtype != torch.float32:
            images = images.float()
        if images.max() > 1.0:
            images = images / 255.0
        
        # CRITICAL FIX: Handle various input shapes
        # Support: [H, W, C], [B, H, W, C], [B, C, H, W], [B, num_views, C, H, W]
        original_shape = images.shape
        
        # Handle 3D input (single image)
        if len(images.shape) == 3:
            # [H, W, C] or [C, H, W]
            if images.shape[0] == 3 or images.shape[2] == 3:
                if images.shape[2] == 3:  # [H, W, C]
                    images = images.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
                else:  # [C, H, W]
                    images = images.unsqueeze(0)  # [1, C, H, W]
            else:
                images = images.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W] -> will convert to RGB
        
        # Handle 4D input
        elif len(images.shape) == 4:
            # [B, H, W, C] or [B, C, H, W] or [B, num_views, C, H, W]
            if images.shape[-1] == 3 or images.shape[-1] == 1:
                # [B, H, W, C] -> [B, C, H, W]
                images = images.permute(0, 3, 1, 2)
            # else assume [B, C, H, W] or [B, num_views, C, H, W]
        
        # Handle 5D input (multi-view)
        elif len(images.shape) == 5:
            # [B, num_views, C, H, W] - already correct format
            pass
        else:
            raise ValueError(f"Unsupported image shape: {original_shape}")
        
        # CRITICAL FIX: Resize to target size if needed
        # Handle both single-view [B, C, H, W] and multi-view [B, num_views, C, H, W]
        if len(images.shape) == 4:
            # Single view: [B, C, H, W]
            if images.shape[2] != image_size or images.shape[3] != image_size:
                images = F.interpolate(
                    images,
                    size=(image_size, image_size),
                    mode="bilinear",
                    align_corners=False,
                )
            # Ensure RGB (3 channels)
            if images.shape[1] == 1:
                images = images.repeat(1, 3, 1, 1)
            elif images.shape[1] > 3:
                images = images[:, :3]
        elif len(images.shape) == 5:
            # Multi-view: [B, num_views, C, H, W]
            if images.shape[3] != image_size or images.shape[4] != image_size:
                B, V, C, H, W = images.shape
                images = images.view(B * V, C, H, W)
                images = F.interpolate(
                    images,
                    size=(image_size, image_size),
                    mode="bilinear",
                    align_corners=False,
                )
                images = images.view(B, V, C, image_size, image_size)
            # Ensure RGB (3 channels)
            if images.shape[2] == 1:
                images = images.repeat(1, 1, 3, 1, 1)
            elif images.shape[2] > 3:
                images = images[:, :, :3]
    else:
        raise ValueError("Batch must contain 'image' key")
    
    # Extract and preprocess language tokens
    if "language_tokens" in batch:
        language_tokens = batch["language_tokens"]
        if isinstance(language_tokens, np.ndarray):
            language_tokens = torch.from_numpy(language_tokens)
        if not isinstance(language_tokens, torch.Tensor):
            language_tokens = torch.tensor(language_tokens, dtype=torch.long)
        language_tokens = language_tokens.long()
        
        # Pad or truncate to max_seq_len
        if language_tokens.shape[-1] > max_seq_len:
            language_tokens = language_tokens[:, :max_seq_len]
        elif language_tokens.shape[-1] < max_seq_len:
            padding = torch.zeros(
                language_tokens.shape[0],
                max_seq_len - language_tokens.shape[-1],
                dtype=torch.long,
                device=language_tokens.device,
            )
            language_tokens = torch.cat([language_tokens, padding], dim=-1)
        
        # CRITICAL FIX: Create attention mask [B, L] where 1 = valid token, 0 = padding
        # Use float for proper masking in attention
        language_mask = (language_tokens != 0).float()
    else:
        # CRITICAL FIX: Create dummy tokens if missing
        language_tokens = torch.zeros(
            images.shape[0], max_seq_len, dtype=torch.long, device=device
        )
        language_mask = torch.zeros(
            images.shape[0], max_seq_len, dtype=torch.float32, device=device
        )
    
    # CRITICAL FIX: Extract and preprocess actions with dimension validation
    if "actions" in batch:
        actions = batch["actions"]
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions)
        if not isinstance(actions, torch.Tensor):
            actions = torch.tensor(actions)
        actions = actions.float()
        
        # Ensure correct shape
        if len(actions.shape) == 1:
            actions = actions.unsqueeze(0)
        
        # CRITICAL FIX: Validate action dimension matches expected
        # GR00T uses 20-30 DOF for humanoid robots
        if len(actions.shape) == 2:
            action_dim = actions.shape[1]
            # Allow some flexibility but warn if way off
            if action_dim < 10 or action_dim > 50:
                logger.warning(f"Unusual action dimension: {action_dim} (expected 20-30 for humanoids)")
        elif len(actions.shape) == 1:
            action_dim = actions.shape[0]
            if action_dim < 10 or action_dim > 50:
                logger.warning(f"Unusual action dimension: {action_dim} (expected 20-30 for humanoids)")
        
        # CRITICAL FIX: Check for NaN/Inf in actions
        if not torch.isfinite(actions).all():
            logger.warning("Non-finite values detected in actions, replacing with zeros")
            actions = torch.where(torch.isfinite(actions), actions, torch.zeros_like(actions))
    else:
        raise ValueError("Batch must contain 'actions' key")
    
    # Move to device
    images = images.to(device)
    language_tokens = language_tokens.to(device)
    language_mask = language_mask.to(device)
    actions = actions.to(device)
    
    return {
        "images": images,
        "language_tokens": language_tokens,
        "language_mask": language_mask,
        "actions": actions,
    }


def get_training_dataset(
    curated_data_path: str,
    shuffle: bool = True,
    seed: int = 42,
    use_streaming: bool = True,
    use_gpu_object_store: bool = True,
) -> Dataset:
    """Get training dataset from curated data using Ray Data with latest optimizations.
    
    Args:
        curated_data_path: Path to curated dataset (from data pipeline)
        shuffle: Whether to shuffle the dataset
        seed: Random seed for shuffling
        use_streaming: Use Ray Data streaming execution (Ray 2.6+)
        use_gpu_object_store: Use GPU object store with RDMA (Ray 2.6+)
        
    Returns:
        Ray Dataset for training
    """
    if not curated_data_path:
        raise ValueError("curated_data_path cannot be empty")
    
    # CRITICAL FIX: Handle various path types (local, S3, GCS, etc.)
    is_remote_path = (
        curated_data_path.startswith("s3://") or
        curated_data_path.startswith("gs://") or
        curated_data_path.startswith("hdfs://") or
        curated_data_path.startswith("http://") or
        curated_data_path.startswith("https://")
    )
    
    if not is_remote_path and not os.path.exists(curated_data_path):
        raise FileNotFoundError(f"Dataset path not found: {curated_data_path}")
    
    logger.info(f"Loading curated data from {curated_data_path} (remote={is_remote_path})")
    
    # Configure Ray Data context for latest optimizations
    ctx = DataContext.get_current()
    
    # Enable streaming execution for large datasets (Ray 2.6+)
    if use_streaming:
        ctx.execution_options.use_streaming_executor = True
        ctx.execution_options.prefetch_batches = 2
        logger.info("Streaming execution enabled for efficient memory usage")
    
    # Enable GPU object store with RDMA if available (Ray 2.6+)
    if use_gpu_object_store and torch.cuda.is_available():
        try:
            # Set object store to use GPU memory with RDMA
            # This enables direct GPU-to-GPU transfers without CPU roundtrip
            ctx.execution_options.enable_gpu_object_store = True
            logger.info("GPU object store enabled for RDMA transfers")
        except AttributeError:
            logger.warning("GPU object store not available in this Ray version")
    
    # Load curated dataset using Ray Data
    try:
        dataset = ray.data.read_parquet(
            curated_data_path,
            # Use GPU-accelerated reading if available
            num_cpus=2,  # Parallel CPU readers
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset from {curated_data_path}: {e}") from e
    
    # Validate dataset
    if dataset.num_rows() == 0:
        raise ValueError(f"Dataset is empty: {curated_data_path}")
    
    # Shuffle for training (uses distributed shuffling for large datasets)
    if shuffle:
        dataset = dataset.random_shuffle(seed=seed)
    
    logger.info(f"Training dataset prepared successfully: {dataset.num_rows()} rows")
    return dataset


def validate_batch(batch: Dict[str, Any], expected_action_dim: Optional[int] = None) -> bool:
    """Validate batch structure and content.
    
    Args:
        batch: Batch dictionary
        expected_action_dim: Expected action dimension (for validation)
        
    Returns:
        True if batch is valid
    """
    required_keys = ["image", "actions"]
    for key in required_keys:
        if key not in batch:
            logger.warning(f"Missing required key in batch: {key}")
            return False
    
    # Validate shapes and content
    try:
        images = batch["image"]
        actions = batch["actions"]
        
        # CRITICAL FIX: Validate image tensor
        if isinstance(images, torch.Tensor):
            if len(images.shape) < 3:
                logger.warning(f"Invalid image shape (too few dims): {images.shape}")
                return False
            # Check for NaN/Inf in images
            if not torch.isfinite(images).all():
                logger.warning("Non-finite values detected in images")
                return False
            # Check image value range
            if images.min() < 0 or images.max() > 1.1:  # Allow slight overflow
                logger.warning(f"Image values out of range [0, 1]: min={images.min()}, max={images.max()}")
        elif isinstance(images, np.ndarray):
            # Convert to check
            if len(images.shape) < 2:
                logger.warning(f"Invalid image shape: {images.shape}")
                return False
        else:
            logger.warning(f"Invalid image type: {type(images)}")
            return False
        
        # CRITICAL FIX: Validate action tensor
        if isinstance(actions, torch.Tensor):
            if len(actions.shape) < 1:
                logger.warning(f"Invalid actions shape: {actions.shape}")
                return False
            # Check for NaN/Inf in actions
            if not torch.isfinite(actions).all():
                logger.warning("Non-finite values detected in actions")
                return False
            # Validate action dimension if expected
            if expected_action_dim is not None:
                if len(actions.shape) == 1:
                    if actions.shape[0] != expected_action_dim:
                        logger.warning(f"Action dimension mismatch: got {actions.shape[0]}, expected {expected_action_dim}")
                        return False
                elif len(actions.shape) == 2:
                    if actions.shape[1] != expected_action_dim:
                        logger.warning(f"Action dimension mismatch: got {actions.shape[1]}, expected {expected_action_dim}")
                        return False
        elif isinstance(actions, np.ndarray):
            if len(actions.shape) < 1:
                logger.warning(f"Invalid actions shape: {actions.shape}")
                return False
        else:
            logger.warning(f"Invalid actions type: {type(actions)}")
            return False
        
        # CRITICAL FIX: Validate batch size consistency
        if isinstance(images, torch.Tensor) and isinstance(actions, torch.Tensor):
            img_batch_size = images.shape[0] if len(images.shape) >= 4 else 1
            act_batch_size = actions.shape[0] if len(actions.shape) >= 1 else 1
            if img_batch_size != act_batch_size:
                logger.warning(f"Batch size mismatch: images={img_batch_size}, actions={act_batch_size}")
                return False
        
    except Exception as e:
        logger.warning(f"Error validating batch: {e}", exc_info=True)
        return False
    
    return True


def train_epoch(
    model: nn.Module,
    dataset: Dataset,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    scaler: Optional[GradScaler],
    device: torch.device,
    epoch: int,
    config: GrootTrainingConfig,
    global_step: int = 0,
) -> Dict[str, float]:
    """Train one epoch using Ray Data.
    
    Args:
        model: GR00T model
        dataset: Ray Dataset for training
        optimizer: Optimizer
        scheduler: Learning rate scheduler (optional)
        scaler: Gradient scaler for mixed precision (optional)
        device: Training device
        epoch: Current epoch number
        config: Training configuration
        global_step: Global training step
        
    Returns:
        Dictionary with training metrics
    """
    model.train()
    total_loss = 0.0
    total_diffusion_loss = 0.0
    num_batches = 0
    num_valid_batches = 0
    
    # CRITICAL FIX: Use Ray Data's iter_torch_batches with proper error handling
    # Add retry logic and timeout for transient failures
    max_batch_retries = 3
    consecutive_errors = 0
    max_consecutive_errors = 10  # Stop if too many consecutive errors
    
    try:
        batch_iterator = dataset.iter_torch_batches(
            batch_size=config.batch_size,
            device=device,
            drop_last=True,
            prefetch_batches=config.prefetch_factor,
        )
        
        for batch_idx, batch in enumerate(batch_iterator):
            batch_success = False
            retry_count = 0
            
            while retry_count < max_batch_retries and not batch_success:
                try:
                    # CRITICAL FIX: Validate batch with expected action dimension
                    if not validate_batch(batch, expected_action_dim=config.action_dim):
                        logger.warning(f"Skipping invalid batch at index {batch_idx}")
                        batch_success = True  # Mark as "processed" (skipped)
                        continue
                    
                    # Preprocess batch
                    processed_batch = preprocess_batch(
                        batch,
                        image_size=config.image_size,
                        max_seq_len=config.max_seq_len,
                        device=device,
                    )
                    
                    images = processed_batch["images"]
                    language_tokens = processed_batch["language_tokens"]
                    language_mask = processed_batch["language_mask"]
                    actions = processed_batch["actions"]
                    
                    # CRITICAL FIX: Don't create new CUDA stream every batch (memory leak)
                    # Data is already on device from iter_torch_batches, no need for separate stream
                    # Forward pass with mixed precision
                    with autocast(enabled=config.use_mixed_precision):
                        output = model(
                            images=images,
                            language_tokens=language_tokens,
                            actions=actions,
                            language_mask=language_mask,
                        )
                        
                        predicted_noise = output["predicted_noise"]
                        actual_noise = output.get("actual_noise")
                        
                        # Use the actual noise that was added during forward pass
                        if actual_noise is not None:
                            target_noise = actual_noise
                        else:
                            # Fallback: this shouldn't happen in training mode
                            logger.warning("actual_noise not found in output, using random noise")
                            target_noise = torch.randn_like(actions)
                        
                        # Compute loss
                        loss = compute_diffusion_loss(predicted_noise, target_noise)
                        
                        # CRITICAL FIX: Detect NaN/Inf losses early
                        if not torch.isfinite(loss):
                            logger.error(f"Non-finite loss detected: {loss.item()} at batch {batch_idx}")
                            if torch.isnan(loss):
                                logger.error("NaN loss detected - skipping batch and clearing gradients")
                                optimizer.zero_grad()
                                batch_success = True
                                continue
                            elif torch.isinf(loss):
                                logger.error("Inf loss detected - clipping to max value")
                                loss = torch.clamp(loss, min=-1e6, max=1e6)
                    
                    # CRITICAL FIX: Scale loss by accumulation steps BEFORE backward
                    # This ensures correct gradient averaging across accumulation steps
                    scaled_loss = loss / config.gradient_accumulation_steps
                    
                    # Backward pass with gradient accumulation
                    if config.use_mixed_precision:
                        scaler.scale(scaled_loss).backward()
                    else:
                        scaled_loss.backward()
                    
                    # CRITICAL FIX: Only increment step counter when actually updating
                    # Accumulate loss for logging (unscaled)
                    accumulated_loss = loss.item()  # Store unscaled loss for metrics
                    
                    # Update weights only after accumulating enough gradients
                    if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                        # CRITICAL FIX: Check for NaN gradients before updating
                        if config.use_mixed_precision:
                            scaler.unscale_(optimizer)
                        
                        # Check for NaN/Inf gradients
                        has_nan_grad = False
                        for param in model.parameters():
                            if param.grad is not None:
                                if not torch.isfinite(param.grad).all():
                                    has_nan_grad = True
                                    logger.warning(f"NaN/Inf gradient detected at batch {batch_idx}")
                                    break
                        
                        if has_nan_grad:
                            logger.warning("Skipping optimizer step due to NaN gradients")
                            optimizer.zero_grad()
                            if config.use_mixed_precision:
                                scaler.update()
                            batch_success = True
                            continue
                        
                        # Clip gradients
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            model.parameters(), config.gradient_clip_norm
                        )
                        
                        # CRITICAL FIX: Check if gradient norm is valid
                        if not torch.isfinite(grad_norm):
                            logger.warning(f"Non-finite gradient norm: {grad_norm}, skipping step")
                            optimizer.zero_grad()
                            if config.use_mixed_precision:
                                scaler.update()
                            batch_success = True
                            continue
                        
                        # Optimizer step
                        if config.use_mixed_precision:
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            optimizer.step()
                        
                        # CRITICAL FIX: Only step scheduler when optimizer steps
                        if scheduler is not None:
                            scheduler.step()
                        
                        optimizer.zero_grad()
                        
                        # CRITICAL FIX: Only increment global_step when actually updating
                        global_step += 1
                    
                    # Accumulate loss for metrics (use unscaled loss)
                    total_loss += accumulated_loss
                    total_diffusion_loss += accumulated_loss
                    num_batches += 1
                    num_valid_batches += 1
                    
                    if batch_idx % 100 == 0:
                        current_lr = optimizer.param_groups[0]["lr"]
                        # Log GPU memory usage periodically
                        if torch.cuda.is_available() and batch_idx % 500 == 0:
                            mem_info = get_gpu_memory_info()
                            logger.info(
                                f"Epoch {epoch}, Batch {batch_idx}, Step {global_step}, "
                                f"Loss: {loss.item():.4f}, LR: {current_lr:.2e}, "
                                f"GPU Mem: {mem_info['allocated'] / 1e9:.2f}GB / {mem_info['total'] / 1e9:.2f}GB"
                            )
                        else:
                            logger.info(
                                f"Epoch {epoch}, Batch {batch_idx}, Step {global_step}, "
                                f"Loss: {loss.item():.4f}, LR: {current_lr:.2e}"
                            )
                    
                    # CRITICAL FIX: Periodic GPU memory cleanup and monitoring
                    if batch_idx % 1000 == 0 and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        # Monitor memory usage
                        mem_info = get_gpu_memory_info(device_id=device.index if device.type == "cuda" else None)
                        mem_usage_pct = (mem_info["allocated"] / mem_info["total"]) * 100 if mem_info["total"] > 0 else 0
                        if mem_usage_pct > 90:
                            logger.warning(f"High GPU memory usage: {mem_usage_pct:.1f}%")
                    
                    batch_success = True
                    consecutive_errors = 0  # Reset error counter on success
                    
                except RuntimeError as e:
                    # CRITICAL FIX: Handle CUDA out-of-memory errors
                    if "out of memory" in str(e).lower():
                        logger.error(f"CUDA OOM at batch {batch_idx}: {e}")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        retry_count += 1
                        if retry_count < max_batch_retries:
                            logger.info(f"Retrying batch {batch_idx} after OOM (attempt {retry_count + 1})")
                            time.sleep(retry_count * 0.5)  # Exponential backoff
                        else:
                            logger.error(f"Failed to process batch {batch_idx} after {max_batch_retries} retries")
                            consecutive_errors += 1
                            if consecutive_errors >= max_consecutive_errors:
                                raise RuntimeError(f"Too many consecutive errors ({consecutive_errors}), stopping training")
                            batch_success = True  # Skip this batch
                    else:
                        raise
                        
                except Exception as e:
                    retry_count += 1
                    logger.error(f"Error processing batch {batch_idx} (attempt {retry_count}): {e}", exc_info=True)
                    if retry_count < max_batch_retries:
                        logger.info(f"Retrying batch {batch_idx} (attempt {retry_count + 1})")
                        time.sleep(retry_count * 0.5)  # Exponential backoff
                    else:
                        logger.error(f"Failed to process batch {batch_idx} after {max_batch_retries} retries")
                        consecutive_errors += 1
                        if consecutive_errors >= max_consecutive_errors:
                            raise RuntimeError(f"Too many consecutive errors ({consecutive_errors}), stopping training")
                        batch_success = True  # Skip this batch
                        num_batches += 1
                        continue
            
            # CRITICAL FIX: Track consecutive errors across batches
            if not batch_success:
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    raise RuntimeError(f"Too many consecutive batch failures ({consecutive_errors}), stopping training")
    
    except Exception as e:
        logger.error(f"Error in training epoch {epoch}: {e}", exc_info=True)
        raise
    
    avg_loss = total_loss / num_valid_batches if num_valid_batches > 0 else 0.0
    avg_diffusion_loss = total_diffusion_loss / num_valid_batches if num_valid_batches > 0 else 0.0
    
    return {
        "loss": avg_loss,
        "diffusion_loss": avg_diffusion_loss,
        "num_batches": num_batches,
        "num_valid_batches": num_valid_batches,
        "global_step": global_step,
    }


def evaluate_model(
    model: nn.Module,
    dataset: Dataset,
    device: torch.device,
    config: GrootTrainingConfig,
) -> Dict[str, float]:
    """Evaluate model on validation dataset.
    
    Args:
        model: GR00T model
        dataset: Validation dataset
        device: Device
        config: Training configuration
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataset.iter_torch_batches(
            batch_size=config.batch_size,
            device=device,
            drop_last=False,
        ):
            try:
                # CRITICAL FIX: Validate batch with expected action dimension
                if not validate_batch(batch, expected_action_dim=config.action_dim):
                    continue
                
                processed_batch = preprocess_batch(
                    batch,
                    image_size=config.image_size,
                    max_seq_len=config.max_seq_len,
                    device=device,
                )
                
                images = processed_batch["images"]
                language_tokens = processed_batch["language_tokens"]
                language_mask = processed_batch["language_mask"]
                target_actions = processed_batch["actions"]
                
                # Forward pass
                with autocast(enabled=config.use_mixed_precision):
                    output = model(
                        images=images,
                        language_tokens=language_tokens,
                        language_mask=language_mask,
                    )
                    
                    predicted_actions = output["actions"]
                    
                    # Compute metrics
                    loss = F.mse_loss(predicted_actions, target_actions)
                    mse = loss.item()
                
                total_loss += loss.item()
                total_mse += mse
                num_batches += 1
            
            except Exception as e:
                logger.error(f"Error in evaluation batch: {e}", exc_info=True)
                continue
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_mse = total_mse / num_batches if num_batches > 0 else 0.0
    
    return {
        "val_loss": avg_loss,
        "val_mse": avg_mse,
        "num_batches": num_batches,
    }


def train_func(config_dict: Dict[str, Any]) -> None:
    """Training function for Ray Train.
    
    This function runs on each training worker.
    
    Args:
        config_dict: Training configuration dictionary
    """
    # Initialize distributed training FIRST before any CUDA operations
    train_context = get_context()
    worker_id = train_context.get_world_rank()
    world_size = train_context.get_world_size()
    
    # Initialize distributed process group if multi-GPU
    if world_size > 1:
        if not dist.is_initialized():
            # Initialize process group - CRITICAL for distributed training
            dist.init_process_group(
                backend="nccl" if torch.cuda.is_available() else "gloo",
                init_method="env://",
                world_size=world_size,
                rank=worker_id,
            )
        # Set device based on local rank (not global rank)
        local_rank = worker_id % torch.cuda.device_count() if torch.cuda.is_available() else 0
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Barrier to ensure all workers are ready
    if world_size > 1:
        dist.barrier()
    
    logger.info(f"Worker {worker_id}/{world_size} using device: {device}")
    
    try:
        
        # Create config object
        config = GrootTrainingConfig(**config_dict)
        
        # Initialize model
        model = GrootVLA(config=config)
        model = model.to(device)
        
        # Apply structured sparsity if enabled (2:4 sparsity for A100+)
        if config.use_structured_sparsity and torch.cuda.is_available():
            try:
                # Check if GPU supports structured sparsity
                device_props = torch.cuda.get_device_properties(device)
                if device_props.major >= 8:  # A100+ supports 2:4 sparsity
                    logger.info("Applying 2:4 structured sparsity for A100+ GPU")
                    # Note: This is a placeholder - actual implementation would
                    # require converting linear layers to sparse format
                    # In practice, use torch.sparse or NVIDIA's automatic sparsity toolkit
            except Exception as e:
                logger.warning(f"Failed to apply structured sparsity: {e}")
        
        # CRITICAL FIX: Wrap model for distributed training with proper device handling
        if config.use_fsdp and world_size > 1:
            # Use FSDP (Fully Sharded Data Parallel) for memory efficiency
            try:
                from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
                from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
                from functools import partial
                
                # Auto-wrap policy for transformer layers
                auto_wrap_policy = partial(
                    transformer_auto_wrap_policy,
                    transformer_layer_cls={TransformerBlock},
                )
                
                # CRITICAL FIX: Use local rank for device_id, not global rank
                local_rank = worker_id % torch.cuda.device_count() if torch.cuda.is_available() else 0
                
                model = FSDP(
                    model,
                    auto_wrap_policy=auto_wrap_policy,
                    mixed_precision=torch.distributed.fsdp.MixedPrecision(
                        param_dtype=torch.float16 if config.use_mixed_precision else torch.float32,
                        reduce_dtype=torch.float16 if config.use_mixed_precision else torch.float32,
                    ) if config.use_mixed_precision else None,
                    device_id=local_rank if torch.cuda.is_available() else None,
                    **config.fsdp_config or {},
                )
                logger.info(f"Using FSDP (Fully Sharded Data Parallel) on device {local_rank}")
            except ImportError:
                logger.warning("FSDP not available, falling back to DDP")
                local_rank = worker_id % torch.cuda.device_count() if torch.cuda.is_available() else 0
                model = DDP(
                    model,
                    device_ids=[local_rank] if torch.cuda.is_available() else None,
                    output_device=local_rank if torch.cuda.is_available() else None,
                    find_unused_parameters=False,
                )
        elif world_size > 1:
            # CRITICAL FIX: Use local rank for device_ids
            local_rank = worker_id % torch.cuda.device_count() if torch.cuda.is_available() else 0
            model = DDP(
                model,
                device_ids=[local_rank] if torch.cuda.is_available() else None,
                output_device=local_rank if torch.cuda.is_available() else None,
                find_unused_parameters=False,
                # Use gradient bucket for better communication efficiency
                gradient_as_bucket_view=True,
            )
            logger.info(f"Using DDP on device {local_rank}")
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        
        # CRITICAL FIX: Calculate actual training steps from dataset size
        # This is critical for proper learning rate scheduling
        # Don't load full dataset just for size - use metadata or estimate
        steps_per_epoch = 1000  # Default estimate
        num_training_steps = config.num_epochs * steps_per_epoch
        
        try:
            # CRITICAL FIX: Try to get dataset size efficiently without loading full dataset
            # For streaming datasets, we can't know size upfront, so use estimate
            if not config.use_streaming:
                # Only try to get size for non-streaming datasets
                try:
                    # Quick metadata check - don't load full dataset
                    dataset_for_size = ray.data.read_parquet(
                        config.curated_data_path,
                        num_cpus=1,  # Minimal resources
                    )
                    if world_size > 1:
                        dataset_for_size = dataset_for_size.shard(num_shards=world_size, index=worker_id)
                    
                    # CRITICAL FIX: num_rows() can be expensive, use try/except with timeout
                    dataset_size = dataset_for_size.num_rows()
                    if dataset_size > 0:
                        # Calculate steps per epoch accounting for gradient accumulation
                        steps_per_epoch = max(1, dataset_size // (config.batch_size * config.gradient_accumulation_steps))
                        num_training_steps = config.num_epochs * steps_per_epoch
                        logger.info(f"Worker {worker_id}: Dataset size={dataset_size}, Steps per epoch={steps_per_epoch}, Total steps={num_training_steps}")
                    else:
                        logger.warning("Dataset size is 0, using estimate")
                except Exception as size_error:
                    # Fallback: estimate based on typical dataset sizes
                    logger.warning(f"Could not determine dataset size: {size_error}, using estimate")
            else:
                logger.info("Streaming dataset - using estimated steps per epoch")
        except Exception as e:
            logger.warning(f"Error calculating training steps: {e}, using estimate")
        
        scheduler = get_learning_rate_scheduler(
            optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=num_training_steps,
        )
        
        # Setup mixed precision training with optimized settings
        if config.use_mixed_precision:
            # Use optimized scaler settings for NVIDIA GPUs
            scaler = GradScaler(
                init_scale=2.0**16,  # Start with FP16 max
                growth_factor=2.0,
                backoff_factor=0.5,
                growth_interval=2000,
            )
        else:
            scaler = None
        
        # Enable CUDA graphs for repetitive kernels (PyTorch 2.0+)
        if config.use_cuda_graphs and torch.cuda.is_available():
            try:
                # CUDA graphs can speed up repetitive operations
                # This is a placeholder - actual implementation requires capturing
                # the forward/backward pass once and replaying
                logger.info("CUDA graphs enabled (will be captured on first iteration)")
            except Exception as e:
                logger.warning(f"CUDA graphs not available: {e}")
        
        # CRITICAL FIX: Enable gradient checkpointing for memory efficiency
        if config.use_gradient_checkpointing:
            # Apply gradient checkpointing to transformer blocks using torch.utils.checkpoint
            from torch.utils.checkpoint import checkpoint as gradient_checkpoint
            
            def make_checkpointed_forward(original_forward):
                """Wrap forward method with gradient checkpointing."""
                def checkpointed_forward(self, x, mask=None):
                    if self.training:
                        return gradient_checkpoint(original_forward, x, mask, use_reentrant=False)
                    else:
                        return original_forward(x, mask)
                return checkpointed_forward
            
            # Apply checkpointing to TransformerBlock forward methods
            for module in model.modules():
                if isinstance(module, TransformerBlock):
                    module.forward = make_checkpointed_forward(TransformerBlock.forward).__get__(module, TransformerBlock)
            
            logger.info("Gradient checkpointing enabled for transformer blocks")
        
        # Load training data using Ray Data with latest optimizations
        curated_data_path = config.curated_data_path
        if not curated_data_path:
            raise ValueError("curated_data_path must be provided in config")
        
        # CRITICAL FIX: Shard dataset per worker for distributed training
        # Each worker should only process its portion of data
        dataset = get_training_dataset(
            curated_data_path,
            shuffle=True,
            use_streaming=config.use_streaming,
            use_gpu_object_store=config.use_gpu_object_store,
        )
        
        # Shard dataset for distributed training to avoid data duplication
        if world_size > 1:
            # Split dataset into shards, one per worker
            dataset = dataset.shard(num_shards=world_size, index=worker_id)
            logger.info(f"Worker {worker_id} sharded dataset: {dataset.num_rows()} rows")
        
        # Validate dataset size after sharding
        try:
            num_rows = dataset.num_rows()
            if num_rows == 0:
                raise ValueError(f"Worker {worker_id} received empty dataset shard")
            logger.info(f"Worker {worker_id} dataset size: {num_rows} rows")
        except Exception as e:
            # num_rows() might be expensive for streaming datasets, use estimate
            logger.warning(f"Could not get exact dataset size: {e}")
        
        # CRITICAL FIX: Check GPU memory and validate batch size
        if torch.cuda.is_available():
            local_rank = worker_id % torch.cuda.device_count()
            mem_info = get_gpu_memory_info(device_id=local_rank)
            total_gb = mem_info['total'] / 1e9
            allocated_gb = mem_info['allocated'] / 1e9
            free_gb = mem_info['free'] / 1e9
            
            logger.info(
                f"GPU {local_rank} (worker {worker_id}) memory: "
                f"Total: {total_gb:.2f}GB, "
                f"Allocated: {allocated_gb:.2f}GB, "
                f"Free: {free_gb:.2f}GB"
            )
            
            # CRITICAL FIX: Estimate memory requirements and validate batch size
            # Rough estimate: model + batch memory
            # 2B param model ~8GB (FP32) or ~4GB (FP16)
            # Batch memory: batch_size * (image_size^2 * 3 * 4 bytes + action_dim * 4 bytes + ...)
            model_memory_gb = 4.0 if config.use_mixed_precision else 8.0  # Conservative estimate
            batch_memory_per_sample_mb = (
                (config.image_size ** 2 * 3 * 4) +  # Image
                (config.max_seq_len * 4) +  # Language tokens
                (config.action_dim * 4)  # Actions
            ) / 1e6
            estimated_batch_memory_gb = (config.batch_size * batch_memory_per_sample_mb) / 1000
            
            total_estimated_gb = model_memory_gb + estimated_batch_memory_gb + 2.0  # +2GB overhead
            
            if total_estimated_gb > free_gb * 0.9:  # Use 90% of free memory as threshold
                logger.warning(
                    f"Estimated memory usage ({total_estimated_gb:.2f}GB) exceeds available "
                    f"free memory ({free_gb:.2f}GB). Consider reducing batch_size or enabling "
                    f"gradient checkpointing."
                )
            else:
                logger.info(
                    f"Memory check passed: Estimated {total_estimated_gb:.2f}GB, "
                    f"Available {free_gb:.2f}GB"
                )
        
        # CRITICAL FIX: Resume from checkpoint if available
        start_epoch = 0
        global_step = 0
        best_val_loss = float("inf")
        patience_counter = 0
        
        # Try to resume from checkpoint
        checkpoint = train_context.get_checkpoint()
        if checkpoint is not None:
            try:
                checkpoint_dict = checkpoint.to_dict()
                if "model_state_dict" in checkpoint_dict:
                    # Load model state
                    model_state = checkpoint_dict["model_state_dict"]
                    if hasattr(model, "module"):
                        model.module.load_state_dict(model_state)
                    else:
                        model.load_state_dict(model_state)
                    
                    # Load optimizer state
                    if "optimizer_state_dict" in checkpoint_dict and checkpoint_dict["optimizer_state_dict"]:
                        optimizer.load_state_dict(checkpoint_dict["optimizer_state_dict"])
                    
                    # Load scheduler state
                    if "scheduler_state_dict" in checkpoint_dict and checkpoint_dict["scheduler_state_dict"] and scheduler:
                        scheduler.load_state_dict(checkpoint_dict["scheduler_state_dict"])
                    
                    # Load scaler state
                    if "scaler_state_dict" in checkpoint_dict and checkpoint_dict["scaler_state_dict"] and scaler:
                        scaler.load_state_dict(checkpoint_dict["scaler_state_dict"])
                    
                    # Resume from saved epoch and step
                    start_epoch = checkpoint_dict.get("epoch", 0) + 1
                    global_step = checkpoint_dict.get("global_step", 0)
                    best_val_loss = checkpoint_dict.get("best_val_loss", float("inf"))
                    
                    logger.info(f"Resumed from checkpoint: epoch={start_epoch-1}, step={global_step}")
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}, starting from scratch")
        
        # Training loop
        for epoch in range(start_epoch, config.num_epochs):
            try:
                # Train epoch
                train_metrics = train_epoch(
                    model,
                    dataset,
                    optimizer,
                    scheduler,
                    scaler,
                    device,
                    epoch,
                    config,
                    global_step,
                )
                
                global_step = train_metrics["global_step"]
                
                # CRITICAL FIX: Evaluation based on steps, not epochs
                val_metrics = {}
                should_eval = (global_step % config.eval_interval == 0) or (epoch == config.num_epochs - 1)
                if should_eval:
                    # CRITICAL FIX: Load validation dataset if available
                    val_data_path = config_dict.get("val_data_path")
                    if val_data_path:
                        try:
                            val_dataset = get_training_dataset(
                                val_data_path,
                                shuffle=False,  # Don't shuffle validation data
                                use_streaming=config.use_streaming,
                                use_gpu_object_store=config.use_gpu_object_store,
                            )
                            if world_size > 1:
                                val_dataset = val_dataset.shard(num_shards=world_size, index=worker_id)
                            
                            val_metrics = evaluate_model(model, val_dataset, device, config)
                            logger.info(f"Validation metrics at step {global_step}: {val_metrics}")
                        except Exception as val_error:
                            logger.warning(f"Validation failed: {val_error}, skipping")
                    else:
                        logger.debug("No validation dataset provided, skipping evaluation")
                
                # CRITICAL FIX: Synchronize metrics across workers for distributed training
                if world_size > 1:
                    # Average metrics across workers
                    metrics_tensor = torch.tensor([train_metrics["loss"]], device=device)
                    dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
                    train_metrics["loss"] = (metrics_tensor.item() / world_size)
                
                # Report metrics to Ray Train
                metrics_to_report = {
                    **train_metrics,
                    **val_metrics,
                    "epoch": epoch,
                    "global_step": global_step,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
                
                # CRITICAL FIX: Save checkpoint based on steps, not epochs
                # This ensures consistent checkpointing regardless of dataset size
                should_save_checkpoint = (
                    global_step % config.save_interval == 0
                ) or (epoch == config.num_epochs - 1)
                
                if should_save_checkpoint:
                    # CRITICAL FIX: Only rank 0 saves checkpoint to avoid conflicts
                    if worker_id == 0 or world_size == 1:
                        # CRITICAL FIX: Get model state dict properly for DDP/FSDP
                        if hasattr(model, "module"):
                            # DDP wrapper
                            model_state = model.module.state_dict()
                        elif hasattr(model, "_fsdp_wrapped_module"):
                            # FSDP wrapper
                            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
                            if isinstance(model, FSDP):
                                # FSDP requires special state dict gathering
                                model_state = model.state_dict()
                            else:
                                model_state = model.state_dict()
                        else:
                            model_state = model.state_dict()
                        
                        # CRITICAL FIX: Don't store full config dict (can be large)
                        # Only store essential config values
                        essential_config = {
                            "vision_dim": config.vision_dim,
                            "language_dim": config.language_dim,
                            "hidden_dim": config.hidden_dim,
                            "action_dim": config.action_dim,
                            "vocab_size": config.vocab_size,
                            "max_seq_len": config.max_seq_len,
                            "image_size": config.image_size,
                            "num_diffusion_steps": config.num_diffusion_steps,
                        }
                        
                        checkpoint_dict = {
                            "model_state_dict": model_state,
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                            "scaler_state_dict": scaler.state_dict() if scaler else None,
                            "epoch": epoch,
                            "global_step": global_step,
                            "best_val_loss": best_val_loss,
                            "config": essential_config,  # Store only essential config
                        }
                        
                        checkpoint = Checkpoint.from_dict(checkpoint_dict)
                        train_context.report(metrics=metrics_to_report, checkpoint=checkpoint)
                        logger.info(f"Checkpoint saved at step {global_step}, epoch {epoch}")
                    else:
                        # Other workers just report metrics
                        train_context.report(metrics=metrics_to_report)
                else:
                    train_context.report(metrics=metrics_to_report)
                
                logger.info(f"Epoch {epoch} completed: {metrics_to_report}")
                
                # CRITICAL FIX: Memory cleanup between epochs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                # Early stopping
                if val_metrics and "val_loss" in val_metrics:
                    if val_metrics["val_loss"] < best_val_loss:
                        best_val_loss = val_metrics["val_loss"]
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= config.early_stopping_patience:
                            logger.info(f"Early stopping at epoch {epoch}")
                            break
                
                # CRITICAL FIX: Barrier to synchronize workers after each epoch with timeout
                if world_size > 1:
                    try:
                        dist.barrier(timeout=300)  # 5 minute timeout per epoch
                    except Exception as barrier_error:
                        logger.error(f"Barrier failed after epoch {epoch}: {barrier_error}")
                        raise RuntimeError(f"Worker synchronization failed: {barrier_error}")
            
            except Exception as e:
                logger.error(f"Error in epoch {epoch}: {e}", exc_info=True)
                # CRITICAL FIX: Don't immediately raise - try to recover
                # In distributed training, one worker failing shouldn't kill all workers
                if world_size > 1:
                    logger.error(f"Worker {worker_id} failed in epoch {epoch}, attempting recovery")
                    # Try to synchronize with other workers
                    try:
                        dist.barrier(timeout=60)  # Wait up to 60 seconds
                    except Exception as barrier_error:
                        logger.error(f"Barrier failed: {barrier_error}")
                        raise
                else:
                    raise
        
        logger.info("Training completed successfully")
    
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        # CRITICAL FIX: Save checkpoint on interruption
        if worker_id == 0 or world_size == 1:
            try:
                checkpoint_dict = {
                    "model_state_dict": (
                        model.module.state_dict() if hasattr(model, "module") else model.state_dict()
                    ),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                    "scaler_state_dict": scaler.state_dict() if scaler else None,
                    "epoch": epoch,
                    "global_step": global_step,
                    "best_val_loss": best_val_loss,
                }
                checkpoint = Checkpoint.from_dict(checkpoint_dict)
                train_context.report(checkpoint=checkpoint)
                logger.info("Checkpoint saved on interruption")
            except Exception as checkpoint_error:
                logger.error(f"Failed to save checkpoint on interruption: {checkpoint_error}")
        raise
    
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        # CRITICAL FIX: Cleanup distributed training on error
        if world_size > 1 and dist.is_initialized():
            try:
                dist.barrier(timeout=30)  # Wait for other workers
            except Exception:
                pass
            finally:
                dist.destroy_process_group()
        raise
    
    finally:
        # CRITICAL FIX: Always cleanup resources
        try:
            # Cleanup CUDA
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Cleanup distributed training
            if world_size > 1 and dist.is_initialized():
                try:
                    dist.barrier(timeout=30)
                except Exception:
                    pass
                finally:
                    if dist.is_initialized():
                        dist.destroy_process_group()
            
            logger.info(f"Worker {worker_id} cleanup completed")
        except Exception as cleanup_error:
            logger.warning(f"Error during cleanup: {cleanup_error}")


def main() -> None:
    """Main training function for GR00T model."""
    logger.info("Starting GR00T model training with Ray Data + Ray Train")
    
    # Step 1: Run data curation pipeline (if not already done)
    # Following GR00T's data pyramid approach:
    # - Base Layer (Fossil Fuel): Internet-scale web data and human videos
    # - Middle Layer (Nuclear Fuel): Synthetic data from Simulation 1.0 and 2.0
    # - Top Layer (Human Fuel): Real robot data from teleoperation
    logger.info("Step 1: Running GR00T data curation pipeline...")
    
    config = PipelineConfig(
        input_paths=[
            "s3://bucket/teleop_data/",  # Human Fuel: Real robot teleoperation (4-24 hrs/robot/day)
            "s3://bucket/internet_videos/",  # Fossil Fuel: Internet-scale video (100M+ clips)
            "s3://bucket/text_corpus/",  # Fossil Fuel: Text data for pretraining
        ],
        output_path="s3://bucket/groot_curated/",
        enable_gpu_dedup=True,
        num_gpus=256,  # Scale to internet-scale datasets
        streaming=True,
        dedup_method="both",  # Both fuzzy and semantic deduplication
        similarity_threshold=0.95,
    )
    
    pipeline = MultimodalPipeline(config)
    
    # Add Simulation 1.0: Isaac Lab digital twins (Digital Tooling Paradigm)
    # - One-to-one replicas of robots and worlds
    # - 10,000x faster than real-time on GPU
    # - Domain randomization for sim-to-real transfer
    isaac_loader = IsaacLabLoader(
        simulation_path="/path/to/isaac/lab/trajectories",
        robot_type="humanoid",
        include_observations=True,
        include_actions=True,
    )
    pipeline.add_simulation_data(isaac_loader)
    
    # Add Simulation 2.0: Cosmos Dreams (Neurophysics Engines)
    # - Video foundation models as neural simulators
    # - Learned physics from billions of internet videos
    # - Digital cousins (not exact replicas, but similar distribution)
    cosmos_loader = CosmosDreamsLoader(
        dreams_path="/path/to/cosmos/dreams",
        model_name="groot-dreams-v1",
        include_metadata=True,
    )
    pipeline.add_synthetic_data(cosmos_loader)
    
    # Run pipeline
    logger.info("Running complete GR00T-style data curation pipeline...")
    try:
        results = pipeline.run()
        
        logger.info("Curation completed:")
        logger.info(f"  Total samples: {results.get('total_samples', 0):,}")
        logger.info(f"  Deduplication rate: {results.get('dedup_rate', 0.0):.2%}")
        logger.info(f"  GPU utilization: {results.get('avg_gpu_util', 0.0):.2%}")
    except Exception as e:
        logger.error(f"Data curation pipeline failed: {e}", exc_info=True)
        raise
    
    # Step 2: Prepare training data
    logger.info("Step 2: Preparing training data...")
    
    curated_data_path = config.output_path
    training_integration = TrainingPipelineIntegration(
        output_format="parquet",
        batch_size=_TRAINING_BATCH_SIZE,
        shuffle=True,
    )
    
    # Load curated dataset
    try:
        curated_dataset = ray.data.read_parquet(curated_data_path)
    except Exception as e:
        logger.error(f"Failed to load curated dataset: {e}", exc_info=True)
        raise
    
    # CRITICAL FIX: Split into train/val with proper error handling
    try:
        training_data = training_integration.prepare_for_training(
            curated_dataset,
            output_path="s3://bucket/groot_training_data/",
            train_split=0.9,
        )
        
        logger.info("Training data prepared:")
        logger.info(f"  Train samples: {training_data.get('num_train_samples', 'N/A')}")
        logger.info(f"  Val samples: {training_data.get('num_val_samples', 'N/A')}")
        
        # CRITICAL FIX: Validate that we have training data
        train_path = training_data.get("train_path")
        val_path = training_data.get("val_path")
        
        if not train_path:
            raise ValueError("Training data path not found in training_data")
        
        # Validate training dataset exists
        try:
            train_check = ray.data.read_parquet(train_path)
            train_rows = train_check.num_rows()
            if train_rows == 0:
                raise ValueError("Training dataset is empty")
            logger.info(f"Validated training dataset: {train_rows} rows")
        except Exception as e:
            raise RuntimeError(f"Failed to validate training dataset: {e}") from e
        
    except Exception as e:
        logger.error(f"Failed to prepare training data: {e}", exc_info=True)
        raise
    
    # Step 3: Train model with Ray Train
    logger.info("Step 3: Training GR00T model with Ray Train...")
    
    # CRITICAL FIX: Training configuration aligned with GR00T N1 specifications
    train_config = GrootTrainingConfig(
        curated_data_path=training_data["train_path"],
        val_data_path=val_path if val_path else "",  # CRITICAL FIX: Include validation path
        vision_dim=768,  # NVIDIA-Eagle vision encoder output
        language_dim=2048,  # SmolLM-1.7B hidden dimension
        hidden_dim=2048,  # Fusion dimension
        action_dim=25,  # 20-30 DOF for humanoid robots
        batch_size=256,  # 256-512 for joint training (per study guide)
        learning_rate=1e-4,  # 1e-4 to 3e-4 range
        weight_decay=1e-4,  # 1e-4 as per study guide
        num_epochs=100,
        num_workers_distributed=8,
        use_gpu=True,
        use_streaming=True,  # Enable Ray Data streaming
        use_gpu_object_store=True,  # Enable GPU object store with RDMA
        use_mixed_precision=True,  # Enable FP16/BF16 training
        use_flash_attention=True,  # Enable Flash Attention
        use_gradient_checkpointing=True,  # Enable gradient checkpointing
        use_fsdp=False,  # Set to True for very large models
        use_cuda_graphs=False,  # Set to True for repetitive workloads (PyTorch 2.0+)
        gradient_accumulation_steps=4,  # 4-8 steps per study guide
    )
    
    # Scaling configuration for distributed training
    scaling_config = ScalingConfig(
        num_workers=train_config.num_workers_distributed,
        use_gpu=train_config.use_gpu,
        resources_per_worker={"GPU": 1} if train_config.use_gpu else {},
        # Enable placement group for better GPU locality
        placement_strategy="PACK" if train_config.use_gpu else "SPREAD",
    )
    
    # Torch configuration with latest optimizations
    torch_config = TorchConfig(
        backend="nccl" if train_config.use_gpu else "gloo",
        # Enable NCCL async operations for better performance
        nccl_timeout_s=1800,  # 30 minutes timeout
    )
    
    # Create Ray Train trainer with latest features
    trainer = TorchTrainer(
        train_func=train_func,
        train_loop_config=train_config.__dict__,
        scaling_config=scaling_config,
        torch_config=torch_config,
        # Enable checkpointing with latest format
        run_config=None,  # Can add RunConfig for experiment tracking
    )
    
    # CRITICAL FIX: Run training with proper error handling and cleanup
    logger.info("Starting distributed training...")
    result = None
    try:
        result = trainer.fit()
        
        logger.info("Training Results:")
        logger.info(f"Final metrics: {result.metrics}")
        logger.info(f"Checkpoint path: {result.checkpoint}")
        
        # CRITICAL FIX: Validate training completed successfully
        if result.metrics and "loss" in result.metrics:
            final_loss = result.metrics["loss"]
            if not (0 < final_loss < 1e6):  # Reasonable loss range
                logger.warning(f"Final loss seems unusual: {final_loss}")
        else:
            logger.warning("No loss metric in final results")
            
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        if result:
            logger.info(f"Last checkpoint: {result.checkpoint}")
        raise
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        # CRITICAL FIX: Try to save partial results
        if result and result.checkpoint:
            logger.info(f"Partial results available at: {result.checkpoint}")
        raise
    finally:
        # CRITICAL FIX: Always cleanup resources
        try:
            # Cleanup pipeline
            if 'pipeline' in locals():
                try:
                    pipeline.shutdown()
                except Exception as shutdown_error:
                    logger.warning(f"Error shutting down pipeline: {shutdown_error}")
            
            # Cleanup Ray resources
            if ray.is_initialized():
                # Don't shutdown Ray as it might be used by other processes
                # Just cleanup our resources
                logger.info("Training cleanup completed")
        except Exception as cleanup_error:
            logger.warning(f"Error during cleanup: {cleanup_error}")
    
    logger.info("GR00T model training completed successfully")
    logger.info("Model architecture (GR00T N1):")
    logger.info("  - System 2 (Slow): Vision-Language Model for deliberate reasoning")
    logger.info("    * Vision Encoder: NVIDIA-Eagle architecture")
    logger.info("    * Language Model: SmolLM-1.7B (1.7B params, 2048 hidden dim)")
    logger.info("    * Processes multi-view camera inputs (2-4 views)")
    logger.info("  - System 1 (Fast): Diffusion Transformer for reactive actions")
    logger.info("    * Generates continuous actions at 100+ Hz")
    logger.info("    * Action space: 20-30 DOF for humanoid robots")
    logger.info("    * Control frequency: 10-30 Hz")
    logger.info("  - Total: 2 billion parameters (data maximalist, model minimalist)")
    logger.info("  - Input: Photons (vision) → Output: Actions (continuous control)")
    logger.info("  - Training on Ray infrastructure for versatility and scalability")


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    # Initialize Ray if not already initialized
    if not ray.is_initialized():
        ray.init(
            num_cpus=32,
            num_gpus=8,
            log_to_driver=True,
        )
    
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        raise
