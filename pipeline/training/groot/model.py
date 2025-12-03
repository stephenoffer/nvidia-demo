"""GR00T Model Architecture.

Complete model implementation including:
- System 2: Vision-Language Model (VLM)
- System 1: Diffusion Transformer for action generation
- Supporting components (encoders, attention, etc.)
"""

import logging
import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.training.groot.config import (
    DEFAULT_ACTION_DIM,
    DEFAULT_ATTENTION_HEADS,
    DEFAULT_ATTENTION_LAYERS,
    DEFAULT_BETA_END,
    DEFAULT_BETA_START,
    DEFAULT_DROPOUT,
    DEFAULT_HIDDEN_DIM,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_LANGUAGE_DIM,
    DEFAULT_MAX_SEQ_LEN,
    DEFAULT_NUM_DIFFUSION_STEPS,
    DEFAULT_VISION_DIM,
    DEFAULT_VOCAB_SIZE,
    GrootTrainingConfig,
)

logger = logging.getLogger(__name__)

class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequences.
    
    Supports longer sequences by computing on-the-fly if needed.
    GR00T uses context lengths up to 4096 tokens.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        """Initialize positional encoding.
        
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length (should be >= max_seq_len)
        """
        super().__init__()
        
        # Ensure max_len is sufficient for GR00T's context length
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
        
        # Class token and positional encoding
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
        
        # Handle multi-view inputs (2-4 camera views)
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
        
        # Ensure positional encoding can handle sequence length
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
        
        # Average across views if multi-view input
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
        
        # Positional encoding with sufficient max_len
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
        
        # Global average pooling with proper mask handling
        x = self.norm(x)
        if mask is not None:
            # Handle both 1D and 2D masks
            if len(mask.shape) == 2:
                # [B, L] mask - use for pooling
                # Ensure mask is boolean or convert to float
                if mask.dtype != torch.float32 and mask.dtype != torch.bool:
                    mask = mask.float()
                elif mask.dtype == torch.bool:
                    mask = mask.float()
                
                mask_expanded = mask.unsqueeze(-1)  # [B, L, 1]
                masked_x = x * mask_expanded  # [B, L, d_model]
                
                # Avoid division by zero
                mask_sum = mask.sum(dim=1, keepdim=True).float().clamp(min=1.0)  # [B, 1]
                lang_features = masked_x.sum(dim=1) / mask_sum  # [B, d_model]
            else:
                # [B, L, L] mask - use diagonal or sum
                # Convert to float if needed
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
        
        # Denoising loop with proper bounds checking
        for i in range(num_steps - 1, -1, -1):
            t = i * step_size
            # Ensure t is within valid range [0, num_diffusion_steps-1]
            t = max(0, min(t, self.num_diffusion_steps - 1))
            
            timestep = torch.full((batch_size,), t, device=device)
            timestep_emb = timestep.float().unsqueeze(-1)  # [B, 1]
            
            # Predict noise
            predicted_noise = self.diffusion_model(actions, timestep_emb, context)
            
            # Denoise using DDPM sampling with bounds checking
            alpha_t = self.alphas[t]
            alpha_cumprod_t = self.alphas_cumprod[t]
            
            # Add epsilon to avoid division by zero
            sqrt_alpha_cumprod_t = torch.sqrt(alpha_cumprod_t + 1e-8)
            sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]
            
            # Predict x_0 from noisy actions
            pred_x0 = (actions - sqrt_one_minus_alpha_cumprod_t * predicted_noise) / sqrt_alpha_cumprod_t
            
            # Compute previous step with correct indexing
            if i > 0:  # Not the final step
                # Ensure t is within valid range for posterior_variance
                t_clamped = min(t, len(self.posterior_variance) - 1)
                posterior_variance = self.posterior_variance[t_clamped]
                
                # Clamp posterior variance to avoid numerical issues
                posterior_variance = torch.clamp(posterior_variance, min=1e-20, max=1.0)
                
                # Sample noise for stochastic step
                noise = torch.randn_like(actions)
                
                # Compute mean of posterior distribution with numerical stability
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
            predicted_noise, actual_noise = self.system1(
                reasoning_features=reasoning,
                actions=actions,
                timesteps=timesteps,
            )
            return {
                "predicted_noise": predicted_noise,
                "actual_noise": actual_noise,
                "reasoning": reasoning,
            }
        else:
            sampled_actions, _ = self.system1(reasoning_features=reasoning)
            return {
                "actions": sampled_actions,
                "reasoning": reasoning,
            }
