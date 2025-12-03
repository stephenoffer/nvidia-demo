"""GR00T Data Loading and Preprocessing.

Functions for loading training data from Ray Data and preprocessing batches.
"""

import logging
import os
from typing import Any, Dict, Optional

import numpy as np
import ray
import torch
import torch.nn.functional as F
from ray.data import Dataset
from ray.data.context import DataContext

from pipeline.training.groot.config import DEFAULT_IMAGE_SIZE, DEFAULT_MAX_SEQ_LEN

logger = logging.getLogger(__name__)

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
    # Extract and preprocess images with multi-view support
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
        
        # Handle various input shapes
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
        
        # Resize to target size if needed
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
        
        # Create attention mask [B, L] where 1 = valid token, 0 = padding
        # Use float for proper masking in attention
        language_mask = (language_tokens != 0).float()
    else:
        # Create dummy tokens if missing
        language_tokens = torch.zeros(
            images.shape[0], max_seq_len, dtype=torch.long, device=device
        )
        language_mask = torch.zeros(
            images.shape[0], max_seq_len, dtype=torch.float32,             device=device
        )
    
    # Extract and preprocess actions with dimension validation
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
        
        # Validate action dimension matches expected
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
        
        # Check for NaN/Inf in actions
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
    
    # Handle various path types (local, S3, GCS, etc.)
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
    
    # Don't call num_rows() on streaming datasets - it can fail or be expensive
    # For streaming datasets, num_rows() may not be available or may trigger full materialization
    if not use_streaming:
        try:
            num_rows = dataset.num_rows()
            if num_rows == 0:
                raise ValueError(f"Dataset is empty: {curated_data_path}")
            logger.info(f"Training dataset prepared successfully: {num_rows} rows")
        except Exception as e:
            logger.warning(f"Could not get dataset size (may be streaming): {e}")
    else:
        logger.info("Training dataset prepared (streaming mode - size unknown)")
    
    # Shuffle for training (uses distributed shuffling for large datasets)
    if shuffle:
        dataset = dataset.random_shuffle(seed=seed)
    
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
        
        # Validate image tensor
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
        
        # Validate action tensor
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
        
        # Validate batch size consistency
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

