"""GR00T Training Functions.

Core training loop, evaluation, and Ray Train integration.
"""

import logging
import time
from typing import Any, Dict, Optional

import ray
import torch
import torch.distributed as dist
import torch.nn as nn
from ray.data import Dataset
from ray.train import Checkpoint, get_context
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP

from pipeline.training.groot.config import GrootTrainingConfig
from pipeline.training.groot.data import get_training_dataset, preprocess_batch, validate_batch
from pipeline.training.groot.losses import compute_diffusion_loss
from pipeline.training.groot.model import GrootVLA
from pipeline.training.groot.schedulers import get_learning_rate_scheduler

# GPU memory utilities
try:
    from pipeline.utils.gpu.memory import get_gpu_memory_info
except ImportError:
    def get_gpu_memory_info(device_id=None):
        return {"total": 0, "allocated": 0, "reserved": 0, "free": 0}

logger = logging.getLogger(__name__)


def train_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    scaler: Optional[GradScaler],
    config: GrootTrainingConfig,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    """Train for one epoch.

    Args:
        model: Model to train
        dataloader: Training dataloader
        optimizer: Optimizer
        scheduler: Learning rate scheduler (optional)
        scaler: Gradient scaler for mixed precision (optional)
        config: Training configuration
        device: Device to train on
        epoch: Current epoch number

    Returns:
        Dictionary of training metrics
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    start_time = time.time()

    for batch_idx, batch in enumerate(dataloader):
        try:
            # Preprocess batch
            preprocessed = preprocess_batch(
                batch,
                image_size=config.image_size,
                max_seq_len=config.max_seq_len,
                device=device,
            )

            # Sample timesteps for diffusion
            timesteps = torch.randint(
                0,
                config.num_diffusion_steps,
                (preprocessed["images"].shape[0],),
                device=device,
            )

            # Forward pass with mixed precision
            with autocast(enabled=config.use_mixed_precision):
                outputs = model(
                    images=preprocessed["images"],
                    language_tokens=preprocessed["language_tokens"],
                    actions=preprocessed["actions"],
                    timesteps=timesteps,
                    language_mask=preprocessed["language_mask"],
                )

                # Compute loss
                loss = compute_diffusion_loss(
                    predicted_noise=outputs["predicted_noise"],
                    target_noise=outputs["actual_noise"],
                    reduction="mean",
                )

                # Scale loss for gradient accumulation
                loss = loss / config.gradient_accumulation_steps

            # Backward pass
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                # Gradient clipping
                if config.gradient_clip_norm > 0:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.gradient_clip_norm
                    )

                # Optimizer step
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad()

                # Learning rate scheduler step
                if scheduler is not None:
                    scheduler.step()

            total_loss += loss.item() * config.gradient_accumulation_steps
            num_batches += 1

            # Logging
            if batch_idx % 100 == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                logger.info(
                    f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item() * config.gradient_accumulation_steps:.4f}, "
                    f"LR: {current_lr:.6f}"
                )

        except Exception as e:
            logger.error(f"Error in training batch {batch_idx}: {e}", exc_info=True)
            raise

    avg_loss = total_loss / max(num_batches, 1)
    elapsed_time = time.time() - start_time

    metrics = {
        "loss": avg_loss,
        "num_batches": num_batches,
        "elapsed_time": elapsed_time,
    }

    return metrics


def evaluate_model(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    config: GrootTrainingConfig,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate model on validation set.

    Args:
        model: Model to evaluate
        dataloader: Validation dataloader
        config: Training configuration
        device: Device to evaluate on

    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    start_time = time.time()

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            try:
                # Preprocess batch
                preprocessed = preprocess_batch(
                    batch,
                    image_size=config.image_size,
                    max_seq_len=config.max_seq_len,
                    device=device,
                )

                # Sample timesteps for diffusion
                timesteps = torch.randint(
                    0,
                    config.num_diffusion_steps,
                    (preprocessed["images"].shape[0],),
                    device=device,
                )

                # Forward pass
                with autocast(enabled=config.use_mixed_precision):
                    outputs = model(
                        images=preprocessed["images"],
                        language_tokens=preprocessed["language_tokens"],
                        actions=preprocessed["actions"],
                        timesteps=timesteps,
                        language_mask=preprocessed["language_mask"],
                    )

                    # Compute loss
                    loss = compute_diffusion_loss(
                        predicted_noise=outputs["predicted_noise"],
                        target_noise=outputs["actual_noise"],
                        reduction="mean",
                    )

                total_loss += loss.item()
                num_batches += 1

            except Exception as e:
                logger.error(f"Error in evaluation batch {batch_idx}: {e}", exc_info=True)
                raise

    avg_loss = total_loss / max(num_batches, 1)
    elapsed_time = time.time() - start_time

    metrics = {
        "val_loss": avg_loss,
        "num_batches": num_batches,
        "elapsed_time": elapsed_time,
    }

    return metrics


def train_func(config_dict: Dict[str, Any]) -> None:
    """Main training function for Ray Train.

    Called by Ray Train for distributed training.
    Handles model initialization, data loading, training loop, and checkpointing.

    Args:
        config_dict: Dictionary containing training configuration
    """
    # Get Ray Train context
    train_context = get_context()
    world_rank = train_context.get_world_rank()
    world_size = train_context.get_world_size()
    local_rank = train_context.get_local_rank()

    # Set device
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)

    # Initialize distributed training if needed
    if world_size > 1:
        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            rank=world_rank,
            world_size=world_size,
        )

    # Create config from dict
    config = GrootTrainingConfig(**config_dict)

    # Log GPU memory info
    if torch.cuda.is_available():
        gpu_info = get_gpu_memory_info(device_id=local_rank)
        logger.info(
            f"Rank {world_rank}: GPU {local_rank} memory - "
            f"Total: {gpu_info['total'] / 1024**3:.2f} GB, "
            f"Allocated: {gpu_info['allocated'] / 1024**3:.2f} GB"
        )

    # Initialize model
    logger.info(f"Rank {world_rank}: Initializing GR00T model...")
    model = GrootVLA(config=config)
    model = model.to(device)

    # Wrap with DDP if distributed
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # Create learning rate scheduler
    num_training_steps = config.num_epochs * 1000  # Estimate steps per epoch
    scheduler = get_learning_rate_scheduler(
        optimizer=optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=num_training_steps,
        min_lr=0.0,
    )

    # Create gradient scaler for mixed precision
    scaler = GradScaler() if config.use_mixed_precision else None

    # Load training dataset
    logger.info(f"Rank {world_rank}: Loading training dataset...")
    train_dataset = get_training_dataset(
        curated_data_path=config.curated_data_path,
        shuffle=True,
        use_streaming=config.use_streaming,
        use_gpu_object_store=config.use_gpu_object_store,
    )

    # Create dataloader
    # Ray Data integration would be done here
    # For now, we'll use a simple iterator
    train_iter = train_dataset.iter_batches(
        batch_size=config.batch_size,
        prefetch_batches=2,
    )

    # Training loop
    logger.info(f"Rank {world_rank}: Starting training...")
    global_step = 0

    for epoch in range(config.num_epochs):
        epoch_start_time = time.time()

        # Create epoch iterator
        epoch_batches = []
        for batch in train_iter:
            epoch_batches.append(batch)
            if len(epoch_batches) >= 1000:  # Limit batches per epoch for demo
                break

        # Create simple dataloader-like structure
        class SimpleDataLoader:
            def __init__(self, batches):
                self.batches = batches

            def __iter__(self):
                return iter(self.batches)

            def __len__(self):
                return len(self.batches)

        dataloader = SimpleDataLoader(epoch_batches)

        # Train epoch
        train_metrics = train_epoch(
            model=model.module if isinstance(model, DDP) else model,
            dataloader=dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            config=config,
            device=device,
            epoch=epoch,
        )

        global_step += len(dataloader)

        # Evaluation
        val_metrics = {}
        if config.val_data_path and epoch % (config.eval_interval // len(dataloader)) == 0:
            logger.info(f"Rank {world_rank}: Running evaluation...")
            val_dataset = get_training_dataset(
                curated_data_path=config.val_data_path,
                shuffle=False,
                use_streaming=config.use_streaming,
                use_gpu_object_store=config.use_gpu_object_store,
            )
            val_iter = val_dataset.iter_batches(
                batch_size=config.batch_size,
                prefetch_batches=2,
            )
            val_batches = list(val_iter)[:100]  # Limit for demo
            val_dataloader = SimpleDataLoader(val_batches)

            val_metrics = evaluate_model(
                model=model.module if isinstance(model, DDP) else model,
                dataloader=val_dataloader,
                config=config,
                device=device,
            )

        # Combine metrics
        metrics = {
            **train_metrics,
            **val_metrics,
            "epoch": epoch,
            "global_step": global_step,
        }

        # Report metrics to Ray Train
        train_context.report(metrics=metrics)

        # Save checkpoint
        if epoch % (config.save_interval // len(dataloader)) == 0:
            checkpoint_data = {
                "model_state_dict": (
                    model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
                ),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                "epoch": epoch,
                "global_step": global_step,
                "config": config_dict,
            }
            checkpoint = Checkpoint.from_dict(checkpoint_data)
            train_context.report(checkpoint=checkpoint, metrics=metrics)

        epoch_time = time.time() - epoch_start_time
        logger.info(
            f"Rank {world_rank}: Epoch {epoch} completed in {epoch_time:.2f}s - "
            f"Train Loss: {train_metrics['loss']:.4f}, "
            f"Val Loss: {val_metrics.get('val_loss', 'N/A')}"
        )

    logger.info(f"Rank {world_rank}: Training completed!")

    # Cleanup distributed training
    if world_size > 1:
        dist.destroy_process_group()
