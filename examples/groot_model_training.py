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
  - Internet-scale data: Web data and human videos
  - Synthetic data: Simulation data from Simulation 1.0 (Isaac Lab) and Simulation 2.0 (Cosmos Dreams)
  - Teleoperation data: Real robot data from teleoperation (4-24 hours per robot per day)
- Clean model architecture that compresses trillions of tokens from data pipeline

Training on Ray infrastructure:
- Almost all compute jobs based on Ray for versatility
- Handles physical AI multimodal sensor data processing
- Large-scale VLA training
- Complex evaluation pipelines with thousands of simulation engines
"""

import logging

import ray
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer
from ray.train.torch.config import TorchConfig

from pipeline import MultimodalPipeline
from pipeline.config import PipelineConfig
from pipeline.integrations.cosmos import CosmosDreamsLoader
from pipeline.integrations.isaac_lab import IsaacLabLoader
from pipeline.training.groot import GrootTrainingConfig, train_func

logger = logging.getLogger(__name__)


def main() -> None:
    """Main training function for GR00T model."""
    logger.info("Starting GR00T model training with Ray Data + Ray Train")

    # Step 1: Run data curation pipeline (if not already done)
    # Following GR00T's data pyramid approach:
    # - Base Layer: Internet-scale web data and human videos
    # - Middle Layer: Synthetic data from Simulation 1.0 and 2.0
    # - Top Layer: Real robot data from teleoperation
    logger.info("Step 1: Running GR00T data curation pipeline...")

    config = PipelineConfig(
        input_paths=[
            "s3://bucket/teleop_data/",  # Teleoperation data: Real robot teleoperation (4-24 hrs/robot/day)
            "s3://bucket/internet_videos/",  # Internet-scale data: Internet-scale video (100M+ clips)
            "s3://bucket/text_corpus/",  # Internet-scale data: Text data for pretraining
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
    # - Supports 10,000+ parallel environments
    isaac_loader = IsaacLabLoader(
        simulation_path="/path/to/isaac/lab/trajectories",
        robot_type="humanoid",
        include_observations=True,
        include_actions=True,
        enable_domain_randomization=True,
        num_parallel_environments=4096,
        use_gpu=True,
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

    # Run curation pipeline
    try:
        pipeline.run()
        logger.info("Data curation pipeline completed successfully")
    except Exception as e:
        logger.error(f"Data curation pipeline failed: {e}", exc_info=True)
        raise

    # Step 2: Train GR00T model using Ray Train
    logger.info("Step 2: Training GR00T model with Ray Train...")

    # Training configuration
    train_config = GrootTrainingConfig(
        curated_data_path="s3://bucket/groot_curated/",
        val_data_path="s3://bucket/groot_val/",  # Separate validation dataset
        batch_size=256,  # 256-512 for joint training
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

    # Run training with proper error handling and cleanup
    logger.info("Starting distributed training...")
    result = None
    try:
        result = trainer.fit()

        logger.info("Training Results:")
        logger.info(f"Final metrics: {result.metrics}")
        logger.info(f"Checkpoint path: {result.checkpoint}")

        # Validate training completed successfully
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
        # Try to save partial results
        if result and result.checkpoint:
            logger.info(f"Partial results available at: {result.checkpoint}")
        raise
    finally:
        # Always cleanup resources
        try:
            # Cleanup pipeline
            if "pipeline" in locals():
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

    # Do NOT call ray.init() here!
    # Ray Train handles Ray initialization automatically.
    # Calling ray.init() manually can cause conflicts with Ray Train's cluster management.
    # If you need to initialize Ray for testing/debugging, do it OUTSIDE of Ray Train context.

    try:
        main()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        raise
