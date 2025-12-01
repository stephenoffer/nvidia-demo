"""Checkpointing and recovery utilities for pipeline state.

Enables pipeline recovery from failures by checkpointing intermediate state.
Critical for GR00T: Large-scale processing requires checkpointing to avoid
losing progress on failures.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional


import ray
from ray.data import Dataset

logger = logging.getLogger(__name__)


class PipelineCheckpoint:
    """Manages pipeline checkpointing and recovery.

    Checkpoints pipeline state at configurable intervals to enable
    recovery from failures. Consolidates functionality from CheckpointRecovery.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        checkpoint_interval: int = 1000,  # Checkpoint every N batches
        max_checkpoints: int = 10,  # Maximum number of checkpoints to keep
    ):
        """Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoints
            checkpoint_interval: Checkpoint every N batches
            max_checkpoints: Maximum number of checkpoints to keep
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_interval = checkpoint_interval
        self.max_checkpoints = max_checkpoints
        self.checkpoint_count = 0

    def save_checkpoint(
        self,
        dataset: Optional[Dataset],
        pipeline_state: dict[str, Any],
        checkpoint_name: Optional[str] = None,
    ) -> str:
        """Save pipeline checkpoint.

        Args:
            dataset: Current Ray Dataset (optional, can be None for metadata-only)
            pipeline_state: Pipeline state dictionary
            checkpoint_name: Custom checkpoint name (auto-generated if None)

        Returns:
            Path to saved checkpoint
        """
        self.checkpoint_count += 1

        if checkpoint_name is None:
            checkpoint_name = f"checkpoint_{self.checkpoint_count:06d}"

        checkpoint_path = self.checkpoint_dir / checkpoint_name
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        # Save pipeline state
        state_path = checkpoint_path / "pipeline_state.json"
        with open(state_path, "w") as f:
            json.dump(pipeline_state, f, indent=2)

        # Save dataset if provided
        # Use compression and optimized file sizes for large-scale checkpoints
        if dataset is not None:
            dataset_path = checkpoint_path / "dataset"
            try:
                # Use configurable compression for checkpoints
                compression = pipeline_state.get("checkpoint_compression", "snappy")
                num_rows_per_file = pipeline_state.get("checkpoint_num_rows_per_file", 500000)
                
                dataset.write_parquet(
                    str(dataset_path),
                    compression=compression,  # Configurable compression
                    num_rows_per_file=num_rows_per_file,  # Smaller files for faster checkpoint saves
                )
                logger.info(f"Saved dataset checkpoint to {dataset_path} (compression={compression})")
            except (IOError, OSError, RuntimeError) as e:
                logger.error(f"Failed to save dataset checkpoint: {e}")
                raise

        # Save checkpoint metadata
        metadata = {
            "checkpoint_name": checkpoint_name,
            "checkpoint_count": self.checkpoint_count,
            "pipeline_state": pipeline_state,
        }
        metadata_path = checkpoint_path / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved checkpoint {checkpoint_name} to {checkpoint_path}")

        # Clean up old checkpoints
        self._cleanup_old_checkpoints()

        return str(checkpoint_path)

    def load_checkpoint(self, checkpoint_name: Optional[str] = None) -> dict[str, Any]:
        """Load pipeline checkpoint.

        Args:
            checkpoint_name: Checkpoint name (loads latest if None)

        Returns:
            Dictionary with checkpoint data
        """
        if checkpoint_name is None:
            # Find latest checkpoint
            checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_*"))
            if not checkpoints:
                raise FileNotFoundError("No checkpoints found")
            checkpoint_path = checkpoints[-1]
        else:
            checkpoint_path = self.checkpoint_dir / checkpoint_name

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Load metadata
        metadata_path = checkpoint_path / "metadata.json"
        with open(metadata_path) as f:
            metadata = json.load(f)

        # Load pipeline state
        state_path = checkpoint_path / "pipeline_state.json"
        with open(state_path) as f:
            pipeline_state = json.load(f)

        # Load dataset if available
        dataset_path = checkpoint_path / "dataset"
        dataset = None
        if dataset_path.exists():
            try:
                dataset = ray.data.read_parquet(str(dataset_path))
                logger.info(f"Loaded dataset from checkpoint {checkpoint_name}")
            except (IOError, OSError, RuntimeError, FileNotFoundError) as e:
                logger.error(f"Failed to load dataset from checkpoint: {e}")
                raise

        return {
            "checkpoint_path": str(checkpoint_path),
            "metadata": metadata,
            "pipeline_state": pipeline_state,
            "dataset": dataset,
        }

    def list_checkpoints(self) -> list[dict[str, Any]]:
        """List available checkpoints.

        Returns:
            List of checkpoint metadata dictionaries
        """
        checkpoints = []
        for checkpoint_path in sorted(self.checkpoint_dir.glob("checkpoint_*")):
            metadata_path = checkpoint_path / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path) as f:
                    metadata = json.load(f)
                    checkpoints.append(
                        {
                            "name": checkpoint_path.name,
                            "path": str(checkpoint_path),
                            "metadata": metadata,
                        }
                    )
        return checkpoints

    def find_latest_checkpoint(self) -> Optional[str]:
        """Find the latest checkpoint.

        Returns:
            Checkpoint name or None if no checkpoints found
        """
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_*"))
        if not checkpoints:
            return None
        return checkpoints[-1].name

    def get_resume_info(self, checkpoint_name: Optional[str] = None) -> dict[str, Any]:
        """Get information about where to resume.

        Args:
            checkpoint_name: Checkpoint name (None = latest)

        Returns:
            Dictionary with resume information
        """
        checkpoint_data = self.load_checkpoint(checkpoint_name)
        pipeline_state = checkpoint_data.get("pipeline_state", {})
        
        return {
            "checkpoint_name": checkpoint_name or self.find_latest_checkpoint(),
            "current_stage": pipeline_state.get("current_stage"),
            "stage_index": pipeline_state.get("stage_index", 0),
            "total_stages": pipeline_state.get("total_stages"),
            "dataset_path": checkpoint_data.get("dataset_path"),
            "has_error": pipeline_state.get("failed", False),
            "error": pipeline_state.get("error"),
        }

    def resume_from_checkpoint(
        self,
        checkpoint_name: Optional[str] = None,
        stages: Optional[list] = None,
    ) -> tuple[Dataset, int]:
        """Resume pipeline execution from checkpoint.

        Args:
            checkpoint_name: Checkpoint name (None = latest)
            stages: List of stages to execute (None = use from checkpoint)

        Returns:
            Tuple of (dataset, next_stage_index)
        """
        checkpoint_data = self.load_checkpoint(checkpoint_name)
        
        # Load dataset
        dataset = checkpoint_data.get("dataset")
        if dataset is None:
            dataset_path = checkpoint_data.get("checkpoint_path")
            if dataset_path:
                dataset = ray.data.read_parquet(str(Path(dataset_path) / "dataset"))
                logger.info(f"Loaded dataset from checkpoint: {dataset_path}")
            else:
                raise ValueError("Checkpoint does not contain dataset")

        # Get stage information
        pipeline_state = checkpoint_data.get("pipeline_state", {})
        next_stage_index = pipeline_state.get("stage_index", 0)
        
        # If error occurred, resume from previous stage
        if pipeline_state.get("failed", False):
            next_stage_index = max(0, next_stage_index - 1)
            logger.warning(
                f"Resuming from previous stage due to error: {next_stage_index}"
            )

        return dataset, next_stage_index

    def _cleanup_old_checkpoints(self) -> None:
        """Clean up old checkpoints, keeping only the most recent N."""
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_*"))
        if len(checkpoints) > self.max_checkpoints:
            # Remove oldest checkpoints
            for old_checkpoint in checkpoints[: -self.max_checkpoints]:
                logger.info(f"Removing old checkpoint: {old_checkpoint}")
                import shutil

                shutil.rmtree(old_checkpoint)


# Alias for backward compatibility
CheckpointManager = PipelineCheckpoint

def create_checkpoint_manager(
    checkpoint_dir: str,
    checkpoint_interval: int = 1000,
    max_checkpoints: int = 10,
) -> PipelineCheckpoint:
    """Create a checkpoint manager instance.

    Args:
        checkpoint_dir: Directory for checkpoints
        checkpoint_interval: Checkpoint every N batches
        max_checkpoints: Maximum checkpoints to keep

    Returns:
        PipelineCheckpoint instance
    """
    return PipelineCheckpoint(
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=checkpoint_interval,
        max_checkpoints=max_checkpoints,
    )

