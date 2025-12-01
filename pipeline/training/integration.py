"""Integration with training pipelines for foundation models.

Provides seamless integration between data curation pipeline and
model training workflows, ensuring curated data is ready for training.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional


import ray  # https://docs.ray.io/
from ray.data import Dataset  # https://docs.ray.io/en/latest/data/data.html

from typing import Any as TypingAny  # For type hints

logger = logging.getLogger(__name__)


class TrainingPipelineIntegration:
    """Integration with model training pipelines.

    Connects curated data output to training workflows, ensuring
    data format compatibility and efficient data loading for training.
    """

    def __init__(
        self,
        output_format: str = "parquet",
        batch_size: Optional[int] = None,
        shuffle: bool = True,
    ):
        """Initialize training pipeline integration.

        Args:
            output_format: Output format for training ('parquet', 'tfrecord', 'arrow')
            batch_size: Batch size for training data loaders
            shuffle: Whether to shuffle data for training
        """
        from pipeline.utils.constants import _TRAINING_BATCH_SIZE

        self.output_format = output_format
        self.batch_size = batch_size if batch_size is not None else _TRAINING_BATCH_SIZE
        self.shuffle = shuffle

    def prepare_for_training(
        self,
        dataset: Dataset,
        output_path: str,
        train_split: float = 0.9,
    ) -> Dict[str, str]:
        """Prepare curated dataset for training.

        Args:
            dataset: Curated Ray Dataset
            output_path: Output directory for training data
            train_split: Fraction of data for training (rest for validation)

        Returns:
            Dictionary with paths to train and validation datasets
        """
        logger.info("Preparing dataset for training")

        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Split into train and validation
        # Use deterministic hash-based splitting for reproducibility
        # This avoids materializing the full dataset
        if train_split < 1.0:
            # Use CPU-based split assignment - fast enough and streaming-compatible
            def assign_split_batch(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
                """Assign train/val split to batch.
                
                CRITICAL: Keep CPU-based for streaming compatibility.
                GPU string operations require DataFrame conversion which adds overhead.
                Hash computation is fast enough on CPU and preserves streaming execution.
                """
                import hashlib
                result = []
                for item in batch:
                    split_key = str(item.get("episode_id", item.get("path", hash(str(item)))))
                    hash_val = int(hashlib.md5(split_key.encode()).hexdigest(), 16)
                    split_val = (hash_val % 100) / 100.0
                    item["_split"] = "train" if split_val < train_split else "val"
                    result.append(item)
                return result
            
            dataset_with_split = dataset.map_batches(
                assign_split_batch, 
                batch_size=self.batch_size, 
                batch_format="pandas"
            )
            
            # Use named functions instead of lambdas for better serialization
            def is_train_split(item: dict[str, Any]) -> bool:
                """Check if item is in train split."""
                return item.get("_split") == "train"
            
            def is_val_split(item: dict[str, Any]) -> bool:
                """Check if item is in val split."""
                return item.get("_split") == "val"
            
            train_dataset = dataset_with_split.filter(is_train_split)
            val_dataset = dataset_with_split.filter(is_val_split)
        else:
            train_dataset = dataset
            val_dataset = None

        # Shuffle if requested - use Ray Data's random_shuffle directly
        if self.shuffle:
            shuffle_seed = 42  # Default seed for reproducibility
            train_dataset = train_dataset.random_shuffle(seed=shuffle_seed)

        # Write training data with compression for large-scale datasets
        train_path = output_path / "train"
        train_dataset.write_parquet(
            str(train_path),
            compression="snappy",  # Fast compression for training data
            num_rows_per_file=1000000,  # Optimize file sizes
        )
        logger.info(f"Wrote training data to {train_path}")

        # Write validation data if split
        val_path = None
        if val_dataset is not None:
            val_path = output_path / "val"
            val_dataset.write_parquet(
                str(val_path),
                compression="snappy",  # Fast compression for validation data
                num_rows_per_file=1000000,  # Optimize file sizes
            )
            logger.info(f"Wrote validation data to {val_path}")

        # Avoid materialization - use num_rows property if available, otherwise skip count
        # Users can check output files or use dataset.num_rows() if needed
        result = {
            "train_path": str(train_path),
            "val_path": str(val_path) if val_path else None,
        }
        # Only count if explicitly needed (materializes dataset)
        # For large datasets, prefer checking output file sizes
        try:
            result["num_train_samples"] = train_dataset.num_rows() if hasattr(train_dataset, 'num_rows') else None
            if val_dataset:
                result["num_val_samples"] = val_dataset.num_rows() if hasattr(val_dataset, 'num_rows') else None
            else:
                result["num_val_samples"] = None
        except (AttributeError, RuntimeError):
            # num_rows not available or dataset not materialized - skip count
            result["num_train_samples"] = None
            result["num_val_samples"] = None
        return result

    def create_dataloader(
        self,
        dataset_path: str,
        num_workers: int = 4,
        pin_memory: bool = True,
    ) -> Any:
        """Create PyTorch DataLoader from curated dataset.

        Args:
            dataset_path: Path to curated dataset
            num_workers: Number of data loading workers
            pin_memory: Whether to pin memory for GPU transfer

        Returns:
            PyTorch DataLoader instance
        """
        from torch.utils.data import DataLoader, IterableDataset  # https://pytorch.org/

        # Create dataset from Parquet files
        dataset = ray.data.read_parquet(dataset_path)

        # Pack sequences if needed
        try:
            from pipeline.training.sequence_packing import SequencePacker

            packer = SequencePacker(max_sequence_length=2048)
            dataset = packer.pack_sequences(dataset)
        except (ImportError, AttributeError, RuntimeError) as e:
            logger.warning(f"Sequence packing failed: {e}")

        # Generate attention masks if needed
        try:
            from pipeline.training.attention_masks import AttentionMaskGenerator

            mask_generator = AttentionMaskGenerator(mask_type="causal")
            dataset = mask_generator.generate_masks(dataset)
        except (ImportError, AttributeError, RuntimeError) as e:
            logger.warning(f"Attention mask generation failed: {e}")

        # Convert to PyTorch IterableDataset
        class RayDatasetWrapper(IterableDataset):
            def __init__(self, ray_dataset, batch_size):
                self.ray_dataset = ray_dataset
                self.batch_size = batch_size

            def __iter__(self):
                yield from self.ray_dataset.iter_batches(
                    batch_size=self.batch_size,
                    drop_last=True,
                )

        pytorch_dataset = RayDatasetWrapper(dataset, self.batch_size)

        # Create DataLoader
        dataloader = DataLoader(
            pytorch_dataset,
            batch_size=self.batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        logger.info("Created PyTorch DataLoader from curated dataset")
        return dataloader
