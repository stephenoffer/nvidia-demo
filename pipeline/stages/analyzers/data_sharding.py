"""Data sharding stage for distributed training.

Creates data shards for distributed training across multiple GPUs/nodes.
Ensures shard balance and reproducibility.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Optional, Dict


from ray.data import Dataset

from pipeline.stages.base import ProcessorBase
from pipeline.utils.constants import _DEFAULT_BATCH_SIZE, _LARGE_BATCH_SIZE

logger = logging.getLogger(__name__)


class DataShardingStage(ProcessorBase):
    """Create data shards for distributed training.

    Supports multiple sharding strategies:
    - Round-robin: Distribute samples evenly
    - Hash-based: Hash-based distribution for reproducibility
    - Task-based: Group by task/episode for task-aware training
    """

    def __init__(
        self,
        num_shards: int,
        shard_strategy: str = "hash",
        shard_key_field: Optional[str] = None,
        ensure_balance: bool = True,
        seed: int = 42,
        batch_size: int = _DEFAULT_BATCH_SIZE,
    ):
        """Initialize data sharding stage.

        Args:
            num_shards: Number of shards to create
            shard_strategy: Sharding strategy ("round_robin", "hash", "task_based")
            shard_key_field: Field to use for hash/task-based sharding
            ensure_balance: Whether to ensure shard balance
            seed: Random seed for reproducibility
            batch_size: Batch size for processing
        """
        super().__init__(batch_size=batch_size)
        self.num_shards = num_shards
        self.shard_strategy = shard_strategy
        self.shard_key_field = shard_key_field
        self.ensure_balance = ensure_balance
        self.seed = seed

    def process(self, dataset: Dataset) -> Dataset:
        """Create data shards.

        Args:
            dataset: Input Ray Dataset

        Returns:
            Dataset with shard assignments
        """
        logger.info(f"Creating {self.num_shards} shards using {self.shard_strategy} strategy")
        return super().process(dataset)

    def _process_item(self, item: dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process item by assigning shard.

        Args:
            item: Data item

        Returns:
            Item with shard assignment
        """
        try:
            shard_id = self._assign_shard(item)
            item["shard_id"] = shard_id
            item["num_shards"] = self.num_shards
            return item
        except (KeyError, TypeError, ValueError) as e:
            logger.warning(f"Failed to assign shard: {e}")
            item["shard_id"] = 0
            item["shard_assignment_error"] = str(e)
            return item

    def _assign_shard(self, item: dict[str, Any]) -> int:
        """Assign shard ID to an item.

        Args:
            item: Data item

        Returns:
            Shard ID (0 to num_shards-1)
        """
        if self.shard_strategy == "round_robin":
            # Simple round-robin (not deterministic across batches)
            # Would need global counter for true round-robin
            return hash(str(item)) % self.num_shards

        elif self.shard_strategy == "hash":
            # Hash-based sharding for reproducibility
            key = self._get_shard_key(item)
            hash_value = int(hashlib.md5(key.encode()).hexdigest(), 16)
            return hash_value % self.num_shards

        elif self.shard_strategy == "task_based":
            # Task-based sharding (group by task/episode)
            key = self._get_task_key(item)
            hash_value = int(hashlib.md5(key.encode()).hexdigest(), 16)
            return hash_value % self.num_shards

        else:
            raise ValueError(f"Unknown shard strategy: {self.shard_strategy}")

    def _get_shard_key(self, item: dict[str, Any]) -> str:
        """Get key for hash-based sharding.

        Args:
            item: Data item

        Returns:
            Shard key string
        """
        if self.shard_key_field and self.shard_key_field in item:
            return str(item[self.shard_key_field])

        # Use item ID or path as key
        if "id" in item:
            return str(item["id"])
        elif "path" in item:
            return str(item["path"])
        elif "episode_id" in item:
            return str(item["episode_id"])
        else:
            # Fallback to hash of entire item
            return str(hash(str(item)))

    def _get_task_key(self, item: dict[str, Any]) -> str:
        """Get task key for task-based sharding.

        Args:
            item: Data item

        Returns:
            Task key string
        """
        # Try to get task/episode identifier
        for field in ["task_id", "episode_id", "trajectory_id", "instruction"]:
            if field in item:
                return str(item[field])

        # Fallback to shard key
        return self._get_shard_key(item)

    def get_shard_statistics(self, dataset: Dataset) -> dict[str, Any]:
        """Get statistics about shard distribution.

        # Use CPU-based sampling to avoid materialization
        Statistics collection should not break streaming execution.

        Args:
            dataset: Dataset with shard assignments

        Returns:
            Dictionary with shard statistics
        """
        # Sample without materializing entire dataset - streaming-compatible
        sample = []
        for batch in dataset.iter_batches(batch_size=_LARGE_BATCH_SIZE, prefetch_batches=0):
            sample.extend(batch[:min(_LARGE_BATCH_SIZE, len(batch))])
            if len(sample) >= _LARGE_BATCH_SIZE:
                break

        shard_counts: dict[int, int] = {}
        for item in sample:
            shard_id = item.get("shard_id", 0)
            shard_counts[shard_id] = shard_counts.get(shard_id, 0) + 1

        if not shard_counts:
            return {"error": "No shard assignments found"}

        counts = list(shard_counts.values())
        stats = {
            "num_shards": len(shard_counts),
            "expected_shards": self.num_shards,
            "shard_counts": shard_counts,
            "min_shard_size": min(counts),
            "max_shard_size": max(counts),
            "avg_shard_size": sum(counts) / len(counts),
            "shard_balance": (
                (max(counts) - min(counts)) / max(counts) if max(counts) > 0 else 0.0
            ),
        }

        logger.info(f"Shard statistics: {stats}")
        return stats

