"""Episode boundary detection stage for robotics trajectories.

Detects episode boundaries in trajectory data and ensures episodes are
not split across batches.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Dict


from ray.data import Dataset

from pipeline.stages.base import ProcessorBase
from pipeline.utils.constants import _DEFAULT_BATCH_SIZE, _LARGE_BATCH_SIZE

logger = logging.getLogger(__name__)


class EpisodeBoundaryDetector(ProcessorBase):
    """Detect and preserve episode boundaries in trajectory data.

    Ensures episodes are not split across batches and adds episode-level
    metadata for proper sequence modeling.
    """

    def __init__(
        self,
        episode_id_field: str = "episode_id",
        step_field: str = "step",
        min_episode_length: int = 1,
        max_episode_length: Optional[int] = None,
        validate_boundaries: bool = True,
        batch_size: int = _DEFAULT_BATCH_SIZE,
    ):
        """Initialize episode boundary detector.

        Args:
            episode_id_field: Field name containing episode ID
            step_field: Field name containing step number
            min_episode_length: Minimum episode length (filter shorter episodes)
            max_episode_length: Maximum episode length (truncate longer episodes)
            validate_boundaries: Whether to validate episode boundaries
            batch_size: Batch size for processing
        """
        super().__init__(batch_size=batch_size)
        self.episode_id_field = episode_id_field
        self.step_field = step_field
        self.min_episode_length = min_episode_length
        self.max_episode_length = max_episode_length
        self.validate_boundaries = validate_boundaries

    def process(self, dataset: Dataset) -> Dataset:
        """Detect and preserve episode boundaries.

        Args:
            dataset: Input Ray Dataset with trajectory data

        Returns:
            Dataset with episode boundaries detected and preserved
        """
        logger.info("Detecting episode boundaries")

        processed = super().process(dataset)

        if self.min_episode_length > 1 or self.max_episode_length:
            processed = self._filter_by_length(processed)

        return processed

    def _process_item(self, item: dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single item for episode detection.

        Args:
            item: Trajectory item

        Returns:
            Item with episode metadata or None if filtered
        """
        processed = dict(item)

        # Extract episode ID
        episode_id = item.get(self.episode_id_field)
        if episode_id is None:
            # Try to infer episode ID from other fields
            episode_id = item.get("episode", item.get("trajectory_id", None))

        if episode_id is None:
            # No episode ID found - assign default or skip
            logger.debug("No episode ID found, assigning default")
            episode_id = "unknown"

        processed["episode_id"] = episode_id
        processed["has_episode_id"] = episode_id != "unknown"

        # Extract step number
        step = item.get(self.step_field)
        if step is None:
            step = item.get("timestep", item.get("t", 0))

        processed["step"] = int(step) if step is not None else 0
        processed["has_step"] = step is not None

        # Detect episode start/end
        processed["is_episode_start"] = processed["step"] == 0
        processed["is_episode_end"] = False  # Will be set during batch processing

        # Add episode metadata
        processed["episode_metadata"] = {
            "episode_id": episode_id,
            "step": processed["step"],
            "has_boundary": processed["has_episode_id"] and processed["has_step"],
        }

        return processed

    def _filter_by_length(self, dataset: Dataset) -> Dataset:
        """Filter episodes by length.

        Args:
            dataset: Dataset with episode metadata

        Returns:
            Filtered dataset
        """
        # Group by episode and filter
        def filter_episodes_batch(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
            """Filter episodes in batch by length.
            
            # Keep CPU-based for streaming compatibility
            GPU operations require DataFrame conversion which adds overhead.
            Episode filtering is fast enough on CPU and preserves streaming execution.

            Args:
                batch: List of items with episode metadata

            Returns:
                Filtered items
            """
            # CPU-based filtering - fast enough and streaming-compatible
            episodes: dict[str, list[dict[str, Any]]] = {}
            for item in batch:
                episode_id = item.get("episode_id", "unknown")
                if episode_id not in episodes:
                    episodes[episode_id] = []
                episodes[episode_id].append(item)

            filtered = []
            for episode_id, items in episodes.items():
                episode_length = len(items)

                if episode_length < self.min_episode_length:
                    logger.debug(
                        f"Filtering episode {episode_id}: length {episode_length} < "
                        f"min {self.min_episode_length}"
                    )
                    continue

                if self.max_episode_length and episode_length > self.max_episode_length:
                    logger.debug(
                        f"Truncating episode {episode_id}: length {episode_length} > "
                        f"max {self.max_episode_length}"
                    )
                    items_sorted = sorted(items, key=lambda x: x.get("step", 0))
                    items = items_sorted[: self.max_episode_length]

                if items:
                    items[-1]["is_episode_end"] = True

                filtered.extend(items)

            return filtered

        return dataset.map_batches(
            filter_episodes_batch,
            batch_size=_LARGE_BATCH_SIZE,
            batch_format="pandas",  # Use pandas format for GPU operations
        )

    def get_episode_statistics(self, dataset: Dataset) -> dict[str, Any]:
        """Get statistics about episodes in dataset.

        # Use take() for sampling - it's streaming-compatible
        GPU aggregation requires DataFrame conversion which adds overhead.
        Statistics collection should be lightweight and not break streaming.

        Args:
            dataset: Dataset with episode metadata

        Returns:
            Dictionary with episode statistics
        """
        # Use iter_batches() for streaming-compatible sampling
        # Avoid materializing entire dataset with take()
        episodes: dict[str, list[int]] = {}
        sample_count = 0
        for batch in dataset.iter_batches(batch_size=_LARGE_BATCH_SIZE, prefetch_batches=0):
            for item in batch:
                episode_id = item.get("episode_id", "unknown")
                step = item.get("step", 0)
                if episode_id not in episodes:
                    episodes[episode_id] = []
                episodes[episode_id].append(step)
                sample_count += 1
                # Limit sample size to avoid excessive processing
                if sample_count >= _LARGE_BATCH_SIZE:
                    break
            if sample_count >= _LARGE_BATCH_SIZE:
                break

        episode_lengths = [max(steps) - min(steps) + 1 for steps in episodes.values()]

        stats = {
            "num_episodes_sampled": len(episodes),
            "avg_episode_length": sum(episode_lengths) / len(episode_lengths) if episode_lengths else 0,
            "min_episode_length": min(episode_lengths) if episode_lengths else 0,
            "max_episode_length": max(episode_lengths) if episode_lengths else 0,
        }

        logger.info(f"Episode statistics: {stats}")
        return stats

