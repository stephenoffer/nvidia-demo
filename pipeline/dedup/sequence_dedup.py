"""Sequence-level deduplication for robotics trajectories.

Deduplicates entire sequences (not just individual samples) using
sequence embeddings and similarity metrics. Critical for GR00T:
Duplicate sequences waste training compute.
"""

from __future__ import annotations

import logging
from typing import Any, List, Dict

from ray.data import Dataset

from pipeline.dedup.semantic import SemanticDeduplicator
from pipeline.utils.constants import _DEFAULT_BATCH_SIZE

logger = logging.getLogger(__name__)


class SequenceDeduplicator:
    """Deduplicate sequences using sequence embeddings.

    Uses sequence embeddings for similarity comparison and supports
    configurable similarity thresholds.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.95,
        sequence_field: str = "observations",
        use_gpu: bool = True,
        num_workers: int = 1,
    ):
        """Initialize sequence deduplicator.

        Args:
            similarity_threshold: Similarity threshold for duplicates (0.0-1.0)
            sequence_field: Field containing sequence data
            use_gpu: Whether to use GPU acceleration
            num_workers: Number of GPU workers
        """
        self.similarity_threshold = similarity_threshold
        self.sequence_field = sequence_field
        self.use_gpu = use_gpu
        self.num_workers = num_workers

    def deduplicate(self, dataset: Dataset) -> Dataset:
        """Deduplicate sequences in dataset.

        Args:
            dataset: Input Ray Dataset with sequences

        Returns:
            Deduplicated Ray Dataset
        """
        logger.info(
            f"Deduplicating sequences (threshold={self.similarity_threshold}, "
            f"field={self.sequence_field})"
        )

        # Extract sequences and create embeddings
        def extract_sequences_batch(batch: List[Dict[str, Any]]) -> List[str]:
            """Extract sequences from batch as strings for deduplication."""
            sequences = []
            for item in batch:
                seq = item.get(self.sequence_field)
                if seq is not None:
                    # Convert sequence to string representation for deduplication
                    if isinstance(seq, (list, tuple)):
                        seq_str = str(seq)
                    else:
                        seq_str = str(seq)
                    sequences.append(seq_str)
                else:
                    sequences.append("")
            return sequences

        # Extract sequences
        sequences_dataset = dataset.map_batches(
            extract_sequences_batch,
            batch_size=_DEFAULT_BATCH_SIZE,
            batch_format="pandas",
        )

        # Use semantic deduplicator for sequence-level deduplication
        semantic_dedup = SemanticDeduplicator(
            similarity_threshold=self.similarity_threshold
        )

        # Collect sequences for deduplication
        sequences_list = []
        for batch in sequences_dataset.iter_batches(
            batch_size=_DEFAULT_BATCH_SIZE, prefetch_batches=0
        ):
            sequences_list.extend(batch)

        if not sequences_list:
            logger.warning("No sequences found for deduplication")
            return dataset

        # Deduplicate sequences
        keep_mask = semantic_dedup.deduplicate(sequences_list, num_workers=self.num_workers)

        # Filter dataset based on keep mask
        keep_indices = {i for i, keep in enumerate(keep_mask) if keep}

        # Add index to dataset and filter
        import itertools
        counter = itertools.count()

        def add_index(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            """Add index to batch items."""
            result = []
            for item in batch:
                item_copy = dict(item)
                item_copy["_sequence_dedup_idx"] = next(counter)
                result.append(item_copy)
            return result

        indexed_dataset = dataset.map_batches(
            add_index,
            batch_size=_DEFAULT_BATCH_SIZE,
            batch_format="pandas",
        )

        def filter_keep_sequences(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            """Filter batch to keep only non-duplicated sequences."""
            filtered = []
            for item in batch:
                idx = item.get("_sequence_dedup_idx")
                if idx is not None and idx in keep_indices:
                    # Remove index field
                    item_clean = {k: v for k, v in item.items() if not k.startswith("_sequence")}
                    filtered.append(item_clean)
            return filtered

        filtered = indexed_dataset.map_batches(
            filter_keep_sequences,
            batch_size=_DEFAULT_BATCH_SIZE,
            batch_format="pandas",
        )

        logger.info(
            f"Kept {len(keep_indices)}/{len(sequences_list)} sequences after deduplication"
        )

        return filtered

