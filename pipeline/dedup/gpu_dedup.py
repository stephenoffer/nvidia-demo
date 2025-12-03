"""GPU-accelerated deduplication orchestrator.

Orchestrates fuzzy (LSH) and semantic deduplication methods.
Uses Ray for distributed processing.
"""

import logging
from typing import Any, List, Dict

from ray.data import Dataset

from pipeline.dedup.lsh import LSHDeduplicator
from pipeline.dedup.semantic import SemanticDeduplicator

logger = logging.getLogger(__name__)

_DEFAULT_BATCH_SIZE = 1000
# Prefetch batches for better GPU utilization in large-scale processing
# Higher prefetch helps keep GPU busy while CPU processes next batch
_PREFETCH_BATCHES = 2  # Prefetch 2 batches ahead for GPU operations


class GPUDeduplicator:
    """GPU-accelerated deduplication stage for the pipeline.

    Supports both fuzzy (LSH) and semantic deduplication methods.
    Uses streaming operations to avoid materialization.
    """

    def __init__(
        self,
        method: str = "both",
        similarity_threshold: float = 0.95,
        num_gpus: int = 1,
    ):
        """Initialize GPU deduplicator.

        Args:
            method: Deduplication method ('fuzzy', 'semantic', or 'both')
            similarity_threshold: Similarity threshold for duplicates
            num_gpus: Number of GPUs to use
        """
        self.method = method
        self.similarity_threshold = similarity_threshold
        self.num_gpus = num_gpus

        # Initialize deduplicators
        if method in ["fuzzy", "both"]:
            self.lsh_dedup = LSHDeduplicator(similarity_threshold=similarity_threshold)

        if method in ["semantic", "both"]:
            self.semantic_dedup = SemanticDeduplicator(similarity_threshold=similarity_threshold)

    def process(self, dataset: Dataset) -> Dataset:
        """Apply GPU deduplication to dataset using streaming operations.

        Uses Ray Data map_batches and filter operations to avoid materialization.
        For global deduplication, collects text signatures in batches, applies
        deduplication, then filters dataset using a keep mask.

        Args:
            dataset: Input Ray Dataset

        Returns:
            Deduplicated Ray Dataset
        """
        logger.info(f"Applying GPU deduplication (method: {self.method})")

        def extract_texts_with_index(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            """Extract texts from batch and add unique index.
            
            Keep this simple and streaming-compatible. GPU string operations
            require DataFrame conversion which adds overhead. For text extraction,
            CPU operations are fast enough and preserve streaming.
            """
            # Simple CPU extraction - fast enough and streaming-compatible
            result = []
            for item in batch:
                text = item.get("text") or item.get("content")
                item_copy = dict(item)
                if text:
                    item_copy["_dedup_text"] = text
                result.append(item_copy)
            return result

        indexed_dataset = dataset.map_batches(
            extract_texts_with_index,
            batch_size=_DEFAULT_BATCH_SIZE,
            batch_format="pandas",  # Specify batch format for GPU operations
            concurrency=10,
        )

        # Extract texts without materializing entire dataset
        # Use map_batches to extract texts in streaming fashion
        def extract_texts_batch(batch: List[Dict[str, Any]]) -> List[str]:
            """Extract texts from batch."""
            return [
                item["_dedup_text"] for item in batch if "_dedup_text" in item
            ]
        
        # Global deduplication requires materializing text signatures
        # This breaks streaming execution but is necessary for accurate deduplication
        # Use prefetch_batches=0 to minimize memory pressure during materialization
        # For truly streaming deduplication, use per-batch deduplication instead
        text_batches = []
        max_batches_to_collect = 1000  # Limit to prevent OOM
        batch_count = 0
        for batch in indexed_dataset.iter_batches(
            batch_size=_DEFAULT_BATCH_SIZE, prefetch_batches=0  # prefetch=0 to minimize memory
        ):
            batch_texts = extract_texts_batch(batch)
            if batch_texts:
                text_batches.append(batch_texts)
            batch_count += 1
            if batch_count >= max_batches_to_collect:
                logger.warning(
                    f"Limiting text collection to {max_batches_to_collect} batches for deduplication. "
                    "This materializes data and breaks streaming execution."
                )
                break

        if not text_batches:
            logger.warning("No text data found for deduplication")
            # Use named function instead of lambda to avoid serialization issues
            def remove_dedup_fields(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
                """Remove deduplication internal fields."""
                return [
                    {k: v for k, v in item.items() if not k.startswith("_dedup")}
                    for item in batch
                ]
            return indexed_dataset.map_batches(
                remove_dedup_fields,
                batch_size=_DEFAULT_BATCH_SIZE,
                batch_format="pandas",
                concurrency=10,  # Limit concurrency for cleanup operations
            )

        # Flatten text batches for deduplication
        all_texts = []
        for batch_texts in text_batches:
            all_texts.extend(batch_texts)

        logger.info(f"Deduplicating {len(all_texts)} items")

        from pipeline.utils.gpu.memory import clear_gpu_cache

        try:
            if self.method == "fuzzy":
                keep_mask = self.lsh_dedup.deduplicate(all_texts, num_workers=self.num_gpus)
            elif self.method == "semantic":
                keep_mask = self.semantic_dedup.deduplicate(all_texts, num_workers=self.num_gpus)
            elif self.method == "both":
                fuzzy_mask = self.lsh_dedup.deduplicate(all_texts, num_workers=self.num_gpus)
                clear_gpu_cache()
                remaining_texts = [text for text, keep in zip(all_texts, fuzzy_mask) if keep]
                remaining_indices = [i for i, keep in enumerate(fuzzy_mask) if keep]

                if remaining_texts:
                    semantic_mask = self.semantic_dedup.deduplicate(
                        remaining_texts, num_workers=self.num_gpus
                    )
                    keep_mask = [False] * len(all_texts)
                    for idx, keep in zip(remaining_indices, semantic_mask):
                        keep_mask[idx] = keep
                else:
                    keep_mask = fuzzy_mask
            else:
                raise ValueError(f"Unknown deduplication method: {self.method}")
        finally:
            clear_gpu_cache()

        keep_indices = {i for i, keep in enumerate(keep_mask) if keep}

        logger.info(f"Keeping {len(keep_indices)}/{len(all_texts)} items after deduplication")

        # Use itertools.count for global indexing instead of mutable class
        # More efficient and Ray-serialization-friendly
        import itertools
        counter = itertools.count()

        def add_global_index(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            """Add global index to each item in batch."""
            result = []
            for item in batch:
                if "_dedup_text" in item:
                    item_copy = dict(item)
                    item_copy["_global_dedup_idx"] = next(counter)
                    result.append(item_copy)
                else:
                    result.append(item)
            return result

        indexed_global = indexed_dataset.map_batches(
            add_global_index, 
            batch_size=_DEFAULT_BATCH_SIZE,
            batch_format="pandas",  # Specify batch format
        )

        def filter_keep_items(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            """Filter batch to keep only non-duplicated items."""
            filtered = []
            for item in batch:
                if "_global_dedup_idx" not in item:
                    filtered.append(
                        {k: v for k, v in item.items() if not k.startswith("_dedup")}
                    )
                elif item["_global_dedup_idx"] in keep_indices:
                    filtered.append(
                        {k: v for k, v in item.items() if not k.startswith("_dedup")}
                    )
            return filtered

        filtered = indexed_global.map_batches(
            filter_keep_items, 
            batch_size=_DEFAULT_BATCH_SIZE,
            batch_format="pandas",  # Specify batch format
        )

        return filtered
