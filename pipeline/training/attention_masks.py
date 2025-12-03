"""Attention mask generation for transformer models.

Generates attention masks for causal, bidirectional, and cross-modal attention.
"""

from __future__ import annotations

import logging
from typing import Any, Optional


from ray.data import Dataset

from pipeline.utils.constants import _DEFAULT_BATCH_SIZE

logger = logging.getLogger(__name__)


class AttentionMaskGenerator:
    """Generate attention masks for transformer training.

    Supports causal masking (for autoregressive models), bidirectional masking
    (for encoder models), and cross-modal masking.
    """

    def __init__(
        self,
        mask_type: str = "causal",  # "causal", "bidirectional", "cross_modal"
        sequence_length_field: str = "sequence_length",
    ):
        """Initialize attention mask generator.

        Args:
            mask_type: Type of attention mask ("causal", "bidirectional", "cross_modal")
            sequence_length_field: Field containing sequence length
        """
        self.mask_type = mask_type
        self.sequence_length_field = sequence_length_field

    def generate_masks(self, dataset: Dataset) -> Dataset:
        """Generate attention masks for dataset.

        Args:
            dataset: Input Ray Dataset

        Returns:
            Dataset with attention masks added
        """
        logger.info(f"Generating {self.mask_type} attention masks")

        def generate_batch(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
            """Generate masks for a batch.

            Args:
                batch: List of data items

            Returns:
                List of items with attention masks
            """
            masked = []
            for item in batch:
                try:
                    masked_item = self._generate_masks_for_item(item)
                    masked.append(masked_item)
                except (KeyError, TypeError, ValueError, AttributeError) as e:
                    logger.warning(f"Failed to generate attention masks: {e}")
                    item["attention_mask_error"] = str(e)
                    masked.append(item)

            return masked

        return dataset.map_batches(
            generate_batch,
            batch_size=_DEFAULT_BATCH_SIZE,
            batch_format="pandas",  # Specify batch format for consistency
        )

    def _generate_masks_for_item(self, item: dict[str, Any]) -> dict[str, Any]:
        """Generate attention masks for a single item.

        Args:
            item: Data item

        Returns:
            Item with attention masks added
        """
        masked = dict(item)

        # Get sequence length
        seq_len = item.get(self.sequence_length_field)
        if seq_len is None:
            # Try to infer from sequence fields
            seq_len = self._infer_sequence_length(item)

        if seq_len is None:
            logger.warning("Could not determine sequence length, skipping mask generation")
            return item

        # Generate mask based on type
        if self.mask_type == "causal":
            mask = self._generate_causal_mask(seq_len)
        elif self.mask_type == "bidirectional":
            mask = self._generate_bidirectional_mask(seq_len)
        elif self.mask_type == "cross_modal":
            mask = self._generate_cross_modal_mask(item, seq_len)
        else:
            raise ValueError(f"Unknown mask type: {self.mask_type}")

        masked["attention_mask"] = mask
        masked["mask_type"] = self.mask_type
        masked["sequence_length"] = seq_len

        return masked

    def _infer_sequence_length(self, item: dict[str, Any]) -> Optional[int]:
        """Infer sequence length from item.

        Args:
            item: Data item

        Returns:
            Sequence length or None
        """
        # Try common sequence fields
        for field in ["frames", "observations", "actions", "sensor_data"]:
            if field in item:
                value = item[field]
                if isinstance(value, (list, tuple)):
                    return len(value)
                elif isinstance(value, dict) and "length" in value:
                    return value["length"]

        return None

    def _generate_causal_mask(self, seq_len: int) -> list[list[int]]:
        """Generate causal (lower triangular) attention mask.

        Args:
            seq_len: Sequence length

        Returns:
            Causal attention mask (seq_len x seq_len)
        """
        mask = []
        for i in range(seq_len):
            row = [1 if j <= i else 0 for j in range(seq_len)]
            mask.append(row)
        return mask

    def _generate_bidirectional_mask(self, seq_len: int) -> list[list[int]]:
        """Generate bidirectional (full) attention mask.

        Args:
            seq_len: Sequence length

        Returns:
            Bidirectional attention mask (seq_len x seq_len)
        """
        return [[1] * seq_len for _ in range(seq_len)]

    def _generate_cross_modal_mask(
        self, item: dict[str, Any], seq_len: int
    ) -> list[list[int]]:
        """Generate cross-modal attention mask.

        Args:
            item: Data item
            seq_len: Sequence length

        Returns:
            Cross-modal attention mask
        """
        # Would need modality-specific logic
        # For now, return bidirectional mask
        return self._generate_bidirectional_mask(seq_len)

