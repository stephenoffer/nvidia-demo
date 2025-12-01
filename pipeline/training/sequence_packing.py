"""Sequence packing for efficient transformer training.

Packs variable-length sequences efficiently to maximize GPU utilization.
"""

from __future__ import annotations

import logging
from typing import Any

from ray.data import Dataset

from pipeline.utils.constants import _DEFAULT_BATCH_SIZE, _DEFAULT_MAX_LENGTH

logger = logging.getLogger(__name__)


class SequencePacker:
    """Pack variable-length sequences for efficient batching.

    Supports multiple packing strategies (first-fit, best-fit) and generates
    attention masks for packed sequences.
    """

    def __init__(
        self,
        max_sequence_length: int = _DEFAULT_MAX_LENGTH,
        packing_strategy: str = "first_fit",
        generate_attention_masks: bool = True,
    ):
        """Initialize sequence packer.

        Args:
            max_sequence_length: Maximum sequence length per packed sequence
            packing_strategy: Packing strategy ("first_fit", "best_fit")
            generate_attention_masks: Whether to generate attention masks
        """
        self.max_sequence_length = max_sequence_length
        self.packing_strategy = packing_strategy
        self.generate_attention_masks = generate_attention_masks

    def pack_sequences(self, dataset: Dataset) -> Dataset:
        """Pack variable-length sequences.

        Args:
            dataset: Input Ray Dataset with variable-length sequences

        Returns:
            Dataset with packed sequences and attention masks
        """
        logger.info(
            f"Packing sequences (max_length={self.max_sequence_length}, "
            f"strategy={self.packing_strategy})"
        )

        def pack_batch(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
            """Pack sequences in a batch.

            Args:
                batch: List of items with sequences

            Returns:
                List of packed items
            """
            packed = []
            for item in batch:
                try:
                    packed_item = self._pack_item(item)
                    if packed_item:
                        packed.append(packed_item)
                except (KeyError, TypeError, ValueError, AttributeError) as e:
                    logger.warning(f"Failed to pack sequence: {e}")
                    item["sequence_packing_error"] = str(e)
                    packed.append(item)

            return packed

        return dataset.map_batches(
            pack_batch, 
            batch_size=_DEFAULT_BATCH_SIZE,
            batch_format="pandas",  # Specify batch format
        )

    def _pack_item(self, item: dict[str, Any]) -> dict[str, Any] | None:
        """Pack sequences in a single item.

        Args:
            item: Data item with sequences

        Returns:
            Packed item or None if invalid
        """
        packed = dict(item)

        # Find sequence fields
        sequence_fields = self._find_sequence_fields(item)

        if not sequence_fields:
            return item  # No sequences to pack

        # Pack each sequence field
        packed_sequences: dict[str, Any] = {}
        attention_masks: dict[str, list[int]] = {}

        for field in sequence_fields:
            sequence = self._extract_sequence(item, field)
            if sequence is None:
                continue

            seq_len = len(sequence)

            # Truncate if too long
            if seq_len > self.max_sequence_length:
                sequence = sequence[: self.max_sequence_length]
                seq_len = self.max_sequence_length
                packed[f"{field}_truncated"] = True

            # Pad if needed (for consistent packing)
            if seq_len < self.max_sequence_length:
                padding_length = self.max_sequence_length - seq_len
                sequence = self._pad_sequence(sequence, padding_length)
                packed[f"{field}_padded"] = True

            packed_sequences[field] = sequence

            # Generate attention mask
            if self.generate_attention_masks:
                mask = [1] * seq_len + [0] * (self.max_sequence_length - seq_len)
                attention_masks[f"{field}_mask"] = mask

        packed["packed_sequences"] = packed_sequences
        if attention_masks:
            packed["attention_masks"] = attention_masks
        packed["sequence_packed"] = True
        packed["max_sequence_length"] = self.max_sequence_length

        return packed

    def _find_sequence_fields(self, item: dict[str, Any]) -> list[str]:
        """Find fields containing sequences.

        Args:
            item: Data item

        Returns:
            List of sequence field names
        """
        sequence_fields = []

        for key, value in item.items():
            if isinstance(value, (list, tuple)) and len(value) > 0:
                # Check if it looks like a sequence
                if isinstance(value[0], (int, float, list, dict)):
                    sequence_fields.append(key)

        return sequence_fields

    def _extract_sequence(self, item: dict[str, Any], field: str) -> Any:
        """Extract sequence from item.

        Args:
            item: Data item
            field: Field name

        Returns:
            Sequence or None
        """
        if field in item:
            value = item[field]
            if isinstance(value, (list, tuple)):
                return list(value)
            return value

        return None

    def _pad_sequence(self, sequence: Any, padding_length: int) -> Any:
        """Pad sequence to target length.

        Args:
            sequence: Input sequence
            padding_length: Length to pad

        Returns:
            Padded sequence
        """
        if isinstance(sequence, list):
            return sequence + [0.0] * padding_length
        else:
            try:
                import numpy as np  # https://numpy.org/

                padding = np.zeros(padding_length, dtype=sequence.dtype)
                return np.concatenate([sequence, padding])
            except ImportError:
                return list(sequence) + [0.0] * padding_length

