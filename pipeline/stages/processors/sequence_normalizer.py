"""Sequence normalization and padding stage.

Normalizes variable-length sequences for efficient batching.
Supports multiple padding strategies and sequence length management.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, List, Dict


from ray.data import Dataset

from pipeline.stages.base import ProcessorBase
from pipeline.utils.constants import _DEFAULT_BATCH_SIZE

logger = logging.getLogger(__name__)


class SequenceNormalizer(ProcessorBase):
    """Normalize variable-length sequences for batching.

    Supports padding, truncation, and dynamic batching strategies.
    """

    def __init__(
        self,
        target_length: Optional[int] = None,
        max_length: Optional[int] = None,
        min_length: int = 1,
        padding_strategy: str = "zero",
        truncation_strategy: str = "end",
        sequence_fields: Optional[List[str]] = None,
        batch_size: int = _DEFAULT_BATCH_SIZE,
    ):
        """Initialize sequence normalizer.

        Args:
            target_length: Target sequence length (None = use max in batch)
            max_length: Maximum sequence length (truncate if longer)
            min_length: Minimum sequence length (pad if shorter)
            padding_strategy: Padding strategy ("zero", "repeat", "learnable")
            truncation_strategy: Truncation strategy ("start", "end", "middle")
            sequence_fields: Fields containing sequences (None = auto-detect)
            batch_size: Batch size for processing
        """
        super().__init__(batch_size=batch_size)
        self.target_length = target_length
        self.max_length = max_length
        self.min_length = min_length
        self.padding_strategy = padding_strategy
        self.truncation_strategy = truncation_strategy
        self.sequence_fields = sequence_fields or [
            "frames",
            "sensor_data",
            "observations",
            "actions",
        ]

    def process(self, dataset: Dataset) -> Dataset:
        """Normalize sequences in dataset.

        Args:
            dataset: Input Ray Dataset with variable-length sequences

        Returns:
            Dataset with normalized sequences
        """
        logger.info(
            f"Normalizing sequences (target_length={self.target_length}, "
            f"max_length={self.max_length})"
        )

        def normalize_batch(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
            """Normalize sequences in a batch."""
            normalized = []

            target_len = self.target_length
            if target_len is None:
                max_len = 0
                for item in batch:
                    for field in self.sequence_fields:
                        seq = self._extract_sequence(item, field)
                        if seq is not None:
                            max_len = max(max_len, len(seq))
                target_len = max_len

            if self.max_length:
                target_len = min(target_len, self.max_length)

            for item in batch:
                try:
                    normalized_item = self._normalize_item(item, target_len)
                    if normalized_item:
                        normalized.append(normalized_item)
                except (KeyError, TypeError, ValueError, AttributeError) as e:
                    logger.warning(f"Failed to normalize item: {e}")
                    item["sequence_normalization_error"] = str(e)
                    normalized.append(item)

            return normalized

        return dataset.map_batches(
            normalize_batch,
            batch_size=self.batch_size,
            batch_format="pandas",  # Specify batch format for consistency
        )

    def _process_item(self, item: dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process item by normalizing sequences.

        Args:
            item: Data item

        Returns:
            Normalized item
        """
        target_len = self.target_length or self._compute_target_length(item)
        if self.max_length:
            target_len = min(target_len, self.max_length)
        return self._normalize_item(item, target_len)

    def _compute_target_length(self, item: dict[str, Any]) -> int:
        """Compute target length from item."""
        max_len = 0
        for field in self.sequence_fields:
            seq = self._extract_sequence(item, field)
            if seq is not None:
                max_len = max(max_len, len(seq))
        return max_len if max_len > 0 else self.min_length

    def _normalize_item(
        self, item: dict[str, Any], target_length: int
    ) -> Optional[Dict[str, Any]]:
        """Normalize sequences in a single item.

        Args:
            item: Data item
            target_length: Target sequence length

        Returns:
            Normalized item or None if invalid
        """
        normalized = dict(item)

        for field in self.sequence_fields:
            sequence = self._extract_sequence(item, field)
            if sequence is not None:
                normalized_seq = self._normalize_sequence(sequence, target_length)
                normalized[field] = normalized_seq
                normalized[f"{field}_original_length"] = len(sequence)
                normalized[f"{field}_normalized_length"] = len(normalized_seq)
                normalized[f"{field}_was_padded"] = len(normalized_seq) > len(sequence)
                normalized[f"{field}_was_truncated"] = len(normalized_seq) < len(sequence)

        return normalized

    def _extract_sequence(self, item: dict[str, Any], field: str) -> Any:
        """Extract sequence from item field.

        Args:
            item: Data item
            field: Field name

        Returns:
            Sequence (list, array) or None if not found
        """
        # Direct field access
        if field in item:
            value = item[field]
            if isinstance(value, (list, tuple)):
                return list(value)
            return value

        # Nested access
        if "sensor_data" in item and isinstance(item["sensor_data"], dict):
            if field in item["sensor_data"]:
                value = item["sensor_data"][field]
                if isinstance(value, (list, tuple)):
                    return list(value)
                return value

        return None

    def _normalize_sequence(self, sequence: Any, target_length: int) -> Any:
        """Normalize a single sequence.

        Args:
            sequence: Input sequence
            target_length: Target length

        Returns:
            Normalized sequence
        """
        seq_len = len(sequence)

        # Truncate if too long
        if seq_len > target_length:
            if self.truncation_strategy == "start":
                sequence = sequence[-target_length:]
            elif self.truncation_strategy == "end":
                sequence = sequence[:target_length]
            elif self.truncation_strategy == "middle":
                start = (seq_len - target_length) // 2
                sequence = sequence[start : start + target_length]
            seq_len = len(sequence)

        # Pad if too short - use GPU acceleration when available
        if seq_len < target_length:
            pad_length = target_length - seq_len
            if self.padding_strategy == "zero":
                # Zero padding with GPU acceleration
                if isinstance(sequence, list):
                    sequence = sequence + [0.0] * pad_length
                else:
                    try:
                        import cupy as cp

                        if isinstance(sequence, cp.ndarray):
                            padding = cp.zeros(pad_length, dtype=sequence.dtype)
                            sequence = cp.concatenate([sequence, padding])
                        else:
                            import numpy as np

                            padding = np.zeros(pad_length, dtype=sequence.dtype)
                            sequence = np.concatenate([sequence, padding])
                    except ImportError:
                        import numpy as np

                        try:
                            padding = np.zeros(pad_length, dtype=sequence.dtype)
                            sequence = np.concatenate([sequence, padding])
                        except (AttributeError, TypeError):
                            sequence = list(sequence) + [0.0] * pad_length
            elif self.padding_strategy == "repeat":
                # Repeat last element with GPU acceleration
                if isinstance(sequence, list):
                    last_elem = sequence[-1] if sequence else 0.0
                    sequence = sequence + [last_elem] * pad_length
                else:
                    try:
                        import cupy as cp

                        if isinstance(sequence, cp.ndarray):
                            last_elem = sequence[-1]
                            padding = cp.full(pad_length, last_elem, dtype=sequence.dtype)
                            sequence = cp.concatenate([sequence, padding])
                        else:
                            import numpy as np

                            last_elem = sequence[-1]
                            padding = np.full(pad_length, last_elem, dtype=sequence.dtype)
                            sequence = np.concatenate([sequence, padding])
                    except ImportError:
                        import numpy as np

                        try:
                            last_elem = sequence[-1]
                            padding = np.full(pad_length, last_elem, dtype=sequence.dtype)
                            sequence = np.concatenate([sequence, padding])
                        except (AttributeError, TypeError):
                            sequence = list(sequence) + [sequence[-1]] * pad_length
            elif self.padding_strategy == "learnable":
                # Placeholder for learnable padding - use GPU acceleration
                if isinstance(sequence, list):
                    sequence = sequence + [0.0] * pad_length
                else:
                    try:
                        import cupy as cp

                        if isinstance(sequence, cp.ndarray):
                            padding = cp.zeros(pad_length, dtype=sequence.dtype)
                            sequence = cp.concatenate([sequence, padding])
                        else:
                            import numpy as np

                            padding = np.zeros(pad_length, dtype=sequence.dtype)
                            sequence = np.concatenate([sequence, padding])
                    except ImportError:
                        import numpy as np

                        try:
                            padding = np.zeros(pad_length, dtype=sequence.dtype)
                            sequence = np.concatenate([sequence, padding])
                        except (AttributeError, TypeError):
                            sequence = list(sequence) + [0.0] * pad_length

        return sequence

