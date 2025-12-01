"""Batch processing utilities for pipeline stages.

Provides common batch processing patterns and helpers.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional, Dict


from ray.data import Dataset

logger = logging.getLogger(__name__)

_DEFAULT_BATCH_SIZE = 100


def create_batch_processor(
    process_func: Callable[[dict[str, Any]], Optional[Dict[str, Any]]],
    batch_size: int = _DEFAULT_BATCH_SIZE,
    error_field: str = "processing_error",
    keep_on_error: bool = True,
) -> Callable[[list[dict[str, Any]]], list[dict[str, Any]]]:
    """Create a batch processing function with error handling.

    Args:
        process_func: Function to process individual items
        batch_size: Batch size (for reference, actual batching handled by Ray)
        error_field: Field name for error messages
        keep_on_error: Whether to keep items that error

    Returns:
        Batch processing function
    """
    def process_batch(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Process batch with error handling."""
        processed = []
        for item in batch:
            try:
                result = process_func(item)
                if result is not None:
                    processed.append(result)
            except (KeyError, TypeError, ValueError, AttributeError) as e:
                logger.warning(f"Failed to process item: {e}")
                if keep_on_error:
                    item[error_field] = str(e)
                    processed.append(item)
        return processed

    return process_batch


def add_metadata_batch(
    batch: list[dict[str, Any]],
    format_val: str,
    data_type: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Add format and data type metadata to batch.

    Args:
        batch: List of items
        format_val: Format value to add
        data_type: Data type value (defaults to format_val)

    Returns:
        List of items with metadata
    """
    if data_type is None:
        data_type = format_val

    return [
        {
            **item,
            "format": format_val,
            "data_type": data_type,
        }
        for item in batch
    ]


def process_dataset_with_batch(
    dataset: Dataset,
    process_func: Callable[[dict[str, Any]], Optional[Dict[str, Any]]],
    batch_size: int = _DEFAULT_BATCH_SIZE,
    error_field: str = "processing_error",
    keep_on_error: bool = True,
) -> Dataset:
    """Process dataset with batch processing and error handling.

    Args:
        dataset: Input dataset
        process_func: Function to process individual items
        batch_size: Batch size for map_batches
        error_field: Field name for error messages
        keep_on_error: Whether to keep items that error

    Returns:
        Processed dataset
    """
    batch_processor = create_batch_processor(
        process_func=process_func,
        batch_size=batch_size,
        error_field=error_field,
        keep_on_error=keep_on_error,
    )
    return dataset.map_batches(
        batch_processor,
        batch_size=batch_size,
        batch_format="pandas",  # Specify batch format for consistency
    )

