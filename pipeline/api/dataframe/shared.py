"""Shared utilities for DataFrame operations."""

from __future__ import annotations

import logging
from typing import Any, Callable

from ray.data import Dataset

logger = logging.getLogger(__name__)


def validate_batch_structure(batch: dict[str, Any]) -> tuple[list[str], int]:
    """Validate batch structure and return keys and item count.
    
    Args:
        batch: Batch dictionary
        
    Returns:
        Tuple of (keys, num_items)
        
    Raises:
        ValueError: If batch structure is invalid
    """
    if not batch:
        return [], 0
    
    keys = list(batch.keys())
    if not keys:
        return [], 0
    
    num_items = len(batch[keys[0]])
    if num_items == 0:
        return [], 0
    
    # Validate all columns have same length
    for key in keys:
        if len(batch[key]) != num_items:
            raise ValueError(f"Column {key} has inconsistent length: {len(batch[key])} != {num_items}")
    
    return keys, num_items


def batch_to_items(batch: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert batch dictionary to list of item dictionaries.
    
    Args:
        batch: Batch dictionary
        
    Returns:
        List of item dictionaries
    """
    keys, num_items = validate_batch_structure(batch)
    if num_items == 0:
        return []
    
    return [
        {key: batch[key][i] for key in keys}
        for i in range(num_items)
    ]


def items_to_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Convert list of item dictionaries to batch dictionary.
    
    Args:
        items: List of item dictionaries
        
    Returns:
        Batch dictionary
        
    Raises:
        ValueError: If items have inconsistent keys
    """
    if not items:
        return {}
    
    # Validate all items have same keys
    result_keys = list(items[0].keys())
    for item in items[1:]:
        if set(item.keys()) != set(result_keys):
            raise ValueError("All items must have the same keys")
    
    # Convert to batch format
    return {
        key: [item[key] for item in items]
        for key in result_keys
    }


def validate_columns(dataset: Dataset, columns: list[str]) -> None:
    """Validate that columns exist in dataset.
    
    Args:
        dataset: Ray Data Dataset
        columns: Column names to validate
        
    Raises:
        ValueError: If columns don't exist
    """
    try:
        sample = dataset.take(1)
        if sample:
            available_keys = set(sample[0].keys())
            missing = set(columns) - available_keys
            if missing:
                raise ValueError(f"Columns not found: {missing}. Available: {available_keys}")
    except Exception:
        logger.debug("Could not validate columns, proceeding anyway")


def _create_dataframe_factory(name: Optional[str] = None):
    """Internal: Factory function to create PipelineDataFrame instances.
    
    Args:
        name: Optional name for the DataFrame
        
    Returns:
        Function that creates PipelineDataFrame instances
    """
    from pipeline.api.dataframe.core import PipelineDataFrame
    
    def factory(dataset: Dataset) -> PipelineDataFrame:
        """Create PipelineDataFrame instance."""
        return PipelineDataFrame(dataset, name=name)
    
    return factory

