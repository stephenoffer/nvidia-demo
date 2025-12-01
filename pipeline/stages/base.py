"""Base classes for pipeline stages.

Provides common functionality for all pipeline stages including batch processing,
error handling, and logging.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Optional

from ray.data import Dataset

logger = logging.getLogger(__name__)

# Import from constants for consistency
from pipeline.utils.constants import _DEFAULT_BATCH_SIZE


class PipelineStage(ABC):
    """Base class for all pipeline processing stages.

    Provides common functionality for batch processing, error handling,
    and result tracking.
    """

    def __init__(self, batch_size: int = _DEFAULT_BATCH_SIZE):
        """Initialize pipeline stage.

        Args:
            batch_size: Batch size for map_batches operations
        """
        self.batch_size = batch_size

    @abstractmethod
    def process(self, dataset: Dataset) -> Dataset:
        """Process dataset through this stage.

        Args:
            dataset: Input Ray Dataset

        Returns:
            Processed Ray Dataset
        """
        pass

    def _process_batch_with_error_handling(
        self,
        dataset: Dataset,
        process_func: Any,
        error_field: str = "processing_error",
        keep_on_error: bool = True,
    ) -> Dataset:
        """Process dataset with standard error handling.

        Args:
            dataset: Input dataset
            process_func: Function to process each batch
            error_field: Field name for error messages
            keep_on_error: Whether to keep items that error

        Returns:
            Processed dataset
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
                    logger.warning(f"Failed to process item: {e}", exc_info=True)
                    if keep_on_error:
                        item[error_field] = str(e)
                        processed.append(item)
            return processed

        return dataset.map_batches(
            process_batch, 
            batch_size=self.batch_size,
            batch_format="pandas"  # Specify batch format for better performance
        )


class ValidatorBase(PipelineStage):
    """Base class for validation stages.

    Provides common validation patterns and result formatting.
    """

    def __init__(
        self,
        reject_invalid: bool = False,
        batch_size: int = _DEFAULT_BATCH_SIZE,
    ):
        """Initialize validator.

        Args:
            reject_invalid: Whether to reject invalid items
            batch_size: Batch size for processing
        """
        super().__init__(batch_size=batch_size)
        self.reject_invalid = reject_invalid

    @abstractmethod
    def _validate_item(self, item: dict[str, Any]) -> dict[str, Any]:
        """Validate a single item.

        Args:
            item: Data item to validate

        Returns:
            Validation result dictionary with 'is_valid' key
        """
        pass

    def process(self, dataset: Dataset) -> Dataset:
        """Validate dataset.

        Args:
            dataset: Input dataset

        Returns:
            Validated dataset
        """
        def validate_batch(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
            """Validate batch."""
            validated = []
            for item in batch:
                try:
                    result = self._validate_item(item)
                    if result.get("is_valid", True) or not self.reject_invalid:
                        item.update(result)
                        validated.append(item)
                except (KeyError, TypeError, ValueError, AttributeError) as e:
                    logger.warning(f"Validation failed: {e}", exc_info=True)
                    if not self.reject_invalid:
                        item["validation_error"] = str(e)
                        item["is_valid"] = False
                        validated.append(item)
            return validated

        return dataset.map_batches(
            validate_batch, 
            batch_size=self.batch_size,
            batch_format="pandas"  # Specify batch format for better performance
        )


class ProcessorBase(PipelineStage):
    """Base class for processing stages.

    Provides common processing patterns.
    """

    @abstractmethod
    def _process_item(self, item: dict[str, Any]) -> Optional[dict[str, Any]]:
        """Process a single item.

        Args:
            item: Data item to process

        Returns:
            Processed item or None to filter
        """
        pass

    def process(self, dataset: Dataset) -> Dataset:
        """Process dataset.

        Args:
            dataset: Input dataset

        Returns:
            Processed dataset
        """
        def process_batch(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
            """Process batch."""
            processed = []
            for item in batch:
                try:
                    result = self._process_item(item)
                    if result is not None:
                        processed.append(result)
                except (KeyError, TypeError, ValueError, AttributeError) as e:
                    logger.warning(f"Processing failed: {e}", exc_info=True)
                    item["processing_error"] = str(e)
                    processed.append(item)
            return processed

        return dataset.map_batches(
            process_batch, 
            batch_size=self.batch_size,
            batch_format="pandas"  # Specify batch format for better performance
        )

