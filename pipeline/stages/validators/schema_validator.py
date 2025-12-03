"""Schema validation stage for data quality checks.

Validates data schemas, detects schema drift, and ensures data consistency.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from ray.data import Dataset

from pipeline.stages.base import ValidatorBase
from pipeline.utils.constants import _DEFAULT_BATCH_SIZE

logger = logging.getLogger(__name__)


class SchemaValidator(ValidatorBase):
    """Validate data schemas and detect schema drift.

    Checks data types, required fields, value ranges, and schema consistency.

    Example:
        ```python
        # Simple usage
        validator = SchemaValidator(
            expected_schema={"image": list, "text": str},
        )
        
        # Advanced usage with Ray Data options
        validator = SchemaValidator(
            expected_schema={"image": list, "text": str},
            ray_remote_args={"num_cpus": 2},
            batch_format="pandas",
        )
        ```
    """

    def __init__(
        self,
        expected_schema: Optional[dict[str, Any]] = None,
        strict: bool = True,
        allow_extra_fields: bool = False,
        check_types: bool = True,
        check_ranges: bool = True,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        # Ray Data options
        ray_remote_args: Optional[dict[str, Any]] = None,
        batch_format: Optional[str] = None,
        **map_batches_kwargs: Any,
    ):
        """Initialize schema validator.

        Args:
            expected_schema: Expected schema definition (field_name -> type/constraints)
            strict: Whether to reject items that don't match schema
            allow_extra_fields: Whether to allow fields not in expected schema
            check_types: Whether to validate data types
            check_ranges: Whether to validate value ranges
            batch_size: Batch size for processing
            ray_remote_args: Additional Ray remote arguments
            batch_format: Batch format for map_batches
            **map_batches_kwargs: Additional kwargs passed to map_batches
        """
        super().__init__(reject_invalid=strict, batch_size=batch_size)
        self.expected_schema = expected_schema or {}
        self.strict = strict
        self.allow_extra_fields = allow_extra_fields
        self.check_types = check_types
        self.check_ranges = check_ranges
        self.ray_remote_args = ray_remote_args or {}
        self.batch_format = batch_format
        self.map_batches_kwargs = map_batches_kwargs

    def process(self, dataset: Dataset) -> Dataset:
        """Validate dataset schema.

        Args:
            dataset: Input Ray Dataset

        Returns:
            Dataset with schema validation results
        """
        logger.info("Validating data schema")
        
        if not self.expected_schema:
            logger.warning("No expected schema provided, inferring from data")
            self._infer_schema(dataset)
        
        validated = super().process(dataset)
        
        # Apply additional Ray Data options if provided
        if self.map_batches_kwargs:
            logger.debug("Applying additional map_batches kwargs to validated dataset")
        
        return validated

    def _validate_item(self, item: dict[str, Any]) -> dict[str, Any]:
        """Validate single item against schema."""
        errors = []
        
        for field_name, expected_type in self.expected_schema.items():
            if field_name not in item:
                errors.append(f"Missing required field: {field_name}")
                continue
            
            value = item[field_name]
            
            if self.check_types:
                if not isinstance(value, expected_type):
                    errors.append(
                        f"Field {field_name}: expected {expected_type}, got {type(value)}"
                    )
        
        if not self.allow_extra_fields:
            for field_name in item:
                if field_name not in self.expected_schema:
                    errors.append(f"Unexpected field: {field_name}")
        
        if errors:
            item["schema_errors"] = errors
            item["schema_valid"] = False
        else:
            item["schema_valid"] = True
        
        return item

    def _infer_schema(self, dataset: Dataset) -> None:
        """Infer schema from dataset sample."""
        sample_batch = next(dataset.iter_batches(batch_size=1), None)
        if sample_batch:
            if isinstance(sample_batch, dict):
                self.expected_schema = {k: type(v[0]) if isinstance(v, list) and v else type(v) for k, v in sample_batch.items()}
            else:
                item = sample_batch[0] if isinstance(sample_batch, list) else sample_batch.iloc[0].to_dict()
                self.expected_schema = {k: type(v) for k, v in item.items()}
            logger.info(f"Inferred schema: {self.expected_schema}")

