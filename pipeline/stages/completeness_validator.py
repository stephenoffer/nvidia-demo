"""Data completeness validation stage.

Validates that required fields are present and complete for each data type.
Ensures data quality before training.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from ray.data import Dataset

from pipeline.stages.base import ValidatorBase
from pipeline.utils.constants import _DEFAULT_BATCH_SIZE

logger = logging.getLogger(__name__)


class CompletenessValidator(ValidatorBase):
    """Validate data completeness for multimodal datasets.

    Checks that required fields are present and complete according to
    data type schemas.
    """

    def __init__(
        self,
        required_fields_by_type: Optional[dict[str, list[str]]] = None,
        allow_partial: bool = False,
        fill_missing: bool = False,
        batch_size: int = _DEFAULT_BATCH_SIZE,
    ):
        """Initialize completeness validator.

        Args:
            required_fields_by_type: Dictionary mapping data_type to required fields
            allow_partial: Whether to allow partial data (flag but don't reject)
            fill_missing: Whether to fill missing fields with defaults (not recommended)
            batch_size: Batch size for processing
        """
        super().__init__(reject_invalid=not allow_partial, batch_size=batch_size)
        self.required_fields_by_type = required_fields_by_type or {
            "video": ["frames", "format"],
            "sensor": ["sensor_data", "format"],
            "text": ["text", "format"],
        }
        self.allow_partial = allow_partial
        self.fill_missing = fill_missing

    def process(self, dataset: Dataset) -> Dataset:
        """Validate data completeness.

        Args:
            dataset: Input Ray Dataset

        Returns:
            Dataset with completeness validation results
        """
        logger.info("Validating data completeness")
        return super().process(dataset)

    def _validate_item(self, item: dict[str, Any]) -> dict[str, Any]:
        """Validate completeness for a single item.

        Args:
            item: Data item

        Returns:
            Validation result dictionary
        """
        from pipeline.utils.data_types import get_data_type

        data_type = get_data_type(item).value
        required_fields = self.required_fields_by_type.get(data_type, [])

        missing_fields: list[str] = []
        for field in required_fields:
            if field not in item or item[field] is None:
                missing_fields.append(field)

        is_complete = len(missing_fields) == 0

        if self.fill_missing and missing_fields:
            item = self._fill_missing_fields(item, missing_fields)

        return {
            "is_valid": is_complete,
            "is_complete": is_complete,
            "missing_fields": missing_fields,
            "completeness_score": 1.0 - (len(missing_fields) / max(len(required_fields), 1)),
            "data_type": data_type,
            "required_fields": required_fields,
        }

    def _fill_missing_fields(
        self, item: dict[str, Any], missing_fields: list[str]
    ) -> dict[str, Any]:
        """Fill missing fields with default values.

        WARNING: This is not recommended for production. Missing data should
        be handled explicitly or data should be rejected.

        Args:
            item: Data item
            missing_fields: List of missing field names

        Returns:
            Item with missing fields filled
        """
        filled = dict(item)

        for field in missing_fields:
            if field == "frames":
                filled[field] = []
            elif field == "sensor_data":
                filled[field] = {}
            elif field == "text":
                filled[field] = ""
            elif field == "format":
                filled[field] = item.get("data_type", "unknown")
            else:
                filled[field] = None

            filled[f"{field}_filled"] = True

        return filled

