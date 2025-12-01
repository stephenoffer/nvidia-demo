"""Data distribution analysis stage.

Analyzes data distributions to detect bias and distribution shifts.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from ray.data import Dataset

from pipeline.stages.base import ProcessorBase
from pipeline.utils.constants import _DEFAULT_BATCH_SIZE

logger = logging.getLogger(__name__)


class DistributionAnalyzer(ProcessorBase):
    """Analyze data distributions for bias detection and quality assessment."""

    def __init__(
        self,
        fields_to_analyze: Optional[list[str]] = None,
        compute_statistics: bool = True,
        detect_shifts: bool = False,
        batch_size: int = _DEFAULT_BATCH_SIZE,
    ):
        """Initialize distribution analyzer.

        Args:
            fields_to_analyze: Fields to analyze (None = auto-detect numeric fields)
            compute_statistics: Whether to compute distribution statistics
            detect_shifts: Whether to detect distribution shifts (requires reference)
            batch_size: Batch size for processing
        """
        super().__init__(batch_size=batch_size)
        self.fields_to_analyze = fields_to_analyze
        self.compute_statistics = compute_statistics
        self.detect_shifts = detect_shifts

    def process(self, dataset: Dataset) -> Dataset:
        """Analyze distributions in dataset.

        Args:
            dataset: Input Ray Dataset

        Returns:
            Dataset with distribution statistics added
        """
        logger.info("Analyzing data distributions")
        return super().process(dataset)

    def _process_item(self, item: dict[str, Any]) -> Optional[dict[str, Any]]:
        """Process item by analyzing distributions.

        Args:
            item: Data item

        Returns:
            Item with distribution statistics
        """
        return self._analyze_item(item)

    def _analyze_item(self, item: dict[str, Any]) -> dict[str, Any]:
        """Analyze distributions for a single item.

        Args:
            item: Data item

        Returns:
            Item with distribution statistics
        """
        analyzed = dict(item)

        # Determine fields to analyze
        fields = self.fields_to_analyze or self._detect_numeric_fields(item)

        if self.compute_statistics:
            distribution_stats: dict[str, dict[str, float]] = {}
            for field in fields:
                value = self._extract_field_value(item, field)
                if value is not None:
                    stats = self._compute_field_statistics(value)
                    distribution_stats[field] = stats

            analyzed["distribution_statistics"] = distribution_stats

        return analyzed

    def _detect_numeric_fields(self, item: dict[str, Any]) -> list[str]:
        """Detect numeric fields in item.

        Args:
            item: Data item

        Returns:
            List of numeric field names
        """
        numeric_fields = []

        for key, value in item.items():
            if isinstance(value, (int, float)):
                numeric_fields.append(key)
            elif isinstance(value, (list, tuple)) and value:
                if isinstance(value[0], (int, float)):
                    numeric_fields.append(key)

        return numeric_fields

    def _extract_field_value(self, item: dict[str, Any], field: str) -> Any:
        """Extract field value from item.

        Args:
            item: Data item
            field: Field name

        Returns:
            Field value or None
        """
        if field in item:
            return item[field]

        # Check nested locations
        if "sensor_data" in item and isinstance(item["sensor_data"], dict):
            if field in item["sensor_data"]:
                return item["sensor_data"][field]

        return None

    def _compute_field_statistics(self, value: Any) -> dict[str, float]:
        """Compute statistics for a field value using GPU acceleration.

        Args:
            value: Field value

        Returns:
            Dictionary with statistics
        """
        from pipeline.utils.gpu.arrays import gpu_array_stats
        from pipeline.config import PipelineConfig

        # Try GPU-accelerated statistics first
        try:
            if isinstance(value, (list, tuple)):
                arr = value
            elif isinstance(value, (int, float)):
                arr = [value]
            else:
                return {}

            if len(arr) == 0:
                return {}

            # Use GPU statistics if available
            # Check if GPU is available (would need config access)
            # For now, try GPU first, fallback to CPU
            try:
                stats = gpu_array_stats(arr, num_gpus=1)
                return stats
            except (ImportError, RuntimeError, ValueError):
                # Fallback to CPU NumPy
                import numpy as np

                arr_np = np.array(arr)
                if arr_np.size == 0:
                    return {}

                return {
                    "mean": float(np.mean(arr_np)),
                    "std": float(np.std(arr_np)),
                    "min": float(np.min(arr_np)),
                    "max": float(np.max(arr_np)),
                    "median": float(np.median(arr_np)),
                }
        except ImportError:
            # Pure Python fallback
            if isinstance(value, (list, tuple)) and value:
                numeric_values = [v for v in value if isinstance(v, (int, float))]
                if numeric_values:
                    return {
                        "mean": sum(numeric_values) / len(numeric_values),
                        "min": min(numeric_values),
                        "max": max(numeric_values),
                    }
            elif isinstance(value, (int, float)):
                return {"mean": float(value), "min": float(value), "max": float(value)}

            return {}

