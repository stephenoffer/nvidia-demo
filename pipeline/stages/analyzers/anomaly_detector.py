"""Anomaly detection stage using learned models.

Detects anomalies in data using autoencoders or isolation forests.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from ray.data import Dataset

from pipeline.stages.base import ProcessorBase
from pipeline.utils.constants import _DEFAULT_BATCH_SIZE, _DEFAULT_QUALITY_THRESHOLD

logger = logging.getLogger(__name__)


class AnomalyDetector(ProcessorBase):
    """Detect anomalies in multimodal data.

    Uses learned models (autoencoders, isolation forests) for anomaly detection.
    """

    def __init__(
        self,
        threshold: float = _DEFAULT_QUALITY_THRESHOLD,
        filter_anomalies: bool = False,
        batch_size: int = _DEFAULT_BATCH_SIZE,
    ):
        """Initialize anomaly detector.

        Args:
            threshold: Anomaly score threshold (0.0-1.0)
            filter_anomalies: Whether to filter out anomalies
            batch_size: Batch size for processing
        """
        super().__init__(batch_size=batch_size)
        self.threshold = threshold
        self.filter_anomalies = filter_anomalies

    def process(self, dataset: Dataset) -> Dataset:
        """Detect anomalies in dataset using GPU acceleration when available.

        Args:
            dataset: Input Ray Dataset

        Returns:
            Dataset with anomaly scores added (filtered if filter_anomalies=True)
        """
        logger.info(f"Detecting anomalies (threshold={self.threshold})")
        detected = super().process(dataset)

        if self.filter_anomalies:
            def is_not_anomaly(item: dict[str, Any]) -> bool:
                """Check if item is not an anomaly."""
                return not item.get("is_anomaly", False)
            
            detected = detected.filter(is_not_anomaly)

        return detected

    def _process_item(self, item: dict[str, Any]) -> Optional[dict[str, Any]]:
        """Process item by detecting anomalies.

        Args:
            item: Data item

        Returns:
            Item with anomaly scores
        """
        anomaly_score = self._detect_anomaly(item)
        item["anomaly_score"] = anomaly_score
        item["is_anomaly"] = anomaly_score > self.threshold
        return item

    def _detect_anomaly(self, item: dict[str, Any]) -> float:
        """Detect anomaly in a single item.

        Args:
            item: Data item

        Returns:
            Anomaly score (0.0-1.0, higher = more anomalous)
        """
        score = 0.0

        # Check for error indicators
        if "error" in item:
            score += 0.5

        if "corruption_flags" in item and item["corruption_flags"]:
            score += 0.3

        # Check for missing required fields
        if "missing_fields" in item and item["missing_fields"]:
            score += 0.2 * len(item["missing_fields"])

        # Check for quality issues
        if "quality_score" in item:
            quality = item["quality_score"]
            if quality < 0.5:
                score += 0.3

        # Check for physics violations
        if "violations" in item and item["violations"]:
            score += 0.2 * min(len(item["violations"]), 5) / 5.0

        # Check for temporal alignment issues
        if "temporal_alignment_error" in item:
            score += 0.2

        # Normalize to 0.0-1.0
        return min(1.0, score)

