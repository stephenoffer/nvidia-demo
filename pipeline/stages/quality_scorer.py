"""Data quality scoring framework for multimodal data.

Provides comprehensive quality scoring across multiple dimensions:
- Visual quality (for video data)
- Sensor accuracy (for sensor data)
- Text coherence (for text data)
- Cross-modal consistency
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from ray.data import Dataset

from pipeline.stages.base import ProcessorBase
from pipeline.utils.constants import _DEFAULT_BATCH_SIZE, _DEFAULT_QUALITY_THRESHOLD

logger = logging.getLogger(__name__)


class DataQualityScorer(ProcessorBase):
    """Score data quality across multiple dimensions."""

    def __init__(
        self,
        quality_dimensions: Optional[list[str]] = None,
        quality_threshold: float = _DEFAULT_QUALITY_THRESHOLD,
        filter_low_quality: bool = False,
        batch_size: int = _DEFAULT_BATCH_SIZE,
    ):
        """Initialize data quality scorer.

        Args:
            quality_dimensions: List of quality dimensions to score
            quality_threshold: Minimum quality score to keep (0.0-1.0)
            filter_low_quality: Whether to filter out low-quality samples
            batch_size: Batch size for processing
        """
        super().__init__(batch_size=batch_size)
        self.quality_dimensions = quality_dimensions or [
            "visual",
            "sensor",
            "text",
            "consistency",
        ]
        self.quality_threshold = quality_threshold
        self.filter_low_quality = filter_low_quality

    def process(self, dataset: Dataset) -> Dataset:
        """Score data quality for all samples.

        Args:
            dataset: Input Ray Dataset

        Returns:
            Dataset with quality scores added
        """
        logger.info(
            f"Scoring data quality (dimensions={self.quality_dimensions}, "
            f"threshold={self.quality_threshold})"
        )

        scored = super().process(dataset)

        if self.filter_low_quality:
            # Use named function instead of lambda
            def meets_quality_threshold(item: dict[str, Any]) -> bool:
                """Check if item meets quality threshold."""
                return item.get("quality_score", 0.0) >= self.quality_threshold
            
            scored = scored.filter(meets_quality_threshold)

        return scored

    def _process_item(self, item: dict[str, Any]) -> Optional[dict[str, Any]]:
        """Process item by scoring quality.

        Args:
            item: Data item

        Returns:
            Item with quality scores
        """
        return self._score_item(item)

    def _score_item(self, item: dict[str, Any]) -> Optional[dict[str, Any]]:
        """Score quality for a single item.

        Args:
            item: Data item

        Returns:
            Item with quality scores added
        """
        scored = dict(item)

        # Score each dimension
        dimension_scores: dict[str, float] = {}

        for dimension in self.quality_dimensions:
            if dimension == "visual":
                score = self._score_visual_quality(item)
            elif dimension == "sensor":
                score = self._score_sensor_quality(item)
            elif dimension == "text":
                score = self._score_text_quality(item)
            elif dimension == "consistency":
                score = self._score_consistency(item)
            else:
                score = 0.5  # Default score for unknown dimensions

            dimension_scores[dimension] = score

        # Compute overall quality score (weighted average)
        overall_score = sum(dimension_scores.values()) / len(dimension_scores)

        scored["quality_scores"] = dimension_scores
        scored["quality_score"] = overall_score
        scored["quality_threshold"] = self.quality_threshold
        scored["meets_quality_threshold"] = overall_score >= self.quality_threshold

        return scored

    def _score_visual_quality(self, item: dict[str, Any]) -> float:
        """Score visual quality for video/image data.

        Args:
            item: Data item

        Returns:
            Quality score (0.0-1.0)
        """
        from pipeline.utils.data_types import get_data_type, DataType

        if get_data_type(item) != DataType.VIDEO:
            return 1.0

        score = 1.0

        # Check for video metadata
        if "frames" in item:
            frames = item["frames"]
            if isinstance(frames, list) and len(frames) == 0:
                score *= 0.0  # No frames

        # Check resolution
        if "resolution" in item:
            width, height = item["resolution"]
            if width < 64 or height < 64:
                score *= 0.5  # Very low resolution
            elif width < 224 or height < 224:
                score *= 0.8  # Low resolution

        # Check for corruption indicators
        if "error" in item:
            score *= 0.0  # Error indicates corruption

        return max(0.0, min(1.0, score))

    def _score_sensor_quality(self, item: dict[str, Any]) -> float:
        """Score sensor data quality.

        Args:
            item: Data item

        Returns:
            Quality score (0.0-1.0)
        """
        from pipeline.utils.data_types import get_data_type, DataType

        if get_data_type(item) != DataType.SENSOR:
            return 1.0

        score = 1.0

        # Check for sensor data
        sensor_data = item.get("sensor_data")
        if sensor_data is None:
            return 0.0

        # Check data completeness
        if isinstance(sensor_data, dict):
            required_fields = ["observations", "actions"]
            missing_fields = [f for f in required_fields if f not in sensor_data]
            if missing_fields:
                score *= 0.7  # Missing some fields

        # Check for outliers using GPU-accelerated operations
        if "sensor_data" in item and isinstance(item["sensor_data"], dict):
            if "outlier_count" in item["sensor_data"]:
                outlier_ratio = item["sensor_data"]["outlier_count"] / max(
                    item["sensor_data"].get("total_samples", 1), 1
                )
                if outlier_ratio > 0.1:
                    score *= 0.5  # High outlier ratio

        return max(0.0, min(1.0, score))

    def _score_text_quality(self, item: dict[str, Any]) -> float:
        """Score text data quality.

        Args:
            item: Data item

        Returns:
            Quality score (0.0-1.0)
        """
        from pipeline.utils.data_types import get_data_type, DataType, extract_text

        if get_data_type(item) != DataType.TEXT:
            return 1.0

        score = 1.0

        text = extract_text(item) or ""
        if not text:
            return 0.0

        # Check length
        text_len = len(text)
        if text_len < 10:
            score *= 0.3  # Too short
        elif text_len > 10000:
            score *= 0.8  # Very long (may be noisy)

        # Check for boilerplate (basic check)
        boilerplate_indicators = ["cookie policy", "privacy policy", "terms of service"]
        text_lower = text.lower()
        if any(indicator in text_lower for indicator in boilerplate_indicators):
            score *= 0.2  # Contains boilerplate

        # Check for processing errors
        if item.get("processed") is False:
            score *= 0.0  # Processing failed

        return max(0.0, min(1.0, score))

    def _score_consistency(self, item: dict[str, Any]) -> float:
        """Score cross-modal consistency.

        Args:
            item: Data item

        Returns:
            Consistency score (0.0-1.0)
        """
        score = 1.0

        # Check temporal alignment
        if "temporal_alignment_error" in item:
            score *= 0.5  # Temporal alignment issues

        if "temporal_consistency_warning" in item:
            score *= 0.7  # Temporal consistency warning

        from pipeline.utils.data_types import get_data_type, DataType

        data_type = get_data_type(item)
        has_video = data_type == DataType.VIDEO
        has_sensor = data_type == DataType.SENSOR
        has_text = data_type == DataType.TEXT

        # If item claims to be multimodal but missing modalities
        if item.get("multimodal", False):
            modalities_present = sum([has_video, has_sensor, has_text])
            if modalities_present < 2:
                score *= 0.6  # Missing expected modalities

        return max(0.0, min(1.0, score))

