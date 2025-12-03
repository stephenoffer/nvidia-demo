"""Cross-modal consistency validation for multimodal data.

Validates that different modalities are consistent with each other:
- Temporal alignment
- Semantic consistency (e.g., text describes video)
- Data completeness across modalities
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from ray.data import Dataset

from pipeline.stages.base import ValidatorBase
from pipeline.utils.constants import _DEFAULT_BATCH_SIZE
from pipeline.utils.data.data_types import detect_modalities, Modality

logger = logging.getLogger(__name__)


class CrossModalValidator(ValidatorBase):
    """Validate consistency across different modalities.

    Checks temporal alignment, semantic consistency, and data completeness.
    """

    def __init__(
        self,
        check_temporal_alignment: bool = True,
        check_semantic_consistency: bool = False,
        check_completeness: bool = True,
        max_temporal_gap: float = 0.1,
        required_modalities: Optional[list[str]] = None,
        batch_size: int = _DEFAULT_BATCH_SIZE,
    ):
        """Initialize cross-modal validator.

        Args:
            check_temporal_alignment: Whether to check temporal alignment
            check_semantic_consistency: Whether to check semantic consistency (requires models)
            check_completeness: Whether to check data completeness
            max_temporal_gap: Maximum allowed temporal gap between modalities
            required_modalities: List of required modalities (None = auto-detect)
            batch_size: Batch size for processing
        """
        super().__init__(reject_invalid=False, batch_size=batch_size)
        self.check_temporal_alignment = check_temporal_alignment
        self.check_semantic_consistency = check_semantic_consistency
        self.check_completeness = check_completeness
        self.max_temporal_gap = max_temporal_gap
        self.required_modalities = required_modalities

    def process(self, dataset: Dataset) -> Dataset:
        """Validate cross-modal consistency.

        Args:
            dataset: Input Ray Dataset with multimodal data

        Returns:
            Dataset with validation results
        """
        logger.info("Validating cross-modal consistency")
        return super().process(dataset)

    def _validate_item(self, item: dict[str, Any]) -> dict[str, Any]:
        """Validate cross-modal consistency for a single item.

        Args:
            item: Multimodal data item

        Returns:
            Validation result dictionary
        """
        violations: list[str] = []
        warnings: list[str] = []

        modalities_list = detect_modalities(item)
        modalities = [m.value for m in modalities_list]

        # Check completeness
        if self.check_completeness:
            completeness_issues = self._check_completeness(item, modalities)
            violations.extend(completeness_issues)

        # Check temporal alignment
        if self.check_temporal_alignment and len(modalities) > 1:
            alignment_issues = self._check_temporal_alignment(item, modalities)
            violations.extend(alignment_issues)

        # Check semantic consistency (placeholder - requires learned models)
        if self.check_semantic_consistency and len(modalities) > 1:
            semantic_issues = self._check_semantic_consistency(item, modalities)
            violations.extend(semantic_issues)

        is_consistent = len(violations) == 0

        return {
            "is_valid": is_consistent,
            "cross_modal_consistent": is_consistent,
            "cross_modal_violations": violations,
            "cross_modal_warnings": warnings,
            "detected_modalities": modalities,
            "num_modalities": len(modalities),
        }

    def _check_completeness(
        self, item: dict[str, Any], modalities: list[str]
    ) -> list[str]:
        """Check data completeness across modalities.

        Args:
            item: Data item
            modalities: List of detected modalities

        Returns:
            List of completeness violation messages
        """
        violations = []

        from pipeline.utils.data.data_types import extract_text, extract_sensor_data

        if self.required_modalities:
            missing = set(self.required_modalities) - set(modalities)
            if missing:
                violations.append(f"Missing required modalities: {missing}")

        if Modality.VIDEO.value in modalities:
            if "frames" not in item and "video" not in item:
                violations.append("Video modality declared but no video data found")

        if Modality.SENSOR.value in modalities:
            sensor_data = extract_sensor_data(item)
            if not sensor_data:
                violations.append("Sensor modality declared but no sensor data found")

        if Modality.TEXT.value in modalities:
            text = extract_text(item)
            if not text:
                violations.append("Text modality declared but no text data found")

        return violations

    def _check_temporal_alignment(
        self, item: dict[str, Any], modalities: list[str]
    ) -> list[str]:
        """Check temporal alignment across modalities.

        Args:
            item: Data item
            modalities: List of detected modalities

        Returns:
            List of temporal alignment violation messages
        """
        violations = []

        # Extract timestamps for each modality
        timestamps: dict[str, float] = {}

        if "video" in modalities:
            video_ts = item.get("video_timestamp") or item.get("timestamp")
            if video_ts:
                timestamps["video"] = float(video_ts)

        if "sensor" in modalities:
            sensor_ts = item.get("sensor_timestamp") or item.get("timestamp")
            if sensor_ts:
                timestamps["sensor"] = float(sensor_ts)

        if "text" in modalities:
            text_ts = item.get("text_timestamp") or item.get("timestamp")
            if text_ts:
                timestamps["text"] = float(text_ts)

        # Check alignment if multiple timestamps available
        if len(timestamps) > 1:
            times = list(timestamps.values())
            time_range = max(times) - min(times)

            if time_range > self.max_temporal_gap:
                violations.append(
                    f"Temporal misalignment: gap {time_range:.3f}s > "
                    f"max {self.max_temporal_gap}s"
                )

        # Check for temporal alignment metadata
        if "temporal_alignment_error" in item:
            violations.append("Temporal alignment error detected")

        if "temporal_consistency_warning" in item:
            violations.append("Temporal consistency warning")

        return violations

    def _check_semantic_consistency(
        self, item: dict[str, Any], modalities: list[str]
    ) -> list[str]:
        """Check semantic consistency across modalities.

        Placeholder - requires learned models (e.g., CLIP for video-text,
        learned encoders for sensor-text).

        Args:
            item: Data item
            modalities: List of detected modalities

        Returns:
            List of semantic consistency violation messages
        """
        violations = []

        # Placeholder: Would use learned models to check consistency
        # For example:
        # - Video-text: Use CLIP to check if text describes video
        # - Sensor-text: Use learned encoders to check if text describes sensor state
        # - Video-sensor: Use learned models to check if sensor matches video scene

        if self.check_semantic_consistency:
            # For now, just check that modalities are present
            # Real implementation would use NeMo/CLIP models
            logger.debug("Semantic consistency check requires learned models (not implemented)")

        return violations

