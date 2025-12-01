"""Modality dropout stage for training robustness.

Randomly drops modalities during training to improve model robustness
to missing modalities. Critical for GR00T: Models must handle missing
modalities gracefully.
"""

from __future__ import annotations

import logging
import random
from typing import Any, Optional, Dict

from ray.data import Dataset

from pipeline.stages.base import ProcessorBase
from pipeline.utils.constants import _DEFAULT_BATCH_SIZE

logger = logging.getLogger(__name__)


class ModalityDropoutStage(ProcessorBase):
    """Randomly drop modalities for training robustness.

    Supports configurable dropout rates per modality and ensures
    training data includes missing modality examples.
    """

    def __init__(
        self,
        video_dropout_rate: float = 0.1,
        sensor_dropout_rate: float = 0.1,
        text_dropout_rate: float = 0.1,
        min_modalities: int = 1,
        seed: Optional[int] = None,
        batch_size: int = _DEFAULT_BATCH_SIZE,
    ):
        """Initialize modality dropout stage.

        Args:
            video_dropout_rate: Probability of dropping video modality (0.0-1.0)
            sensor_dropout_rate: Probability of dropping sensor modality (0.0-1.0)
            text_dropout_rate: Probability of dropping text modality (0.0-1.0)
            min_modalities: Minimum number of modalities to keep
            seed: Random seed for reproducibility
            batch_size: Batch size for processing
        """
        super().__init__(batch_size=batch_size)
        self.video_dropout_rate = video_dropout_rate
        self.sensor_dropout_rate = sensor_dropout_rate
        self.text_dropout_rate = text_dropout_rate
        self.min_modalities = min_modalities

        if seed is not None:
            random.seed(seed)

    def process(self, dataset: Dataset) -> Dataset:
        """Apply modality dropout to dataset.

        Args:
            dataset: Input Ray Dataset with multimodal data

        Returns:
            Dataset with modalities randomly dropped
        """
        logger.info(
            f"Applying modality dropout (video={self.video_dropout_rate}, "
            f"sensor={self.sensor_dropout_rate}, text={self.text_dropout_rate})"
        )
        return super().process(dataset)

    def _process_item(self, item: dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process item by applying modality dropout.

        Args:
            item: Data item

        Returns:
            Item with modalities dropped
        """
        return self._apply_dropout(item)

    def _apply_dropout(self, item: dict[str, Any]) -> dict[str, Any]:
        """Apply modality dropout to a single item.

        Args:
            item: Data item

        Returns:
            Item with modalities dropped
        """
        dropped = dict(item)
        dropped_modalities = []

        # Detect available modalities
        has_video = "video" in item or "frames" in item or "video_data" in item
        has_sensor = (
            "sensor_data" in item
            or "observations" in item
            or "actions" in item
            or "sensor" in item
        )
        has_text = "text" in item or "content" in item or "instruction" in item

        available_modalities = []
        if has_video:
            available_modalities.append("video")
        if has_sensor:
            available_modalities.append("sensor")
        if has_text:
            available_modalities.append("text")

        # Apply dropout
        if has_video and random.random() < self.video_dropout_rate:
            if len(available_modalities) > self.min_modalities:
                self._drop_modality(dropped, "video")
                dropped_modalities.append("video")
                available_modalities.remove("video")

        if has_sensor and random.random() < self.sensor_dropout_rate:
            if len(available_modalities) > self.min_modalities:
                self._drop_modality(dropped, "sensor")
                dropped_modalities.append("sensor")
                if "sensor" in available_modalities:
                    available_modalities.remove("sensor")

        if has_text and random.random() < self.text_dropout_rate:
            if len(available_modalities) > self.min_modalities:
                self._drop_modality(dropped, "text")
                dropped_modalities.append("text")
                if "text" in available_modalities:
                    available_modalities.remove("text")

        # Record dropout information
        dropped["modality_dropout_applied"] = True
        dropped["dropped_modalities"] = dropped_modalities
        dropped["remaining_modalities"] = available_modalities
        dropped["has_missing_modalities"] = len(dropped_modalities) > 0

        return dropped

    def _drop_modality(self, item: dict[str, Any], modality: str) -> None:
        """Drop a specific modality from item.

        Args:
            item: Data item
            modality: Modality to drop
        """
        if modality == "video":
            for key in ["video", "frames", "video_data", "video_bytes"]:
                if key in item:
                    item[f"{key}_dropped"] = True
                    # Don't delete, mark as None for training compatibility
                    item[key] = None

        elif modality == "sensor":
            for key in [
                "sensor_data",
                "observations",
                "actions",
                "sensor",
                "joint_positions",
                "joint_velocities",
            ]:
                if key in item:
                    item[f"{key}_dropped"] = True
                    item[key] = None

        elif modality == "text":
            for key in ["text", "content", "instruction", "caption"]:
                if key in item:
                    item[f"{key}_dropped"] = True
                    item[key] = None

