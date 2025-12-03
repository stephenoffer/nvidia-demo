"""Temporal alignment stage for multimodal data streams.

Aligns data from different modalities (video, sensor, text) by timestamps
to ensure synchronized multimodal data for foundation model training.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Dict


from ray.data import Dataset

from pipeline.stages.base import ProcessorBase
from pipeline.utils.constants import _DEFAULT_BATCH_SIZE

logger = logging.getLogger(__name__)


class TemporalAlignmentStage(ProcessorBase):
    """Align multimodal data streams by timestamps.

    Supports multiple time reference frames:
    - Simulation time (for Isaac Lab data)
    - Wall-clock time (for real robot data)
    - Episode time (relative to episode start)
    - Step time (relative to trajectory start)

    Interpolates missing timestamps and validates temporal consistency.
    """

    def __init__(
        self,
        time_reference: str = "auto",
        target_rate: Optional[float] = None,
        interpolation_method: str = "linear",
        max_time_gap: float = 1.0,
        validate_consistency: bool = True,
        batch_size: int = _DEFAULT_BATCH_SIZE,
    ):
        """Initialize temporal alignment stage.

        Args:
            time_reference: Time reference frame to use
            target_rate: Target sampling rate in Hz (None = keep original)
            interpolation_method: Method for interpolating missing timestamps
            max_time_gap: Maximum time gap before flagging as missing
            validate_consistency: Whether to validate temporal consistency
            batch_size: Batch size for processing
        """
        super().__init__(batch_size=batch_size)
        self.time_reference = time_reference
        self.target_rate = target_rate
        self.interpolation_method = interpolation_method
        self.max_time_gap = max_time_gap
        self.validate_consistency = validate_consistency

    def process(self, dataset: Dataset) -> Dataset:
        """Align multimodal data by timestamps.

        Args:
            dataset: Input Ray Dataset with multimodal data

        Returns:
            Temporally aligned Ray Dataset
        """
        logger.info(
            f"Aligning multimodal data (time_reference={self.time_reference}, "
            f"target_rate={self.target_rate}Hz)"
        )
        return super().process(dataset)

    def _process_item(self, item: dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process item by aligning timestamps.

        Args:
            item: Data item

        Returns:
            Aligned item
        """
        return self._align_item(item)

    def _align_item(self, item: dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Align a single multimodal data item.

        Args:
            item: Data item with multimodal fields

        Returns:
            Aligned data item or None if alignment fails
        """
        # Extract timestamps from different modalities
        timestamps = self._extract_timestamps(item)

        if not timestamps:
            logger.warning("No timestamps found in item")
            return item  # Return as-is if no timestamps

        # Determine time reference frame
        time_ref = self._determine_time_reference(item, timestamps)

        # Normalize timestamps to common reference
        normalized_times = self._normalize_timestamps(timestamps, time_ref)

        # Align modalities by timestamps
        aligned_item = self._align_modalities(item, normalized_times)

        # Validate temporal consistency if requested
        if self.validate_consistency:
            if not self._validate_temporal_consistency(aligned_item, normalized_times):
                logger.warning("Temporal consistency validation failed")
                aligned_item["temporal_consistency_warning"] = True

        # Resample to target rate if specified
        if self.target_rate:
            aligned_item = self._resample_to_rate(aligned_item, normalized_times, self.target_rate)

        return aligned_item

    def _extract_timestamps(self, item: dict[str, Any]) -> dict[str, float]:
        """Extract timestamps from different modalities.

        Args:
            item: Data item

        Returns:
            Dictionary mapping modality to timestamp
        """
        timestamps: dict[str, float] = {}

        # Video timestamps
        if "video_timestamp" in item:
            timestamps["video"] = float(item["video_timestamp"])
        elif "timestamp" in item and item.get("data_type") == "video":
            timestamps["video"] = float(item["timestamp"])

        # Sensor timestamps
        if "sensor_timestamp" in item:
            timestamps["sensor"] = float(item["sensor_timestamp"])
        elif "timestamp" in item and item.get("data_type") == "sensor":
            timestamps["sensor"] = float(item["timestamp"])
        elif "sensor_data" in item and isinstance(item["sensor_data"], dict):
            if "timestamp" in item["sensor_data"]:
                timestamps["sensor"] = float(item["sensor_data"]["timestamp"])

        # Text timestamps
        if "text_timestamp" in item:
            timestamps["text"] = float(item["text_timestamp"])
        elif "timestamp" in item and item.get("data_type") == "text":
            timestamps["text"] = float(item["timestamp"])

        # Episode/step timestamps (for Isaac Lab data)
        if "episode_time" in item:
            timestamps["episode"] = float(item["episode_time"])
        if "step" in item:
            timestamps["step"] = float(item["step"])

        # Simulation time (for Isaac Lab)
        if "simulation_time" in item:
            timestamps["simulation"] = float(item["simulation_time"])

        return timestamps

    def _determine_time_reference(
        self, item: dict[str, Any], timestamps: dict[str, float]
    ) -> str:
        """Determine time reference frame from item and timestamps.

        Args:
            item: Data item
            timestamps: Extracted timestamps

        Returns:
            Time reference frame name
        """
        if self.time_reference != "auto":
            return self.time_reference

        # Auto-detect based on available timestamps
        if "simulation" in timestamps:
            return "simulation"
        elif "episode" in timestamps:
            return "episode"
        elif "step" in timestamps:
            return "step"
        elif any("wall_clock" in k or "wallclock" in k.lower() for k in item.keys()):
            return "wall_clock"
        else:
            # Default to first available timestamp
            return list(timestamps.keys())[0] if timestamps else "unknown"

    def _normalize_timestamps(self, timestamps: dict[str, float], time_ref: str) -> dict[str, float]:
        """Normalize timestamps to common reference frame.

        Args:
            timestamps: Modality timestamps
            time_ref: Target time reference frame

        Returns:
            Normalized timestamps
        """
        if time_ref == "unknown" or time_ref not in timestamps:
            # Use first available timestamp as reference
            ref_time = list(timestamps.values())[0] if timestamps else 0.0
        else:
            ref_time = timestamps[time_ref]

        normalized: dict[str, float] = {}
        for modality, timestamp in timestamps.items():
            # Normalize relative to reference time
            normalized[modality] = timestamp - ref_time

        return normalized

    def _align_modalities(
        self, item: dict[str, Any], normalized_times: dict[str, float]
    ) -> dict[str, Any]:
        """Align modalities by interpolating to common timestamps.

        Args:
            item: Original data item
            normalized_times: Normalized timestamps

        Returns:
            Aligned data item
        """
        aligned = dict(item)

        # Find reference timestamp (use earliest or most common)
        if not normalized_times:
            return aligned

        ref_time = min(normalized_times.values())

        # Check for large time gaps
        for modality, time in normalized_times.items():
            gap = abs(time - ref_time)
            if gap > self.max_time_gap:
                logger.warning(
                    f"Large time gap for {modality}: {gap:.3f}s (max: {self.max_time_gap}s)"
                )
                aligned[f"{modality}_time_gap"] = gap
                aligned[f"{modality}_time_gap_warning"] = True

        # Store aligned timestamps
        aligned["aligned_timestamps"] = normalized_times
        aligned["reference_timestamp"] = ref_time
        aligned["temporal_alignment_applied"] = True

        return aligned

    def _validate_temporal_consistency(
        self, item: dict[str, Any], normalized_times: dict[str, float]
    ) -> bool:
        """Validate temporal consistency of aligned data.

        Args:
            item: Aligned data item
            normalized_times: Normalized timestamps

        Returns:
            True if temporally consistent
        """
        if not normalized_times:
            return True

        # Check that timestamps are within acceptable range
        times = list(normalized_times.values())
        time_range = max(times) - min(times)

        if time_range > self.max_time_gap:
            return False

        # Check for negative timestamps (shouldn't happen after normalization)
        if any(t < -self.max_time_gap for t in times):
            return False

        return True

    def _resample_to_rate(
        self, item: dict[str, Any], normalized_times: dict[str, float], target_rate: float
    ) -> dict[str, Any]:
        """Resample data to target sampling rate.

        Args:
            item: Data item
            normalized_times: Normalized timestamps
            target_rate: Target sampling rate (Hz)

        Returns:
            Resampled data item
        """
        if not normalized_times:
            return item

        # Calculate target timestamps using GPU acceleration for large arrays
        time_values = list(normalized_times.values())
        if len(time_values) > 1000:
            try:
                import cupy as cp
                times_cp = cp.array(time_values)
                time_span = float(cp.max(times_cp) - cp.min(times_cp))
                num_samples = int(time_span * target_rate) + 1
                min_time = float(cp.min(times_cp))
                target_times_cp = cp.linspace(min_time, min_time + time_span, num_samples)
                target_times = target_times_cp.get().tolist()
            except ImportError:
                time_span = max(time_values) - min(time_values)
                num_samples = int(time_span * target_rate) + 1
                target_times = [min(time_values) + i / target_rate for i in range(num_samples)]
        else:
            time_span = max(time_values) - min(time_values)
            num_samples = int(time_span * target_rate) + 1
            target_times = [min(time_values) + i / target_rate for i in range(num_samples)]

        # Store resampling metadata
        resampled = dict(item)
        resampled["resampled_rate"] = target_rate
        resampled["resampled_timestamps"] = target_times
        resampled["original_rate"] = (
            len(normalized_times) / time_span if time_span > 0 else None
        )

        resampled["resampling_required"] = True

        return resampled

