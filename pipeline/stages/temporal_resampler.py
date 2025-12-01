"""Temporal resampling stage for multimodal data.

Resamples data from different modalities to a common temporal resolution.
Supports GPU-accelerated resampling for large sequences.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Dict

from ray.data import Dataset

from pipeline.stages.base import ProcessorBase
from pipeline.utils.constants import _DEFAULT_BATCH_SIZE

logger = logging.getLogger(__name__)


class TemporalResampler(ProcessorBase):
    """Resample multimodal data to target temporal resolution.

    Supports multiple resampling methods (linear, cubic, nearest) with
    GPU acceleration for large sequences.
    """

    def __init__(
        self,
        target_rate: float,
        resampling_method: str = "linear",
        preserve_temporal_relationships: bool = True,
        use_gpu: bool = True,
        batch_size: int = _DEFAULT_BATCH_SIZE,
    ):
        """Initialize temporal resampler.

        Args:
            target_rate: Target sampling rate in Hz
            resampling_method: Resampling method ("linear", "cubic", "nearest")
            preserve_temporal_relationships: Whether to preserve temporal relationships
            use_gpu: Whether to use GPU acceleration
            batch_size: Batch size for processing
        """
        super().__init__(batch_size=batch_size)
        self.target_rate = target_rate
        self.resampling_method = resampling_method
        self.preserve_temporal_relationships = preserve_temporal_relationships
        self.use_gpu = use_gpu

    def process(self, dataset: Dataset) -> Dataset:
        """Resample data to target rate.

        Args:
            dataset: Input Ray Dataset with multimodal data

        Returns:
            Resampled Ray Dataset
        """
        logger.info(
            f"Resampling data to {self.target_rate}Hz using {self.resampling_method} method"
        )
        return super().process(dataset)

    def _process_item(self, item: dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process item by resampling temporal data.

        Args:
            item: Data item

        Returns:
            Resampled item
        """
        return self._resample_item(item)

    def _resample_item(self, item: dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Resample a single data item.

        Args:
            item: Data item with temporal data

        Returns:
            Resampled data item
        """
        resampled = dict(item)

        # Extract timestamps
        timestamps = self._extract_timestamps(item)
        if not timestamps:
            logger.debug("No timestamps found, skipping resampling")
            return item

        # Calculate target timestamps
        time_span = max(timestamps) - min(timestamps)
        num_samples = int(time_span * self.target_rate) + 1
        target_timestamps = self._generate_target_timestamps(
            min(timestamps), max(timestamps), num_samples
        )

        # Resample each modality
        for modality in ["video", "sensor", "text"]:
            if modality in item or f"{modality}_data" in item:
                resampled_data = self._resample_modality(
                    item, modality, timestamps, target_timestamps
                )
                if resampled_data is not None:
                    resampled[f"{modality}_resampled"] = resampled_data
                    resampled[f"{modality}_original_rate"] = len(timestamps) / time_span if time_span > 0 else None
                    resampled[f"{modality}_resampled_rate"] = self.target_rate

        resampled["temporal_resampling_applied"] = True
        resampled["resampling_method"] = self.resampling_method

        return resampled

    def _extract_timestamps(self, item: dict[str, Any]) -> list[float]:
        """Extract timestamps from item.

        Args:
            item: Data item

        Returns:
            List of timestamps
        """
        timestamps = []

        # Try various timestamp fields
        for field in [
            "timestamp",
            "timestamps",
            "time",
            "video_timestamp",
            "sensor_timestamp",
            "text_timestamp",
        ]:
            if field in item:
                value = item[field]
                if isinstance(value, (list, tuple)):
                    timestamps.extend([float(t) for t in value])
                elif isinstance(value, (int, float)):
                    timestamps.append(float(value))

        # Extract from aligned timestamps if available
        if "aligned_timestamps" in item:
            aligned = item["aligned_timestamps"]
            if isinstance(aligned, dict):
                timestamps.extend([float(t) for t in aligned.values()])

        return sorted(set(timestamps)) if timestamps else []

    def _generate_target_timestamps(
        self, start_time: float, end_time: float, num_samples: int
    ) -> list[float]:
        """Generate target timestamps for resampling.

        Args:
            start_time: Start time
            end_time: End time
            num_samples: Number of samples

        Returns:
            List of target timestamps
        """
        if num_samples <= 1:
            return [start_time]

        # Use GPU acceleration for large arrays
        if self.use_gpu and num_samples > 1000:
            try:
                import cupy as cp

                target_times_cp = cp.linspace(start_time, end_time, num_samples)
                return target_times_cp.get().tolist()
            except ImportError:
                pass

        # CPU fallback
        import numpy as np

        return np.linspace(start_time, end_time, num_samples).tolist()

    def _resample_modality(
        self,
        item: dict[str, Any],
        modality: str,
        original_timestamps: list[float],
        target_timestamps: list[float],
    ) -> Optional[Any]:
        """Resample a single modality.

        Args:
            item: Data item
            modality: Modality name
            original_timestamps: Original timestamps
            target_timestamps: Target timestamps

        Returns:
            Resampled data or None
        """
        # Extract modality data
        data = item.get(modality) or item.get(f"{modality}_data")
        if data is None:
            return None

        # Convert to array format
        if isinstance(data, (list, tuple)):
            data_array = list(data)
        else:
            data_array = data

        if len(data_array) != len(original_timestamps):
            logger.warning(
                f"Data length ({len(data_array)}) doesn't match timestamps ({len(original_timestamps)})"
            )
            return None

        # Resample using selected method
        if self.resampling_method == "linear":
            return self._linear_resample(
                original_timestamps, data_array, target_timestamps
            )
        elif self.resampling_method == "cubic":
            return self._cubic_resample(
                original_timestamps, data_array, target_timestamps
            )
        elif self.resampling_method == "nearest":
            return self._nearest_resample(
                original_timestamps, data_array, target_timestamps
            )
        else:
            logger.warning(f"Unknown resampling method: {self.resampling_method}")
            return None

    def _linear_resample(
        self, x_original: list[float], y_original: list[Any], x_target: list[float]
    ) -> list[Any]:
        """Linear interpolation resampling.

        Args:
            x_original: Original x values (timestamps)
            y_original: Original y values (data)
            x_target: Target x values (timestamps)

        Returns:
            Resampled y values
        """
        if self.use_gpu and len(x_target) > 1000:
            try:
                import cupy as cp

                x_orig_cp = cp.array(x_original)
                x_targ_cp = cp.array(x_target)
                y_orig_cp = cp.array(y_original, dtype=cp.float32)

                # Linear interpolation on GPU
                y_target_cp = cp.interp(x_targ_cp, x_orig_cp, y_orig_cp)
                return y_target_cp.get().tolist()
            except (ImportError, ValueError, TypeError):
                pass

        # CPU fallback using NumPy
        import numpy as np

        y_target = np.interp(x_target, x_original, y_original)
        return y_target.tolist()

    def _cubic_resample(
        self, x_original: list[float], y_original: list[Any], x_target: list[float]
    ) -> list[Any]:
        """Cubic spline interpolation resampling.

        Args:
            x_original: Original x values (timestamps)
            y_original: Original y values (data)
            x_target: Target x values (timestamps)

        Returns:
            Resampled y values
        """
        try:
            from scipy.interpolate import interp1d

            f = interp1d(x_original, y_original, kind="cubic", fill_value="extrapolate")
            return f(x_target).tolist()
        except ImportError:
            logger.warning("scipy not available, falling back to linear interpolation")
            return self._linear_resample(x_original, y_original, x_target)

    def _nearest_resample(
        self, x_original: list[float], y_original: list[Any], x_target: list[float]
    ) -> list[Any]:
        """Nearest neighbor resampling.

        Args:
            x_original: Original x values (timestamps)
            y_original: Original y values (data)
            x_target: Target x values (timestamps)

        Returns:
            Resampled y values
        """
        if self.use_gpu and len(x_target) > 1000:
            try:
                import cupy as cp

                x_orig_cp = cp.array(x_original)
                x_targ_cp = cp.array(x_target)
                y_orig_cp = cp.array(y_original, dtype=cp.float32)

                # Find nearest indices
                indices = cp.searchsorted(x_orig_cp, x_targ_cp, side="left")
                # Clamp indices to valid range
                indices = cp.clip(indices, 0, len(y_original) - 1)
                y_target_cp = y_orig_cp[indices]
                return y_target_cp.get().tolist()
            except (ImportError, ValueError, TypeError):
                pass

        # CPU fallback
        import numpy as np

        indices = np.searchsorted(x_original, x_target, side="left")
        indices = np.clip(indices, 0, len(y_original) - 1)
        y_target = np.array(y_original)[indices]
        return y_target.tolist()

