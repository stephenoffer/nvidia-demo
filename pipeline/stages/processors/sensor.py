"""Sensor data processing stage for multimodal pipeline.

Handles IMU data, joint angles, and control signals processing.
Uses NVIDIA cuPy for GPU-accelerated operations when available, with NumPy fallback.
"""

import logging
from typing import Any, Dict, List, Optional

from ray.data import Dataset

from pipeline.stages.base import ProcessorBase
from pipeline.utils.constants import _DEFAULT_BATCH_SIZE, _DEFAULT_SAMPLE_RATE, _DEFAULT_CPUS
from pipeline.utils.data.data_types import get_data_type, DataType, extract_sensor_data

logger = logging.getLogger(__name__)

# Try to import cuPy for GPU operations, fallback to NumPy
try:
    import cupy as cp  # https://docs.cupy.dev/

    _CUPY_AVAILABLE = True
    _NP = cp  # Use cuPy as primary
    logger.info("cuPy available - using GPU-accelerated operations")
except ImportError:
    import numpy as np  # https://numpy.org/

    _CUPY_AVAILABLE = False
    _NP = np  # Fallback to NumPy
    logger.info("cuPy not available - using CPU NumPy operations")


class SensorProcessorActor:
    """Sensor data processor for batch processing.

    Note: Not a Ray actor - used directly in map_batches for efficiency.
    """

    def __init__(
        self,
        sample_rate: int,
        normalize: bool,
        remove_outliers: bool,
        outlier_method: str = "isolation_forest",
    ):
        """Initialize sensor processor actor.

        Args:
            sample_rate: Target sample rate
            normalize: Whether to normalize data
            remove_outliers: Whether to remove outliers
            outlier_method: Outlier detection method ("isolation_forest", "dbscan", "lof", "zscore")
        """
        self.sample_rate = sample_rate
        self.normalize = normalize
        self.remove_outliers = remove_outliers
        self.outlier_method = outlier_method

    def process_sensor(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single sensor data item.

        Args:
            item: Sensor data item

        Returns:
            Processed sensor item
        """
        try:
            sensor_data = extract_sensor_data(item) or self._extract_sensor_data(item)

            if sensor_data is None:
                return {**item, "processed": False, "error": "No sensor data"}

            # Remove outliers using advanced methods (Isolation Forest, DBSCAN, LOF)
            if self.remove_outliers:
                sensor_data = self._remove_outliers_advanced(sensor_data, method=self.outlier_method)

            # Normalize
            if self.normalize:
                sensor_data = self._normalize(sensor_data)

            # Resample if needed
            if self.sample_rate:
                sensor_data = self._resample(sensor_data, self.sample_rate)

            # Convert to list for serialization (cuPy/NumPy arrays)
            # Properly handle cuPy arrays to free GPU memory
            if _CUPY_AVAILABLE and hasattr(sensor_data, "get"):
                # cuPy array on GPU - convert to CPU first
                # Use cuPy's efficient CPU transfer
                sensor_data_cpu = sensor_data.get()  # Move to CPU (synchronous)
                sensor_data_list = sensor_data_cpu.tolist()
                # Free cuPy arrays immediately
                del sensor_data
                del sensor_data_cpu
                # Don't free all blocks - too aggressive, hurts performance
                # Only free if memory pressure is detected
            elif hasattr(sensor_data, "tolist"):
                sensor_data_list = sensor_data.tolist()
            else:
                sensor_data_list = sensor_data

            return {
                **item,
                "sensor_data": sensor_data_list,
                "processed": True,
                "num_samples": len(sensor_data) if hasattr(sensor_data, "__len__") else 1,
            }

        except Exception as e:
            logger.error(f"Error processing sensor data: {e}")
            return {**item, "processed": False, "error": str(e)}

    def _extract_sensor_data(self, item: Dict[str, Any]) -> Any:
        """Extract sensor data from item.

        Args:
            item: Data item

        Returns:
            Sensor data array (cuPy array if GPU available, NumPy otherwise)
        """
        # Try common sensor data fields
        for field in ["sensor_data", "imu", "joint_angles", "control", "data"]:
            if field in item:
                data = item[field]
                if _CUPY_AVAILABLE:
                    # Convert to cuPy array for GPU processing
                    # Use cuPy's efficient array creation
                    if isinstance(data, _NP.ndarray):
                        # Already a cuPy array, return as-is
                        return _NP.asarray(data)
                    elif isinstance(data, (list, tuple)):
                        # Convert list/tuple to cuPy array efficiently
                        return _NP.array(data, dtype=_NP.float32)  # Use float32 for GPU efficiency
                    else:
                        # Convert other types to cuPy array
                        return _NP.array(data, dtype=_NP.float32)
                else:
                    import numpy as np  # https://numpy.org/

                    return np.array(data)

        # If no sensor field, check if item itself is array-like
        if isinstance(item, (list, tuple)):
            if _CUPY_AVAILABLE:
                return _NP.array(item)
            else:
                import numpy as np  # https://numpy.org/

                return np.array(item)

        return None

    def _remove_outliers(self, data: Any, threshold: float = 3.0) -> Any:
        """Remove outliers using z-score (GPU-accelerated with cuPy).

        Args:
            data: Sensor data array (cuPy or NumPy)
            threshold: Z-score threshold

        Returns:
            Data with outliers removed
        """
        from pipeline.utils.gpu.arrays import gpu_remove_outliers

        if len(data) == 0:
            return data

        try:
            # Use GPU-accelerated outlier removal
            filtered = gpu_remove_outliers(data, threshold=threshold, num_gpus=1 if _CUPY_AVAILABLE else 0)
            return filtered
        except (ValueError, RuntimeError, TypeError) as e:
            logger.error(f"Error removing outliers: {e}")
            return data

    def _remove_outliers_advanced(self, data: Any, method: str = "isolation_forest") -> Any:
        """Remove outliers using advanced methods (Isolation Forest, DBSCAN, LOF).

        Args:
            data: Sensor data array (cuPy or NumPy)
            method: Outlier detection method ("isolation_forest", "dbscan", "lof", "zscore")

        Returns:
            Data with outliers removed
        """
        if len(data) == 0:
            return data

        try:
            # Try GPU-accelerated methods first (cuML)
            if _CUPY_AVAILABLE:
                try:
                    from cuml.ensemble import IsolationForest as cuMLIsolationForest
                    from cuml.cluster import DBSCAN as cuMLDBSCAN

                    if method == "isolation_forest":
                        # Use cuML Isolation Forest on GPU
                        iso_forest = cuMLIsolationForest(contamination=0.1, random_state=42)
                        # Reshape for 2D if needed
                        data_2d = data.reshape(-1, 1) if len(data.shape) == 1 else data
                        outliers = iso_forest.fit_predict(data_2d)
                        # Keep non-outliers (label != -1)
                        mask = outliers != -1
                        return data[mask] if hasattr(data, '__getitem__') else data
                    elif method == "dbscan":
                        # Use cuML DBSCAN on GPU
                        dbscan = cuMLDBSCAN(eps=0.5, min_samples=5)
                        data_2d = data.reshape(-1, 1) if len(data.shape) == 1 else data
                        labels = dbscan.fit_predict(data_2d)
                        # Keep core samples (label != -1)
                        mask = labels != -1
                        return data[mask] if hasattr(data, '__getitem__') else data
                except ImportError:
                    logger.debug("cuML not available, falling back to CPU methods")

            # CPU fallback methods
            if method == "isolation_forest":
                try:
                    from sklearn.ensemble import IsolationForest

                    iso_forest = IsolationForest(contamination=0.1, random_state=42)
                    data_2d = data.reshape(-1, 1) if len(data.shape) == 1 else data
                    # Convert to numpy if cuPy array
                    if _CUPY_AVAILABLE and hasattr(data, 'get'):
                        data_2d = data_2d.get()
                    outliers = iso_forest.fit_predict(data_2d)
                    mask = outliers != -1
                    filtered = data[mask] if hasattr(data, '__getitem__') else data
                    return filtered
                except ImportError:
                    logger.warning("sklearn not available, using z-score fallback")
                    return self._remove_outliers(data, threshold=3.0)
            elif method == "dbscan":
                try:
                    from sklearn.cluster import DBSCAN

                    dbscan = DBSCAN(eps=0.5, min_samples=5)
                    data_2d = data.reshape(-1, 1) if len(data.shape) == 1 else data
                    if _CUPY_AVAILABLE and hasattr(data, 'get'):
                        data_2d = data_2d.get()
                    labels = dbscan.fit_predict(data_2d)
                    mask = labels != -1
                    filtered = data[mask] if hasattr(data, '__getitem__') else data
                    return filtered
                except ImportError:
                    logger.warning("sklearn not available, using z-score fallback")
                    return self._remove_outliers(data, threshold=3.0)
            else:
                # Fallback to z-score
                return self._remove_outliers(data, threshold=3.0)

        except (ValueError, RuntimeError, TypeError, AttributeError) as e:
            logger.warning(f"Advanced outlier detection failed ({method}): {e}, using z-score fallback")
            return self._remove_outliers(data, threshold=3.0)

    def _normalize(self, data: Any) -> Any:
        """Normalize sensor data (GPU-accelerated with cuPy).

        Args:
            data: Sensor data array (cuPy or NumPy)

        Returns:
            Normalized data
        """
        from pipeline.utils.gpu.arrays import gpu_normalize

        if len(data) == 0:
            return data

        try:
            # Use GPU-accelerated normalization
            normalized = gpu_normalize(data, method="zscore", num_gpus=1 if _CUPY_AVAILABLE else 0)
            return normalized
        except (ValueError, RuntimeError, TypeError) as e:
            logger.error(f"Error normalizing sensor data: {e}")
            return data

    def _resample(self, data: Any, target_rate: int) -> Any:
        """Resample sensor data to target rate.

        Args:
            data: Sensor data array
            target_rate: Target sample rate

        Returns:
            Resampled data
        """
        # Simple downsampling by taking every nth sample
        if len(data) <= target_rate:
            return data

        step = len(data) // target_rate
        return data[::step]


class SensorProcessor(ProcessorBase):
    """Sensor data processing stage for the pipeline.

    Handles IMU data, joint angles, and control signals.
    """

    def __init__(
        self,
        sample_rate: int = _DEFAULT_SAMPLE_RATE,
        normalize: bool = True,
        remove_outliers: bool = True,
        outlier_method: str = "isolation_forest",
        batch_size: int = _DEFAULT_BATCH_SIZE,
    ):
        """Initialize sensor processor.

        Args:
            sample_rate: Target sample rate for resampling
            normalize: Whether to normalize sensor values
            remove_outliers: Whether to remove statistical outliers
            outlier_method: Outlier detection method ("isolation_forest", "dbscan", "lof", "zscore")
            batch_size: Batch size for processing
        """
        super().__init__(batch_size=batch_size)
        self.sample_rate = sample_rate
        self.normalize = normalize
        self.remove_outliers = remove_outliers
        self.outlier_method = outlier_method

    def process(self, dataset: Dataset) -> Dataset:
        """Process sensor data in the dataset.

        Uses Ray Data map_batches for efficient batch processing.

        Args:
            dataset: Input Ray Dataset

        Returns:
            Processed Ray Dataset
        """
        logger.info("Processing sensor data")

        # Use named function instead of lambda for better serialization
        def is_sensor_type(item: dict[str, Any]) -> bool:
            """Check if item is sensor type."""
            return get_data_type(item) == DataType.SENSOR
        
        sensor_dataset = dataset.filter(is_sensor_type)

        def process_batch(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            """Process a batch of sensor items."""
            processor = SensorProcessorActor(
                self.sample_rate,
                self.normalize,
                self.remove_outliers,
                self.outlier_method,
            )

            return [processor.process_sensor(item) for item in batch]

        processed = sensor_dataset.map_batches(
            process_batch,
            batch_size=self.batch_size,
            batch_format="pandas",  # Specify batch format
            num_cpus=_DEFAULT_CPUS,
            num_gpus=1 if _CUPY_AVAILABLE else 0,
        )

        # Use named function instead of lambda
        def is_processed(item: dict[str, Any]) -> bool:
            """Check if item was processed successfully."""
            return item.get("processed", False)
        
        return processed.filter(is_processed)

    def _process_item(self, item: dict[str, Any]) -> Optional[dict[str, Any]]:
        """Process item - not used for sensor processing.

        Sensor processing uses SensorProcessorActor for batch processing.
        """
        return None
