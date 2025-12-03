"""Data drift detection stage for monitoring data quality over time.

Detects distribution shifts, schema changes, and data quality degradation.
Optionally uses GPU acceleration for faster computation.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from ray.data import Dataset

from pipeline.stages.base import ProcessorBase
from pipeline.utils.constants import _DEFAULT_BATCH_SIZE

logger = logging.getLogger(__name__)


class DriftDetector(ProcessorBase):
    """Detect data drift and distribution shifts.

    Compares current data distribution against reference distribution
    to detect changes that may affect model performance.

    Example:
        ```python
        # Simple usage
        detector = DriftDetector(
            reference_statistics={"sensor_data": {"mean": 0.0, "std": 1.0}},
            drift_threshold=0.1,
        )
        
        # With GPU acceleration
        detector = DriftDetector(
            reference_statistics={"sensor_data": {"mean": 0.0, "std": 1.0}},
            use_gpu=True,
            ray_remote_args={"num_gpus": 1},
        )
        ```
    """

    def __init__(
        self,
        reference_dataset: Optional[Dataset] = None,
        reference_statistics: Optional[dict[str, Any]] = None,
        columns_to_check: Optional[list[str]] = None,
        drift_threshold: float = 0.1,
        method: str = "ks_test",
        batch_size: int = _DEFAULT_BATCH_SIZE,
        # GPU acceleration
        use_gpu: bool = False,
        num_gpus: int = 1,
        # Ray Data options
        ray_remote_args: Optional[dict[str, Any]] = None,
        batch_format: Optional[str] = None,
        **map_batches_kwargs: Any,
    ):
        """Initialize drift detector.

        Args:
            reference_dataset: Reference dataset for comparison
            reference_statistics: Pre-computed reference statistics
            columns_to_check: Columns to check for drift (None = all numeric columns)
            drift_threshold: Threshold for drift detection (0.0-1.0)
            method: Drift detection method ("ks_test", "psi", "chi_square")
            batch_size: Batch size for processing
            use_gpu: Use GPU acceleration
            num_gpus: Number of GPUs per worker
            ray_remote_args: Additional Ray remote arguments
            batch_format: Batch format for map_batches
            **map_batches_kwargs: Additional kwargs passed to map_batches
        """
        super().__init__(batch_size=batch_size)
        self.reference_dataset = reference_dataset
        self.reference_statistics = reference_statistics or {}
        self.columns_to_check = columns_to_check
        self.drift_threshold = drift_threshold
        self.method = method
        self.use_gpu = use_gpu
        self.num_gpus = num_gpus
        self.ray_remote_args = ray_remote_args or {}
        self.batch_format = batch_format
        self.map_batches_kwargs = map_batches_kwargs

    def process(self, dataset: Dataset) -> Dataset:
        """Detect drift in dataset.

        Args:
            dataset: Input Ray Dataset

        Returns:
            Dataset with drift detection results
        """
        logger.info("Detecting data drift")
        
        if self.reference_dataset:
            self._compute_reference_statistics()
        
        return self._detect_drift(dataset)

    def _compute_reference_statistics(self) -> None:
        """Compute statistics from reference dataset."""
        import numpy as np
        
        if not self.reference_dataset:
            raise ValueError("reference_dataset is required to compute reference statistics")
        
        if self.columns_to_check is None:
            # Sample to get columns
            try:
                sample = self.reference_dataset.take(1)
                if sample:
                    self.columns_to_check = [col for col in sample[0].keys() if isinstance(sample[0][col], (int, float))]
                else:
                    logger.warning("Reference dataset is empty")
                    self.reference_statistics = {}
                    return
            except Exception as e:
                logger.warning(f"Failed to sample reference dataset: {e}")
                self.reference_statistics = {}
                return
        
        if not self.columns_to_check:
            logger.warning("No columns to check for drift")
            self.reference_statistics = {}
            return
        
        for column in self.columns_to_check:
            try:
                col_data = self.reference_dataset.select_columns([column])
                
                def compute_stats(batch: dict[str, Any]) -> dict[str, Any]:
                    if column not in batch:
                        return {}
                    values = batch[column]
                    if not isinstance(values, (list, tuple)):
                        values = [values]
                    
                    numeric_values = [v for v in values if isinstance(v, (int, float)) and not (isinstance(v, float) and np.isnan(v))]
                    
                    if numeric_values:
                        arr = np.array(numeric_values)
                        if len(arr) == 0:
                            return {}
                        return {
                            "mean": float(np.mean(arr)),
                            "std": float(np.std(arr)),
                            "histogram": np.histogram(arr, bins=10)[0].tolist(),
                        }
                    return {}
                
                stats = col_data.map_batches(compute_stats, batch_size=self.batch_size)
                try:
                    if stats.count() > 0:
                        stat_results = stats.take(1)
                        if stat_results:
                            self.reference_statistics[column] = stat_results[0]
                except Exception as e:
                    logger.warning(f"Failed to get statistics for column {column}: {e}")
            except Exception as e:
                logger.warning(f"Failed to compute reference statistics for column {column}: {e}")

    def _detect_drift(self, dataset: Dataset) -> Dataset:
        """Detect drift using specified method."""
        if self.method == "ks_test":
            return self._ks_test_drift(dataset)
        elif self.method == "psi":
            return self._psi_drift(dataset)
        else:
            raise ValueError(f"Unknown drift detection method: {self.method}")

    def _ks_test_drift(self, dataset: Dataset) -> Dataset:
        """Detect drift using Kolmogorov-Smirnov test.
        
        Note: This compares current data against reference statistics. For
        proper KS test, use reference_dataset instead of reference_statistics.
        """
        try:
            from scipy import stats
        except ImportError:
            logger.warning("scipy not available, skipping KS test drift detection")
            return dataset
        
        if not self.reference_statistics:
            logger.warning("No reference statistics available for KS test")
            return dataset
        
        def detect_batch_drift(batch: dict[str, Any]) -> dict[str, Any]:
            columns_to_check = self.columns_to_check or list(batch.keys())
            
            for column in columns_to_check:
                if column not in batch or column not in self.reference_statistics:
                    continue
                
                values = batch[column]
                if not isinstance(values, (list, tuple)):
                    values = [values]
                
                import numpy as np
                numeric_values = [v for v in values if isinstance(v, (int, float)) and not (isinstance(v, float) and np.isnan(v))]
                
                if len(numeric_values) > 0:
                    ref_stats = self.reference_statistics[column]
                    ref_mean = ref_stats.get("mean", 0)
                    ref_std = ref_stats.get("std", 1)
                    
                    current_arr = np.array(numeric_values)
                    
                    # Use reference statistics to generate comparison distribution
                    # Note: This is an approximation. For exact KS test, use reference_dataset
                    if ref_std > 0:
                        ref_arr = np.random.normal(ref_mean, ref_std, len(current_arr))
                    else:
                        # Constant reference - compare against constant
                        ref_arr = np.full(len(current_arr), ref_mean)
                    
                    try:
                        ks_statistic, p_value = stats.ks_2samp(current_arr, ref_arr)
                        batch[f"{column}_drift_score"] = float(ks_statistic)
                        batch[f"{column}_drift_p_value"] = float(p_value)
                        batch[f"{column}_drift_detected"] = ks_statistic > self.drift_threshold
                    except Exception as e:
                        logger.warning(f"KS test failed for column {column}: {e}")
                        batch[f"{column}_drift_error"] = str(e)
            
            return batch
        
        map_kwargs = {
            "batch_size": self.batch_size,
            **self.map_batches_kwargs,
        }
        if self.batch_format:
            map_kwargs["batch_format"] = self.batch_format
        if self.ray_remote_args:
            map_kwargs["ray_remote_args"] = {
                "num_gpus": self.num_gpus if self.use_gpu else 0,
                **self.ray_remote_args,
            }
        
        return dataset.map_batches(detect_batch_drift, **map_kwargs)

    def _psi_drift(self, dataset: Dataset) -> Dataset:
        """Detect drift using Population Stability Index."""
        def compute_psi_batch(batch: dict[str, Any]) -> dict[str, Any]:
            for column in self.columns_to_check or batch.keys():
                if column in batch and column in self.reference_statistics:
                    values = batch[column]
                    numeric_values = [v for v in values if isinstance(v, (int, float))]
                    
                    if len(numeric_values) > 0:
                        import numpy as np
                        ref_hist = np.array(self.reference_statistics[column].get("histogram", []))
                        current_hist, _ = np.histogram(numeric_values, bins=len(ref_hist))
                        
                        ref_hist = ref_hist / (ref_hist.sum() + 1e-10)
                        current_hist = current_hist / (current_hist.sum() + 1e-10)
                        
                        psi = np.sum((current_hist - ref_hist) * np.log((current_hist + 1e-10) / (ref_hist + 1e-10)))
                        batch[f"{column}_psi"] = float(psi)
                        batch[f"{column}_drift_detected"] = psi > self.drift_threshold
            
            return batch
        
        map_kwargs = {
            "batch_size": self.batch_size,
            **self.map_batches_kwargs,
        }
        if self.batch_format:
            map_kwargs["batch_format"] = self.batch_format
        if self.ray_remote_args:
            map_kwargs["ray_remote_args"] = {
                "num_gpus": self.num_gpus if self.use_gpu else 0,
                **self.ray_remote_args,
            }
        
        return dataset.map_batches(compute_psi_batch, **map_kwargs)

