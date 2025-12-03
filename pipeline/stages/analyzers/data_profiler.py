"""Data profiling stage for comprehensive data quality analysis.

Generates statistics, detects anomalies, and provides data quality reports.
Optionally uses NVIDIA cuDF for GPU-accelerated profiling.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from ray.data import Dataset

from pipeline.stages.base import ProcessorBase
from pipeline.utils.constants import _DEFAULT_BATCH_SIZE

logger = logging.getLogger(__name__)


class DataProfiler(ProcessorBase):
    """Profile data and generate quality statistics.

    Computes statistics, detects outliers, identifies missing values,
    and generates data quality reports. Optionally uses cuDF for GPU acceleration.

    Example:
        ```python
        # Simple usage
        profiler = DataProfiler(profile_columns=["image", "sensor_data"])
        
        # GPU-accelerated profiling
        profiler = DataProfiler(
            profile_columns=["image", "sensor_data"],
            use_gpu=True,
            ray_remote_args={"num_gpus": 1},
        )
        ```
    """

    def __init__(
        self,
        profile_columns: Optional[list[str]] = None,
        compute_statistics: bool = True,
        detect_outliers: bool = True,
        detect_missing: bool = True,
        output_path: Optional[str] = None,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        # GPU acceleration
        use_gpu: bool = False,
        num_gpus: int = 1,
        # Ray Data options
        ray_remote_args: Optional[dict[str, Any]] = None,
        batch_format: Optional[str] = None,
        **map_batches_kwargs: Any,
    ):
        """Initialize data profiler.

        Args:
            profile_columns: Columns to profile (None = all columns)
            compute_statistics: Whether to compute statistical summaries
            detect_outliers: Whether to detect outliers
            detect_missing: Whether to detect missing values
            output_path: Path to save profiling report
            batch_size: Batch size for processing
            use_gpu: Use GPU acceleration with cuDF
            num_gpus: Number of GPUs per worker
            ray_remote_args: Additional Ray remote arguments
            batch_format: Batch format for map_batches
            **map_batches_kwargs: Additional kwargs passed to map_batches
        """
        super().__init__(batch_size=batch_size)
        self.profile_columns = profile_columns
        self.compute_statistics = compute_statistics
        self.detect_outliers = detect_outliers
        self.detect_missing = detect_missing
        self.output_path = output_path
        self.use_gpu = use_gpu
        self.num_gpus = num_gpus
        self.ray_remote_args = ray_remote_args or {}
        self.batch_format = batch_format
        self.map_batches_kwargs = map_batches_kwargs
        self._statistics = {}

    def process(self, dataset: Dataset) -> Dataset:
        """Profile dataset and generate statistics.

        Args:
            dataset: Input Ray Dataset

        Returns:
            Dataset with profiling metadata added
        """
        logger.info("Profiling dataset")
        
        if self.compute_statistics:
            self._compute_statistics(dataset)
        
        if self.detect_outliers:
            dataset = self._detect_outliers(dataset)
        
        if self.detect_missing:
            dataset = self._detect_missing(dataset)
        
        if self.output_path:
            self._save_report()
        
        return dataset

    def _compute_statistics(self, dataset: Dataset) -> None:
        """Compute statistical summaries."""
        if self.use_gpu:
            self._compute_statistics_gpu(dataset)
        else:
            self._compute_statistics_cpu(dataset)

    def _compute_statistics_cpu(self, dataset: Dataset) -> None:
        """Compute statistics on CPU."""
        import numpy as np
        
        stats = {}
        
        try:
            sample_batch = next(dataset.iter_batches(batch_size=100), None)
        except StopIteration:
            sample_batch = None
        
        if not sample_batch:
            logger.warning("No data available for profiling")
            self._statistics = {}
            return
        
        columns_to_profile = self.profile_columns or list(sample_batch.keys())
        
        if not columns_to_profile:
            logger.warning("No columns to profile")
            self._statistics = {}
            return
        
        for column in columns_to_profile:
            if column not in sample_batch:
                logger.debug(f"Column {column} not found in sample batch")
                continue
            
            try:
                values = sample_batch[column]
                if not isinstance(values, (list, tuple)):
                    values = [values]
                
                numeric_values = [v for v in values if isinstance(v, (int, float)) and not (isinstance(v, float) and np.isnan(v))]
                
                if numeric_values:
                    arr = np.array(numeric_values)
                    if len(arr) == 0:
                        stats[column] = {"count": 0, "type": "empty"}
                    else:
                        stats[column] = {
                            "mean": float(np.mean(arr)),
                            "std": float(np.std(arr)),
                            "min": float(np.min(arr)),
                            "max": float(np.max(arr)),
                            "median": float(np.median(arr)),
                            "count": len(numeric_values),
                        }
                else:
                    stats[column] = {"count": len(values), "type": "non_numeric"}
            except Exception as e:
                logger.warning(f"Failed to compute statistics for column {column}: {e}")
                stats[column] = {"error": str(e)}
        
        self._statistics = stats
        logger.info(f"Computed statistics for {len(stats)} columns")

    def _compute_statistics_gpu(self, dataset: Dataset) -> None:
        """Compute statistics on GPU using cuDF."""
        try:
            import cudf
            import pandas as pd
            
            stats = {}
            
            sample_batch = next(dataset.iter_batches(batch_size=100, batch_format="pandas"), None)
            if not sample_batch:
                logger.warning("No data available for profiling")
                return
            
            columns_to_profile = self.profile_columns or list(sample_batch.columns)
            
            gdf = cudf.from_pandas(sample_batch)
            
            for column in columns_to_profile:
                if column not in gdf.columns:
                    continue
                
                if gdf[column].dtype in ['int64', 'float64', 'int32', 'float32']:
                    stats[column] = {
                        "mean": float(gdf[column].mean()),
                        "std": float(gdf[column].std()),
                        "min": float(gdf[column].min()),
                        "max": float(gdf[column].max()),
                        "median": float(gdf[column].median()),
                        "count": int(gdf[column].count()),
                    }
                else:
                    stats[column] = {"count": len(gdf[column]), "type": "non_numeric"}
            
            self._statistics = stats
            logger.info(f"Computed GPU-accelerated statistics for {len(stats)} columns")
        except ImportError:
            logger.warning("cuDF not available, falling back to CPU profiling")
            self._compute_statistics_cpu(dataset)
        except Exception as e:
            logger.warning(f"GPU profiling failed: {e}, falling back to CPU")
            self._compute_statistics_cpu(dataset)

    def _detect_outliers(self, dataset: Dataset) -> Dataset:
        """Detect outliers using IQR method."""
        if self.use_gpu:
            return self._detect_outliers_gpu(dataset)
        else:
            return self._detect_outliers_cpu(dataset)

    def _detect_outliers_cpu(self, dataset: Dataset) -> Dataset:
        """Detect outliers on CPU."""
        import numpy as np
        
        def detect_batch_outliers(batch: dict[str, Any]) -> dict[str, Any]:
            columns_to_check = self.profile_columns or batch.keys()
            for column in columns_to_check:
                if column in batch:
                    values = batch[column]
                    if not isinstance(values, list):
                        continue
                    
                    numeric_values = [v for v in values if isinstance(v, (int, float))]
                    
                    if len(numeric_values) > 4:
                        arr = np.array(numeric_values)
                        q1, q3 = np.percentile(arr, [25, 75])
                        iqr = q3 - q1
                        if iqr > 0:
                            lower_bound = q1 - 1.5 * iqr
                            upper_bound = q3 + 1.5 * iqr
                            
                            outliers = [
                                i for i, v in enumerate(values)
                                if isinstance(v, (int, float)) and (v < lower_bound or v > upper_bound)
                            ]
                            batch[f"{column}_outliers"] = outliers
                            batch[f"{column}_outlier_count"] = len(outliers)
            
            return batch
        
        map_kwargs = {
            "batch_size": self.batch_size,
            **self.map_batches_kwargs,
        }
        if self.batch_format:
            map_kwargs["batch_format"] = self.batch_format
        if self.ray_remote_args:
            map_kwargs["ray_remote_args"] = self.ray_remote_args
        
        return dataset.map_batches(detect_batch_outliers, **map_kwargs)

    def _detect_outliers_gpu(self, dataset: Dataset) -> Dataset:
        """Detect outliers on GPU using cuDF."""
        try:
            import cudf
            import pandas as pd
            
            def detect_batch_outliers_gpu(batch: pd.DataFrame) -> pd.DataFrame:
                gdf = cudf.from_pandas(batch)
                columns_to_check = self.profile_columns or list(gdf.columns)
                
                for column in columns_to_check:
                    if column not in gdf.columns:
                        continue
                    
                    if gdf[column].dtype in ['int64', 'float64', 'int32', 'float32']:
                        q1 = gdf[column].quantile(0.25)
                        q3 = gdf[column].quantile(0.75)
                        iqr = q3 - q1
                        
                        if iqr > 0:
                            lower_bound = q1 - 1.5 * iqr
                            upper_bound = q3 + 1.5 * iqr
                            
                            outlier_mask = (gdf[column] < lower_bound) | (gdf[column] > upper_bound)
                            batch[f"{column}_outlier_count"] = outlier_mask.sum()
                
                return gdf.to_pandas()
            
            map_kwargs = {
                "batch_size": self.batch_size,
                "batch_format": "pandas",
                "ray_remote_args": {
                    "num_gpus": self.num_gpus,
                    **self.ray_remote_args,
                },
                **self.map_batches_kwargs,
            }
            
            return dataset.map_batches(detect_batch_outliers_gpu, **map_kwargs)
        except ImportError:
            logger.warning("cuDF not available, falling back to CPU outlier detection")
            return self._detect_outliers_cpu(dataset)
        except Exception as e:
            logger.warning(f"GPU outlier detection failed: {e}, falling back to CPU")
            return self._detect_outliers_cpu(dataset)

    def _detect_missing(self, dataset: Dataset) -> Dataset:
        """Detect missing values."""
        import numpy as np
        
        def detect_missing_batch(batch: dict[str, Any]) -> dict[str, Any]:
            columns_to_check = self.profile_columns or batch.keys()
            for column in columns_to_check:
                if column in batch:
                    values = batch[column]
                    if isinstance(values, list):
                        missing_count = sum(1 for v in values if v is None or (isinstance(v, float) and np.isnan(v)))
                        batch[f"{column}_missing_count"] = missing_count
                        batch[f"{column}_missing_rate"] = missing_count / len(values) if values else 0.0
            
            return batch
        
        map_kwargs = {
            "batch_size": self.batch_size,
            **self.map_batches_kwargs,
        }
        if self.batch_format:
            map_kwargs["batch_format"] = self.batch_format
        if self.ray_remote_args:
            map_kwargs["ray_remote_args"] = self.ray_remote_args
        
        return dataset.map_batches(detect_missing_batch, **map_kwargs)

    def _save_report(self) -> None:
        """Save profiling report to file."""
        import json
        
        report = {
            "statistics": self._statistics,
            "columns_profiled": self.profile_columns,
        }
        
        with open(self.output_path, "w") as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Saved profiling report to {self.output_path}")

