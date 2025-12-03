"""GPU-accelerated analytics stage using NVIDIA RAPIDS cuDF and cuML.

This stage demonstrates production-grade GPU-accelerated data processing
following best practices from NVIDIA's data infrastructure frameworks:

- NVIDIA NeMo Data Curator: Large-scale multimodal data curation
- NVIDIA Merlin NVTabular: GPU-accelerated feature engineering
- RAPIDS Accelerator: End-to-end GPU data science

Uses cuDF for GPU DataFrame operations and cuML for advanced analytics.

NVIDIA Libraries:
- cuDF: https://docs.rapids.ai/api/cudf/stable/ (GPU DataFrames)
- cuML: https://docs.rapids.ai/api/cuml/stable/ (GPU ML algorithms)
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Sequence

import pandas as pd  # Ray passes pandas batches by default

from pipeline.utils.constants import _GPU_BATCH_SIZE

logger = logging.getLogger(__name__)


class GPUAnalyticsStage:
    """Apply GPU-accelerated analytics (normalization + metrics) via cuDF.

    Args:
        target_columns: Columns to operate on.
        metrics: Metrics to compute per column (mean, std, min, max, median).
        normalize: Whether to append z-score normalized columns.
        num_gpus: GPUs to allocate per Ray task.

    Raises:
        ImportError: If cuDF is not available. We require RAPIDS rather
            than silently falling back to CPU to keep GPU semantics clear.
    """

    SUPPORTED_METRICS = {"mean", "std", "min", "max", "median"}

    def __init__(
        self,
        target_columns: Sequence[str],
        metrics: Optional[Sequence[str]] = None,
        normalize: bool = True,
        num_gpus: int = 1,
    ) -> None:
        if not target_columns:
            raise ValueError("GPUAnalyticsStage requires at least one target column")

        try:
            import cudf  # type: ignore[attr-defined]
        except ImportError as exc:
            raise ImportError(
                "GPUAnalyticsStage requires RAPIDS cuDF. "
                "Install via `pip install cudf-cu12` (matching your CUDA version)."
            ) from exc

        # Check cuDF compatibility
        from pipeline.utils.gpu.rapids import check_cudf_compatibility

        cudf_info = check_cudf_compatibility()
        if not cudf_info.get("available"):
            raise ImportError("cuDF not available or incompatible version")
        if cudf_info.get("issues"):
            for issue in cudf_info["issues"]:
                logger.warning(f"cuDF compatibility issue: {issue}")

        self._cudf = cudf

        # Ensure RMM is initialized for optimal cuDF performance
        # RMM pool should be initialized once at application startup
        # This is a safety check - RMM should already be initialized in core.py
        from pipeline.utils.gpu.rapids import initialize_rmm_pool

        rmm_initialized = initialize_rmm_pool()
        if not rmm_initialized:
            logger.warning("RMM pool not initialized, cuDF performance may be suboptimal")
        else:
            logger.debug("RMM pool verified/initialized for GPU analytics")
        self.target_columns = list(target_columns)
        self.metrics = list(metrics or ["mean", "std"])
        unsupported = set(self.metrics) - self.SUPPORTED_METRICS
        if unsupported:
            raise ValueError(
                f"Unsupported metrics {unsupported}. Supported: {self.SUPPORTED_METRICS}"
            )
        self.normalize = normalize
        self.num_gpus = num_gpus

    def process(self, dataset) -> Any:
        """Apply GPU analytics via map_batches."""

        def _gpu_transform(batch: pd.DataFrame) -> pd.DataFrame:
            from pipeline.utils.gpu.memory import check_gpu_memory, gpu_memory_cleanup

            if not isinstance(batch, pd.DataFrame):
                batch = pd.DataFrame(batch)
            
            # Skip empty batches early
            if len(batch) == 0:
                return batch

            # Estimate required GPU memory more accurately
            # Account for cuDF overhead and intermediate operations
            num_rows = len(batch)
            num_cols = len(batch.columns)
            # Estimate: rows * cols * 8 bytes (float64) + overhead for operations
            estimated_memory = num_rows * num_cols * 8 * 3  # 3x for cuDF operations overhead
            has_memory, mem_info = check_gpu_memory(estimated_memory)
            if not has_memory:
                raise RuntimeError(
                    f"Insufficient GPU memory for cuDF operations. "
                    f"Required: {estimated_memory}, Available: {mem_info['free']}"
                )

            # Use GPU memory cleanup context to ensure cuDF memory is freed
            with gpu_memory_cleanup():
                try:
                    # Convert to cuDF DataFrame
                    # Check dtype compatibility first
                    try:
                        # Try zero-copy conversion first (faster)
                        gdf = self._cudf.DataFrame.from_pandas(batch, allow_copy=False)
                    except (ValueError, TypeError) as e:
                        # Fallback to copy if dtype incompatibility
                        logger.debug(f"Zero-copy conversion failed, using copy: {e}")
                        # Convert problematic dtypes before conversion
                        batch_fixed = batch.copy()
                        for col in batch_fixed.columns:
                            dtype = batch_fixed[col].dtype
                            # cuDF doesn't support all pandas dtypes
                            if dtype == "object":
                                # Try to convert object columns to string
                                try:
                                    batch_fixed[col] = batch_fixed[col].astype("string")
                                except Exception:
                                    pass
                        gdf = self._cudf.DataFrame.from_pandas(batch_fixed, allow_copy=True)
                except RuntimeError as e:
                    if "out of memory" in str(e).lower() or "CUDA" in str(e):
                        logger.error(f"GPU memory error in cuDF conversion: {e}")
                        # Don't call rmm.reinitialize() here - it's expensive
                        # Instead, clear PyTorch cache and let RMM pool handle memory naturally.
                        # RMM pool should be initialized once at startup, not per-operation.
                        try:
                            import torch
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        except ImportError:
                            pass
                        raise RuntimeError(f"GPU OOM in cuDF operations: {e}") from e
                    raise

                try:
                    for column in self.target_columns:
                        if column not in gdf.columns:
                            logger.warning("Column %s missing in batch; skipping", column)
                            continue

                        stats: dict[str, Any] = {}
                        for metric in self.metrics:
                            # Use cuDF's native aggregation methods (GPU-accelerated)
                            col_series = gdf[column]
                            if metric == "mean":
                                value = col_series.mean()
                            elif metric == "std":
                                value = col_series.std()
                            elif metric == "min":
                                value = col_series.min()
                            elif metric == "max":
                                value = col_series.max()
                            elif metric == "median":
                                value = col_series.median()
                            else:
                                # Fallback to generic method
                                value = getattr(col_series, metric)()

                            # Convert cuDF scalar to Python value
                            if hasattr(value, "to_pandas"):
                                value = value.to_pandas()
                            elif hasattr(value, "item"):
                                value = value.item()
                            elif hasattr(value, "__float__"):
                                value = float(value)
                            stats[metric] = value

                            # Broadcast scalar to all rows using cuDF's efficient broadcasting
                            # cuDF handles broadcasting efficiently on GPU
                            gdf[f"{column}_{metric}_gpu"] = value

                        if self.normalize:
                            mean_val = stats.get("mean")
                            std_val = stats.get("std")
                            if mean_val is None:
                                mean_val = gdf[column].mean()
                                if hasattr(mean_val, "item"):
                                    mean_val = mean_val.item()
                            if std_val is None:
                                std_val = gdf[column].std()
                                if hasattr(std_val, "item"):
                                    std_val = std_val.item()
                            if std_val == 0 or std_val is None:
                                logger.warning("Std dev zero for column %s; skipping normalization", column)
                            else:
                                # Use cuDF's vectorized operations (GPU-accelerated)
                                # cuDF handles division and subtraction efficiently on GPU
                                col_data = gdf[column]
                                normalized = (col_data - mean_val) / std_val
                                gdf[f"{column}_zscore_gpu"] = normalized

                    # Convert back to pandas (this copies data to CPU)
                    # Use cuDF's efficient to_pandas() which handles dtype conversion
                    # Use zero-copy conversion when possible (faster)
                    try:
                        result = gdf.to_pandas(zero_copy_only=True)
                    except ValueError:
                        # Fallback to copy if zero-copy not possible
                        result = gdf.to_pandas()
                finally:
                    # Explicitly free cuDF DataFrame to release GPU memory
                    # cuDF DataFrames hold GPU memory, must be freed explicitly
                    del gdf
                    # Force GPU memory cleanup for large batches
                    if self.num_gpus > 0:
                        try:
                            import torch
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        except ImportError:
                            pass
                    # Don't reinitialize RMM pool here - too aggressive
                    # RMM pool should be managed at application level, not per-operation

                return result

        logger.info(
            "Running GPU analytics stage on columns %s with metrics %s",
            self.target_columns,
            self.metrics,
        )

        # Use map_batches with optimal settings for cuDF operations
        # zero_copy_batch=False ensures pandas DataFrame is created (needed for cuDF conversion)
        # batch_format="pandas" ensures we get pandas DataFrames (cuDF can convert efficiently)
        # Add memory limit to prevent GPU OOM
        ray_remote_args = {
            "num_gpus": self.num_gpus,
            "memory": 4 * 1024 * 1024 * 1024,  # 4GB memory limit per task
        }
        return dataset.map_batches(
            _gpu_transform,
            batch_format="pandas",
            ray_remote_args=ray_remote_args,
            zero_copy_batch=False,  # Must be False for cuDF conversion
            batch_size=_GPU_BATCH_SIZE,
            concurrency=min(4, self.num_gpus * 2) if self.num_gpus > 0 else None,  # Limit concurrency
        )

