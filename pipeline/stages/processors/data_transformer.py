"""Data transformation stage for ETL operations.

Provides common data transformations like filtering, mapping, aggregations,
and joins for multimodal robotics data. Optionally uses GPU acceleration.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from ray.data import Dataset

from pipeline.stages.base import ProcessorBase
from pipeline.utils.constants import _DEFAULT_BATCH_SIZE

logger = logging.getLogger(__name__)


class DataTransformer(ProcessorBase):
    """Transform data using configurable transformation functions.

    Example:
        ```python
        # Simple transformation
        transformer = DataTransformer(
            transform_func=lambda x: {**x, "processed": True},
        )
        
        # With filtering and GPU acceleration
        transformer = DataTransformer(
            transform_func=lambda x: {**x, "processed": True},
            filter_func=lambda x: x.get("quality", 0) > 0.8,
            use_gpu=True,
            ray_remote_args={"num_gpus": 1},
        )
        ```
    """

    def __init__(
        self,
        transform_func: Callable[[dict[str, Any]], dict[str, Any]],
        filter_func: Optional[Callable[[dict[str, Any]], bool]] = None,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        # GPU acceleration
        use_gpu: bool = False,
        num_gpus: int = 1,
        # Ray Data options
        ray_remote_args: Optional[dict[str, Any]] = None,
        batch_format: Optional[str] = None,
        **map_batches_kwargs: Any,
    ):
        """Initialize data transformer.

        Args:
            transform_func: Function to transform each item
            filter_func: Optional function to filter items (None = keep all)
            batch_size: Batch size for processing
            use_gpu: Use GPU acceleration with cuDF
            num_gpus: Number of GPUs per worker
            ray_remote_args: Additional Ray remote arguments
            batch_format: Batch format for map_batches
            **map_batches_kwargs: Additional kwargs passed to map_batches
        """
        super().__init__(batch_size=batch_size)
        self.transform_func = transform_func
        self.filter_func = filter_func
        self.use_gpu = use_gpu
        self.num_gpus = num_gpus
        self.ray_remote_args = ray_remote_args or {}
        self.batch_format = batch_format
        self.map_batches_kwargs = map_batches_kwargs

    def process(self, dataset: Dataset) -> Dataset:
        """Transform dataset.

        Args:
            dataset: Input Ray Dataset

        Returns:
            Transformed dataset
        """
        logger.info("Transforming dataset")
        
        def transform_batch(batch: dict[str, Any]) -> dict[str, Any]:
            """Transform batch."""
            items = self._batch_to_items(batch)
            
            transformed_items = []
            for item in items:
                if self.filter_func is None or self.filter_func(item):
                    transformed_item = self.transform_func(item)
                    transformed_items.append(transformed_item)
            
            if not transformed_items:
                return {}
            
            return self._items_to_batch(transformed_items)
        
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
        
        return dataset.map_batches(transform_batch, **map_kwargs)

    def _batch_to_items(self, batch: dict[str, Any]) -> list[dict[str, Any]]:
        """Convert batch dict to list of items."""
        if not batch:
            return []
        
        keys = list(batch.keys())
        num_items = len(batch[keys[0]]) if keys else 0
        
        return [
            {key: batch[key][i] for key in keys}
            for i in range(num_items)
        ]

    def _items_to_batch(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        """Convert list of items to batch dict."""
        if not items:
            return {}
        
        keys = items[0].keys()
        batch = {key: [item[key] for item in items] for key in keys}
        return batch


class DataAggregator(ProcessorBase):
    """Aggregate data using groupby operations.

    Example:
        ```python
        # Simple aggregation
        aggregator = DataAggregator(
            groupby_columns=["episode_id"],
            agg_functions={"sensor_data": "mean", "timestamp": "max"},
        )
        
        # GPU-accelerated aggregation
        aggregator = DataAggregator(
            groupby_columns=["episode_id"],
            agg_functions={"sensor_data": "mean"},
            use_gpu=True,
        )
        ```
    """

    def __init__(
        self,
        groupby_columns: list[str],
        agg_functions: dict[str, str],
        batch_size: int = _DEFAULT_BATCH_SIZE,
        # GPU acceleration
        use_gpu: bool = False,
        num_gpus: int = 1,
        # Ray Data options
        ray_remote_args: Optional[dict[str, Any]] = None,
        **groupby_kwargs: Any,
    ):
        """Initialize data aggregator.

        Args:
            groupby_columns: Columns to group by
            agg_functions: Dictionary mapping column names to aggregation functions
            batch_size: Batch size for processing
            use_gpu: Use GPU acceleration with cuDF
            num_gpus: Number of GPUs per worker
            ray_remote_args: Additional Ray remote arguments
            **groupby_kwargs: Additional kwargs passed to groupby
        """
        super().__init__(batch_size=batch_size)
        self.groupby_columns = groupby_columns
        self.agg_functions = agg_functions
        self.use_gpu = use_gpu
        self.num_gpus = num_gpus
        self.ray_remote_args = ray_remote_args or {}
        self.groupby_kwargs = groupby_kwargs

    def process(self, dataset: Dataset) -> Dataset:
        """Aggregate dataset.

        Args:
            dataset: Input Ray Dataset

        Returns:
            Aggregated dataset
        """
        logger.info(f"Aggregating by {self.groupby_columns}")
        
        if self.use_gpu:
            return self._aggregate_gpu(dataset)
        else:
            grouped = dataset.groupby(self.groupby_columns, **self.groupby_kwargs)
            return grouped.aggregate(self.agg_functions)

    def _aggregate_gpu(self, dataset: Dataset) -> Dataset:
        """Aggregate using GPU-accelerated cuDF."""
        try:
            import cudf
            import pandas as pd
            
            def aggregate_batch_gpu(batch: pd.DataFrame) -> pd.DataFrame:
                gdf = cudf.from_pandas(batch)
                grouped = gdf.groupby(self.groupby_columns)
                aggregated = grouped.agg(self.agg_functions)
                return aggregated.to_pandas()
            
            map_kwargs = {
                "batch_size": self.batch_size,
                "batch_format": "pandas",
                "ray_remote_args": {
                    "num_gpus": self.num_gpus,
                    **self.ray_remote_args,
                },
            }
            
            # GPU aggregation is batch-level, then need global aggregation
            # Note: This is a two-stage process - batch GPU agg, then global CPU agg
            gpu_agg_result = dataset.map_batches(aggregate_batch_gpu, **map_kwargs)
            grouped = gpu_agg_result.groupby(self.groupby_columns, **self.groupby_kwargs)
            return grouped.aggregate(self.agg_functions)
        except ImportError:
            logger.warning("cuDF not available, falling back to CPU aggregation")
            grouped = dataset.groupby(self.groupby_columns, **self.groupby_kwargs)
            return grouped.aggregate(self.agg_functions)
        except Exception as e:
            logger.warning(f"GPU aggregation failed: {e}, falling back to CPU")
            grouped = dataset.groupby(self.groupby_columns, **self.groupby_kwargs)
            return grouped.aggregate(self.agg_functions)

