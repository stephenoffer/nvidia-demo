"""GPU-accelerated batch processing utilities.

Provides GPU-accelerated replacements for common batch processing patterns:
- Aggregations
- Transformations
- Filtering
- Sorting
- Groupby operations

Uses cuDF for DataFrame operations and cuPy for array operations.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional, Union, List, Dict

import pandas as pd

logger = logging.getLogger(__name__)

try:
    import cudf  # type: ignore[attr-defined]

    _CUDF_AVAILABLE = True
except ImportError:
    _CUDF_AVAILABLE = False


def gpu_batch_aggregate(
    batch: pd.DataFrame,
    agg_func: Callable[[Any], Any],
    groupby: Optional[Union[str, List[str]]] = None,
    num_gpus: int = 1,
) -> pd.DataFrame:
    """GPU-accelerated batch aggregation.

    This function operates on individual batches only. It does not perform
    global aggregations across the entire dataset. For global aggregations that require
    seeing all data, use Ray Data's native aggregate() function with AggregateFnV2:
    
    ```python
    # For global aggregations - use Ray Data native function
    dataset.aggregate(AggregateFnV2(...))  # Use AggregateFnV2 for GPU acceleration
    
    # For batch-level aggregations - use this function
    dataset.map_batches(
        lambda batch: gpu_batch_aggregate(batch, agg_func=sum, num_gpus=1),
        batch_format="pandas",  # REQUIRED: Must specify pandas format
        batch_size=1000,
    )
    ```

    When calling this function via dataset.map_batches(), you must specify
    batch_format="pandas".

    Args:
        batch: Pandas DataFrame batch (must be pandas, not cuDF)
        agg_func: Aggregation function
        groupby: Column(s) to group by (only groups within the batch, not globally)
        num_gpus: Number of GPUs to use

    Returns:
        Aggregated pandas DataFrame (ALWAYS pandas, never cuDF)
    """
    from pipeline.utils.gpu.etl import gpu_aggregate

    if groupby:
        agg_dict = {col: agg_func.__name__ if hasattr(agg_func, "__name__") else "sum" for col in batch.columns if col != groupby}
    else:
        agg_dict = {col: agg_func.__name__ if hasattr(agg_func, "__name__") else "sum" for col in batch.columns}

    return gpu_aggregate(batch, agg_dict=agg_dict, groupby=groupby, num_gpus=num_gpus)


def gpu_batch_transform(
    batch: pd.DataFrame,
    transform_func: Callable[[Any], Any],
    num_gpus: int = 1,
) -> pd.DataFrame:
    """GPU-accelerated batch transformation.

    When calling this function via dataset.map_batches(), you must specify
    batch_format="pandas":
    
    ```python
    dataset.map_batches(
        lambda batch: gpu_batch_transform(batch, transform_func, num_gpus=1),
        batch_format="pandas",  # REQUIRED: Must specify pandas format
        batch_size=1000,
    )
    ```

    Args:
        batch: Pandas DataFrame batch (must be pandas, not cuDF)
        transform_func: Transformation function
        num_gpus: Number of GPUs to use

    Returns:
        Transformed pandas DataFrame (ALWAYS pandas, never cuDF)
    """
    from pipeline.utils.gpu.etl import gpu_map_batches_transform

    return gpu_map_batches_transform(batch, transform_func, num_gpus=num_gpus)


def gpu_batch_sort(
    batch: pd.DataFrame,
    by: Union[str, List[str]],
    ascending: Union[bool, List[bool]] = True,
    num_gpus: int = 1,
) -> pd.DataFrame:
    """GPU-accelerated batch sorting.

    When calling this function via dataset.map_batches(), you must specify
    batch_format="pandas":
    
    ```python
    dataset.map_batches(
        lambda batch: gpu_batch_sort(batch, by=["value"], num_gpus=1),
        batch_format="pandas",  # REQUIRED: Must specify pandas format
        batch_size=1000,
    )
    ```

    Args:
        batch: Pandas DataFrame batch (must be pandas, not cuDF)
        by: Column(s) to sort by
        ascending: Sort order
        num_gpus: Number of GPUs to use

    Returns:
        Sorted pandas DataFrame (ALWAYS pandas, never cuDF)
    """
    from pipeline.utils.gpu.etl import gpu_sort

    return gpu_sort(batch, by=by, ascending=ascending, num_gpus=num_gpus)


def gpu_batch_groupby(
    batch: pd.DataFrame,
    by: Union[str, List[str]],
    agg_dict: Dict[str, Union[str, List[str]]],
    num_gpus: int = 1,
) -> pd.DataFrame:
    """GPU-accelerated batch groupby operation.

    This function operates on individual batches only. It does not perform
    global groupby operations across the entire dataset. For global groupby that requires
    seeing all data, use Ray Data's native groupby() function instead:
    
    ```python
    # For global groupby - use Ray Data native function
    dataset.groupby("key").agg({"value": "sum"})  # Correct for global groupby
    
    # For batch-level groupby - use this function
    dataset.map_batches(
        lambda batch: gpu_batch_groupby(batch, by=["group"], agg_dict={"value": "sum"}, num_gpus=1),
        batch_format="pandas",  # REQUIRED: Must specify pandas format
        batch_size=1000,
    )
    ```

    When calling this function via dataset.map_batches(), you must specify
    batch_format="pandas".

    Args:
        batch: Pandas DataFrame batch (must be pandas, not cuDF)
        by: Column(s) to group by (only groups within the batch, not globally)
        agg_dict: Aggregation dictionary
        num_gpus: Number of GPUs to use

    Returns:
        Grouped and aggregated pandas DataFrame (ALWAYS pandas, never cuDF)
    """
    from pipeline.utils.gpu.etl import gpu_aggregate

    return gpu_aggregate(batch, agg_dict=agg_dict, groupby=by, num_gpus=num_gpus)

