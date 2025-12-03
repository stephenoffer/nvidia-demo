"""GPU-accelerated join operator for Ray Data.

Replaces PolarsJoin operator with GPU-accelerated cuDF join operations.
Provides significant speedup for large-scale join operations.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, List, Union

import pandas as pd

logger = logging.getLogger(__name__)

try:
    import cudf  # type: ignore[attr-defined]

    _CUDF_AVAILABLE = True
except ImportError:
    _CUDF_AVAILABLE = False


def gpu_join_operator(
    left_batch: pd.DataFrame,
    right_batch: pd.DataFrame,
    on: Optional[Union[str, List[str]]] = None,
    left_on: Optional[Union[str, List[str]]] = None,
    right_on: Optional[Union[str, List[str]]] = None,
    how: str = "inner",
    num_gpus: int = 1,
) -> pd.DataFrame:
    """GPU-accelerated join operator for batch-level joins.

    This function operates on individual batches only. It does not perform
    global joins across entire datasets. For global joins between datasets, always use
    Ray Data's native join() function:
    
    ```python
    # For global joins - use Ray Data native function (CORRECT)
    dataset1.join(dataset2, on="key", how="inner")
    
    # For GPU-accelerated global joins, consider creating a custom operator:
    # - Replace PolarsJoin with PolarsGPUJoin that uses GPU-based Polars
    # - Or use Ray Data's join() with GPU-accelerated underlying operators
    
    # This function is only for batch-level joins within map_batches()
    ```

    When calling this function via dataset.map_batches(), you must specify
    batch_format="pandas". Note: Ray Data's native join() is preferred for streaming
    operations as it handles streaming correctly without materialization.

    Uses cuDF for GPU-accelerated join operations, providing significant
    speedup over CPU-based joins for large datasets.

    Args:
        left_batch: Left DataFrame batch
        right_batch: Right DataFrame batch
        on: Column name(s) to join on
        left_on: Left DataFrame column(s) to join on
        right_on: Right DataFrame column(s) to join on
        how: Join type ("inner", "left", "right", "outer")
        num_gpus: Number of GPUs to use

    Returns:
        Joined DataFrame
    """
    # Validate num_gpus parameter
    if num_gpus < 0:
        raise ValueError(f"num_gpus must be >= 0, got {num_gpus}")

    # Validate join type
    valid_join_types = {"inner", "left", "right", "outer"}
    if how not in valid_join_types:
        raise ValueError(f"Invalid join type '{how}'. Must be one of {valid_join_types}")

    # Validate join keys are specified
    if on is None and left_on is None and right_on is None:
        raise ValueError("Must specify at least one of: 'on', 'left_on'/'right_on'")

    if not _CUDF_AVAILABLE or num_gpus == 0:
        # Fallback to pandas join
        return pd.merge(left_batch, right_batch, on=on, left_on=left_on, right_on=right_on, how=how)

    left_gdf = None
    right_gdf = None
    result_gdf = None
    try:
        # Try zero-copy conversion first for performance
        try:
            left_gdf = cudf.DataFrame.from_pandas(left_batch, allow_copy=False)
            right_gdf = cudf.DataFrame.from_pandas(right_batch, allow_copy=False)
        except (ValueError, TypeError):
            # Fallback to copy if dtype incompatibility
            left_gdf = cudf.DataFrame.from_pandas(left_batch, allow_copy=True)
            right_gdf = cudf.DataFrame.from_pandas(right_batch, allow_copy=True)

        # Validate join keys exist in DataFrames before merge
        if on is not None:
            if isinstance(on, str):
                on_list = [on]
            else:
                on_list = on
            missing_left = [col for col in on_list if col not in left_gdf.columns]
            missing_right = [col for col in on_list if col not in right_gdf.columns]
            if missing_left:
                raise ValueError(f"Join key(s) not found in left DataFrame: {missing_left}")
            if missing_right:
                raise ValueError(f"Join key(s) not found in right DataFrame: {missing_right}")
        elif left_on is not None or right_on is not None:
            # Validate left_on/right_on columns exist
            if left_on is not None:
                if isinstance(left_on, str):
                    left_on_list = [left_on]
                else:
                    left_on_list = left_on
                missing_left = [col for col in left_on_list if col not in left_gdf.columns]
                if missing_left:
                    raise ValueError(f"left_on key(s) not found in left DataFrame: {missing_left}")
            if right_on is not None:
                if isinstance(right_on, str):
                    right_on_list = [right_on]
                else:
                    right_on_list = right_on
                missing_right = [col for col in right_on_list if col not in right_gdf.columns]
                if missing_right:
                    raise ValueError(f"right_on key(s) not found in right DataFrame: {missing_right}")

        # Perform GPU-accelerated join
        result_gdf = left_gdf.merge(
            right_gdf,
            on=on,
            left_on=left_on,
            right_on=right_on,
            how=how,
        )

        # Convert back to pandas for Ray Data compatibility
        result_pd = result_gdf.to_pandas()
        
        # Validate pandas conversion succeeded
        if not isinstance(result_pd, pd.DataFrame):
            raise RuntimeError(f"cuDF to_pandas() returned non-DataFrame: {type(result_pd)}")
        
        # Explicitly free GPU memory
        del result_gdf
        del left_gdf
        del right_gdf
        
        return result_pd
    except (ValueError, TypeError, RuntimeError) as e:
        logger.warning(f"GPU join failed, falling back to CPU: {e}")
        # Cleanup on error
        if result_gdf is not None:
            del result_gdf
        if left_gdf is not None:
            del left_gdf
        if right_gdf is not None:
            del right_gdf
        return pd.merge(left_batch, right_batch, on=on, left_on=left_on, right_on=right_on, how=how)
    finally:
        # Final cleanup
        if result_gdf is not None:
            del result_gdf
        if left_gdf is not None:
            del left_gdf
        if right_gdf is not None:
            del right_gdf


def gpu_union_operator(
    batches: List[pd.DataFrame],
    num_gpus: int = 1,
) -> pd.DataFrame:
    """GPU-accelerated union operator.

    Args:
        batches: List of DataFrames to union
        num_gpus: Number of GPUs to use

    Returns:
        Unioned DataFrame
    """
    from pipeline.utils.gpu.etl import gpu_concat

    return gpu_concat(batches, axis=0, num_gpus=num_gpus)

