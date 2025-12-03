"""GPU-accelerated ETL operations using cuDF and cuPy.

Provides GPU-accelerated replacements for common ETL operations:
- Joins/merges (replacing PolarsJoin)
- Aggregations
- Filtering
- Sorting
- Groupby operations
- String operations
- Window functions

Uses RAPIDS cuDF for DataFrame operations and cuPy for array operations.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional, Union, List, Dict

import pandas as pd

logger = logging.getLogger(__name__)

# Try to import cuDF for GPU DataFrame operations
try:
    import cudf  # type: ignore[attr-defined]

    _CUDF_AVAILABLE = True
except ImportError:
    _CUDF_AVAILABLE = False
    logger.warning("cuDF not available - GPU ETL operations will fallback to CPU")


def gpu_map_batches_transform(
    batch: pd.DataFrame,
    transform_func: Callable[[Any], Any],
    num_gpus: int = 1,
) -> pd.DataFrame:
    """Transform batch using GPU-accelerated cuDF operations.

    This function must return pandas DataFrame for Ray Data streaming compatibility.
    cuDF DataFrames cannot be serialized by Ray and would break streaming execution.

    When calling this function via dataset.map_batches(), you must specify
    batch_format="pandas" to ensure Ray Data passes pandas DataFrames:
    
    ```python
    dataset.map_batches(
        lambda batch: gpu_map_batches_transform(batch, transform_func, num_gpus=1),
        batch_format="pandas",  # REQUIRED: Must specify pandas format
        batch_size=1000,
    )
    ```

    Args:
        batch: Pandas DataFrame batch (must be pandas, not cuDF)
        transform_func: Function that takes cuDF DataFrame and returns cuDF DataFrame
        num_gpus: Number of GPUs to use

    Returns:
        Transformed pandas DataFrame (ALWAYS pandas, never cuDF)
    
    Raises:
        RuntimeError: If transform_func returns None or invalid type
        ValueError: If batch is empty or invalid
    """
    if not isinstance(batch, pd.DataFrame):
        raise ValueError(f"Expected pandas DataFrame, got {type(batch)}")

    if batch.empty:
        logger.warning("Empty batch passed to gpu_map_batches_transform")
        return batch

    # Validate num_gpus parameter
    if num_gpus < 0:
        raise ValueError(f"num_gpus must be >= 0, got {num_gpus}")

    if not _CUDF_AVAILABLE or num_gpus == 0:
        # Fallback to CPU pandas operations
        result = transform_func(batch)
        if not isinstance(result, pd.DataFrame):
            raise RuntimeError(f"transform_func must return pandas DataFrame, got {type(result)}")
        return result

    gdf = None
    result_gdf = None
    try:
        # Try zero-copy conversion first for performance
        # Fallback to copy if zero-copy fails
        try:
            gdf = cudf.DataFrame.from_pandas(batch, allow_copy=False)  # Try zero-copy first
        except (ValueError, TypeError):
            # Fallback to copy if dtype incompatibility
            gdf = cudf.DataFrame.from_pandas(batch, allow_copy=True)

        # Apply GPU transformation
        result_gdf = transform_func(gdf)
        
        # Validate result is cuDF DataFrame
        if result_gdf is None:
            raise RuntimeError("transform_func returned None - must return cuDF DataFrame")
        if not hasattr(result_gdf, 'to_pandas'):
            raise RuntimeError(f"transform_func must return cuDF DataFrame, got {type(result_gdf)}")

        # Convert back to pandas before returning
        # Ray Data cannot serialize cuDF DataFrames - this breaks streaming
        result_pd = result_gdf.to_pandas()
        
        # Validate pandas conversion succeeded
        if not isinstance(result_pd, pd.DataFrame):
            raise RuntimeError(f"cuDF to_pandas() returned non-DataFrame: {type(result_pd)}")
        
        # Explicitly free GPU memory
        del result_gdf
        del gdf
        
        # Don't call rmm.reinitialize() here - it's expensive
        # RMM pool should be initialized once at application startup
        # Calling reinitialize() destroys the memory pool and causes severe performance degradation.
        # The explicit del statements above are sufficient for memory cleanup.
        
        return result_pd
    except (ValueError, TypeError, RuntimeError) as e:
        logger.warning(f"GPU transformation failed, falling back to CPU: {e}")
        # Ensure cleanup on error
        if result_gdf is not None:
            del result_gdf
        if gdf is not None:
            del gdf
        # CPU fallback must also validate return type
        try:
            cpu_result = transform_func(batch)
            if not isinstance(cpu_result, pd.DataFrame):
                raise RuntimeError(
                    f"CPU fallback transform_func must return pandas DataFrame, got {type(cpu_result)}"
                )
            return cpu_result
        except Exception as cpu_error:
            # If CPU fallback also fails, raise with context
            raise RuntimeError(
                f"Both GPU and CPU transformations failed. GPU error: {e}, CPU error: {cpu_error}"
            ) from cpu_error
    finally:
        # Final cleanup - ensure GPU memory is freed
        if result_gdf is not None:
            del result_gdf
        if gdf is not None:
            del gdf


def gpu_join(
    left: pd.DataFrame,
    right: pd.DataFrame,
    on: Optional[Union[str, List[str]]] = None,
    left_on: Optional[Union[str, List[str]]] = None,
    right_on: Optional[Union[str, List[str]]] = None,
    how: str = "inner",
    num_gpus: int = 1,
) -> pd.DataFrame:
    """GPU-accelerated join operation using cuDF for batch-level joins.

    This function operates on individual batches only. It does not perform
    global joins across entire datasets. For global joins between Ray Data datasets,
    always use Ray Data's native join() function:
    
    ```python
    # For global joins - use Ray Data native function (CORRECT)
    dataset1.join(dataset2, on="key", how="inner")
    
    # For GPU-accelerated global joins, consider:
    # - Creating a custom PolarsGPUJoin operator that replaces PolarsJoin
    # - Using GPU-accelerated Polars with cuDF backend
    # - Using Ray Data's join() with GPU-accelerated underlying operators
    
    # This function is only for batch-level joins within map_batches()
    ```

    Valid join types are "inner", "left", "right", "outer".

    Args:
        left: Left DataFrame batch
        right: Right DataFrame batch
        on: Column name(s) to join on
        left_on: Left DataFrame column(s) to join on
        right_on: Right DataFrame column(s) to join on
        how: Join type ("inner", "left", "right", "outer")
        num_gpus: Number of GPUs to use

    Returns:
        Joined DataFrame
    
    Raises:
        ValueError: If join keys are invalid or DataFrames are empty
    """
    # Validate inputs
    if not isinstance(left, pd.DataFrame) or not isinstance(right, pd.DataFrame):
        raise ValueError(f"Both inputs must be pandas DataFrames, got {type(left)} and {type(right)}")
    if left.empty or right.empty:
        logger.warning("One or both DataFrames are empty, join may return empty result")

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
        return pd.merge(left, right, on=on, left_on=left_on, right_on=right_on, how=how)

    left_gdf = None
    right_gdf = None
    result_gdf = None
    try:
        # Try zero-copy conversion first for performance
        try:
            left_gdf = cudf.DataFrame.from_pandas(left, allow_copy=False)
            right_gdf = cudf.DataFrame.from_pandas(right, allow_copy=False)
        except (ValueError, TypeError):
            # Fallback to copy if dtype incompatibility
            left_gdf = cudf.DataFrame.from_pandas(left, allow_copy=True)
            right_gdf = cudf.DataFrame.from_pandas(right, allow_copy=True)

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
        return pd.merge(left, right, on=on, left_on=left_on, right_on=right_on, how=how)
    finally:
        # Final cleanup
        if result_gdf is not None:
            del result_gdf
        if left_gdf is not None:
            del left_gdf
        if right_gdf is not None:
            del right_gdf


def gpu_aggregate(
    batch: pd.DataFrame,
    agg_dict: Dict[str, Union[str, List[str]]],
    groupby: Optional[Union[str, List[str]]] = None,
    num_gpus: int = 1,
) -> pd.DataFrame:
    """GPU-accelerated aggregation using cuDF.

    This function operates on individual batches only. It does not perform
    global aggregations across the entire dataset. For global aggregations that require
    seeing all data (e.g., global groupby, global sum/min/max), use Ray Data's native
    functions instead:
    
    ```python
    # For global aggregations - use Ray Data native functions
    dataset.groupby("key").agg({"value": "sum"})  # Correct for global groupby
    dataset.aggregate(AggregateFnV2(...))  # Use AggregateFnV2 for GPU acceleration
    
    # For batch-level aggregations - use this function
    dataset.map_batches(
        lambda batch: gpu_aggregate(batch, agg_dict={"value": "sum"}, num_gpus=1),
        batch_format="pandas",  # REQUIRED: Must specify pandas format
        batch_size=1000,
    )
    ```

    When calling this function via dataset.map_batches(), you must specify
    batch_format="pandas".

    Args:
        batch: Pandas DataFrame batch (must be pandas, not cuDF)
        agg_dict: Dictionary mapping columns to aggregation functions
        groupby: Column(s) to group by (None = aggregate entire DataFrame)
            Note: This only groups within the batch, not globally across the dataset
        num_gpus: Number of GPUs to use

    Returns:
        Aggregated pandas DataFrame (ALWAYS pandas, never cuDF)
    """
    # Validate num_gpus parameter
    if num_gpus < 0:
        raise ValueError(f"num_gpus must be >= 0, got {num_gpus}")

    # Validate aggregation columns exist
    missing_cols = [col for col in agg_dict.keys() if col not in batch.columns]
    if missing_cols:
        raise ValueError(f"Aggregation columns not found in DataFrame: {missing_cols}. Available: {list(batch.columns)}")

    # Validate groupby columns exist if specified
    if groupby is not None:
        if isinstance(groupby, str):
            groupby_list = [groupby]
        else:
            groupby_list = groupby
        missing_groupby_cols = [col for col in groupby_list if col not in batch.columns]
        if missing_groupby_cols:
            raise ValueError(f"Groupby columns not found in DataFrame: {missing_groupby_cols}. Available: {list(batch.columns)}")

    if not _CUDF_AVAILABLE or num_gpus == 0:
        # Fallback to pandas aggregation
        if groupby:
            return batch.groupby(groupby).agg(agg_dict).reset_index()
        else:
            return batch.agg(agg_dict).to_frame().T

    gdf = None
    result_gdf = None
    try:
        # Try zero-copy conversion first for performance
        try:
            gdf = cudf.DataFrame.from_pandas(batch, allow_copy=False)
        except (ValueError, TypeError):
            gdf = cudf.DataFrame.from_pandas(batch, allow_copy=True)

        if groupby:
            result_gdf = gdf.groupby(groupby).agg(agg_dict).reset_index()
        else:
            result_gdf = gdf.agg(agg_dict).to_frame().T

        # Convert to pandas for Ray Data compatibility
        result_pd = result_gdf.to_pandas()
        
        # Validate pandas conversion succeeded
        if not isinstance(result_pd, pd.DataFrame):
            raise RuntimeError(f"cuDF to_pandas() returned non-DataFrame: {type(result_pd)}")
        
        # Explicitly free GPU memory
        del result_gdf
        del gdf
        
        return result_pd
    except (ValueError, TypeError, RuntimeError) as e:
        logger.warning(f"GPU aggregation failed, falling back to CPU: {e}")
        # Cleanup on error
        if result_gdf is not None:
            del result_gdf
        if gdf is not None:
            del gdf
        if groupby:
            return batch.groupby(groupby).agg(agg_dict).reset_index()
        else:
            return batch.agg(agg_dict).to_frame().T
    finally:
        # Final cleanup
        if result_gdf is not None:
            del result_gdf
        if gdf is not None:
            del gdf


def gpu_filter(
    batch: pd.DataFrame,
    condition: Callable[[Any], Any],
    num_gpus: int = 1,
) -> pd.DataFrame:
    """GPU-accelerated filtering using cuDF.

    When calling this function via dataset.map_batches(), you must specify
    batch_format="pandas":
    
    ```python
    dataset.map_batches(
        lambda batch: gpu_filter(batch, lambda x: x["value"] > 100, num_gpus=1),
        batch_format="pandas",  # REQUIRED: Must specify pandas format
        batch_size=1000,
    )
    ```

    Args:
        batch: Pandas DataFrame batch
        condition: Function that takes cuDF DataFrame and returns boolean Series
        num_gpus: Number of GPUs to use

    Returns:
        Filtered DataFrame
    """
    # Validate num_gpus parameter
    if num_gpus < 0:
        raise ValueError(f"num_gpus must be >= 0, got {num_gpus}")

    if not _CUDF_AVAILABLE or num_gpus == 0:
        # Fallback to pandas filtering
        # Validate condition function returns boolean Series/array
        mask = condition(batch)
        if not hasattr(mask, '__getitem__') or not hasattr(mask, '__len__'):
            raise ValueError(f"condition function must return boolean Series/array, got {type(mask)}")
        return batch[mask]

    gdf = None
    result_gdf = None
    try:
        # Try zero-copy conversion first for performance
        try:
            gdf = cudf.DataFrame.from_pandas(batch, allow_copy=False)
        except (ValueError, TypeError):
            gdf = cudf.DataFrame.from_pandas(batch, allow_copy=True)
        mask = condition(gdf)
        # Validate mask is boolean Series/array
        if not hasattr(mask, '__getitem__') or not hasattr(mask, '__len__'):
            raise ValueError(f"condition function must return boolean Series/array, got {type(mask)}")
        result_gdf = gdf[mask]
        
        # Convert to pandas for Ray Data compatibility
        result_pd = result_gdf.to_pandas()
        
        # Validate pandas conversion succeeded
        if not isinstance(result_pd, pd.DataFrame):
            raise RuntimeError(f"cuDF to_pandas() returned non-DataFrame: {type(result_pd)}")
        
        # Explicitly free GPU memory
        del result_gdf
        del gdf
        
        return result_pd
    except (ValueError, TypeError, RuntimeError) as e:
        logger.warning(f"GPU filter failed, falling back to CPU: {e}")
        # Cleanup on error
        if result_gdf is not None:
            del result_gdf
        if gdf is not None:
            del gdf
        mask = condition(batch)
        return batch[mask]
    finally:
        # Final cleanup
        if result_gdf is not None:
            del result_gdf
        if gdf is not None:
            del gdf


def gpu_sort(
    batch: pd.DataFrame,
    by: Union[str, List[str]],
    ascending: Union[bool, List[bool]] = True,
    num_gpus: int = 1,
) -> pd.DataFrame:
    """GPU-accelerated sorting using cuDF.

    When calling this function via dataset.map_batches(), you must specify
    batch_format="pandas":
    
    ```python
    dataset.map_batches(
        lambda batch: gpu_sort(batch, by=["value"], num_gpus=1),
        batch_format="pandas",  # REQUIRED: Must specify pandas format
        batch_size=1000,
    )
    ```

    Args:
        batch: Pandas DataFrame batch
        by: Column(s) to sort by
        ascending: Sort order
        num_gpus: Number of GPUs to use

    Returns:
        Sorted DataFrame
    """
    # Validate num_gpus parameter
    if num_gpus < 0:
        raise ValueError(f"num_gpus must be >= 0, got {num_gpus}")

    # Validate sort columns exist
    if isinstance(by, str):
        by_list = [by]
    else:
        by_list = by
    missing_cols = [col for col in by_list if col not in batch.columns]
    if missing_cols:
        raise ValueError(f"Sort columns not found in DataFrame: {missing_cols}. Available: {list(batch.columns)}")

    if not _CUDF_AVAILABLE or num_gpus == 0:
        return batch.sort_values(by=by, ascending=ascending)

    gdf = None
    result_gdf = None
    try:
        # Try zero-copy conversion first for performance
        try:
            gdf = cudf.DataFrame.from_pandas(batch, allow_copy=False)
        except (ValueError, TypeError):
            gdf = cudf.DataFrame.from_pandas(batch, allow_copy=True)
        result_gdf = gdf.sort_values(by=by, ascending=ascending)
        
        # Convert to pandas for Ray Data compatibility
        result_pd = result_gdf.to_pandas()
        
        # Validate pandas conversion succeeded
        if not isinstance(result_pd, pd.DataFrame):
            raise RuntimeError(f"cuDF to_pandas() returned non-DataFrame: {type(result_pd)}")
        
        # Explicitly free GPU memory
        del result_gdf
        del gdf
        
        return result_pd
    except (ValueError, TypeError, RuntimeError) as e:
        logger.warning(f"GPU sort failed, falling back to CPU: {e}")
        # Cleanup on error
        if result_gdf is not None:
            del result_gdf
        if gdf is not None:
            del gdf
        return batch.sort_values(by=by, ascending=ascending)
    finally:
        # Final cleanup
        if result_gdf is not None:
            del result_gdf
        if gdf is not None:
            del gdf


def gpu_string_operations(
    batch: pd.DataFrame,
    column: str,
    operation: str,
    pattern: Optional[str] = None,
    replacement: Optional[str] = None,
    num_gpus: int = 1,
) -> pd.DataFrame:
    """GPU-accelerated string operations using cuDF.

    Args:
        batch: Pandas DataFrame batch
        column: Column to operate on
        operation: Operation type ("lower", "upper", "strip", "replace", "contains")
        pattern: Pattern for replace/contains operations
        replacement: Replacement string for replace operation
        num_gpus: Number of GPUs to use

    Returns:
        DataFrame with transformed string column
    """
    # Validate num_gpus parameter
    if num_gpus < 0:
        raise ValueError(f"num_gpus must be >= 0, got {num_gpus}")

    if not _CUDF_AVAILABLE or num_gpus == 0:
        # Fallback to pandas string operations
        # Validate column exists
        if column not in batch.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame. Available columns: {list(batch.columns)}")
        result = batch.copy()
        if operation == "lower":
            result[column] = result[column].str.lower()
        elif operation == "upper":
            result[column] = result[column].str.upper()
        elif operation == "strip":
            result[column] = result[column].str.strip()
        elif operation == "replace" and pattern and replacement:
            result[column] = result[column].str.replace(pattern, replacement)
        elif operation == "contains" and pattern:
            result[column] = result[column].str.contains(pattern)
        else:
            raise ValueError(f"Invalid operation '{operation}'. Must be one of: lower, upper, strip, replace, contains")
        return result

    # Validate column exists before processing
    if column not in batch.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame. Available columns: {list(batch.columns)}")

    # Validate operation type
    valid_operations = {"lower", "upper", "strip", "replace", "contains"}
    if operation not in valid_operations:
        raise ValueError(f"Invalid operation '{operation}'. Must be one of: {valid_operations}")

    # Validate operation-specific parameters
    if operation in {"replace", "contains"} and pattern is None:
        raise ValueError(f"Operation '{operation}' requires 'pattern' parameter")

    if operation == "replace" and replacement is None:
        raise ValueError("Operation 'replace' requires 'replacement' parameter")

    gdf = None
    try:
        # Try zero-copy conversion first for performance
        try:
            gdf = cudf.DataFrame.from_pandas(batch, allow_copy=False)
        except (ValueError, TypeError):
            gdf = cudf.DataFrame.from_pandas(batch, allow_copy=True)
        if operation == "lower":
            gdf[column] = gdf[column].str.lower()
        elif operation == "upper":
            gdf[column] = gdf[column].str.upper()
        elif operation == "strip":
            gdf[column] = gdf[column].str.strip()
        elif operation == "replace" and pattern and replacement:
            gdf[column] = gdf[column].str.replace(pattern, replacement)
        elif operation == "contains" and pattern:
            gdf[column] = gdf[column].str.contains(pattern)
        
        # Convert to pandas for Ray Data compatibility
        result_pd = gdf.to_pandas()
        
        # Validate pandas conversion succeeded
        if not isinstance(result_pd, pd.DataFrame):
            raise RuntimeError(f"cuDF to_pandas() returned non-DataFrame: {type(result_pd)}")
        
        # Explicitly free GPU memory
        del gdf
        
        return result_pd
    except (ValueError, TypeError, RuntimeError) as e:
        logger.warning(f"GPU string operation failed, falling back to CPU: {e}")
        # Cleanup on error
        if gdf is not None:
            del gdf
        result = batch.copy()
        if operation == "lower":
            result[column] = result[column].str.lower()
        elif operation == "upper":
            result[column] = result[column].str.upper()
        elif operation == "strip":
            result[column] = result[column].str.strip()
        elif operation == "replace" and pattern and replacement:
            result[column] = result[column].str.replace(pattern, replacement)
        elif operation == "contains" and pattern:
            result[column] = result[column].str.contains(pattern)
        else:
            raise ValueError(f"Invalid operation '{operation}'. Must be one of: lower, upper, strip, replace, contains")
        return result
    finally:
        # Final cleanup
        if gdf is not None:
            del gdf


def gpu_window_function(
    batch: pd.DataFrame,
    column: str,
    window_func: str,
    partition_by: Optional[Union[str, List[str]]] = None,
    order_by: Optional[Union[str, List[str]]] = None,
    num_gpus: int = 1,
) -> pd.DataFrame:
    """GPU-accelerated window functions using cuDF.

    Args:
        batch: Pandas DataFrame batch
        column: Column to apply window function to
        window_func: Window function ("sum", "mean", "max", "min", "rank", "row_number")
        partition_by: Column(s) to partition by
        order_by: Column(s) to order by
        num_gpus: Number of GPUs to use

    Returns:
        DataFrame with window function applied
    """
    # Validate num_gpus parameter
    if num_gpus < 0:
        raise ValueError(f"num_gpus must be >= 0, got {num_gpus}")

    # Validate column exists
    if column not in batch.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame. Available columns: {list(batch.columns)}")

    # Validate window function
    valid_window_funcs = {"sum", "mean", "max", "min", "rank", "row_number"}
    if window_func not in valid_window_funcs:
        raise ValueError(f"Invalid window function '{window_func}'. Must be one of: {valid_window_funcs}")

    # Validate partition_by columns exist if specified
    if partition_by is not None:
        if isinstance(partition_by, str):
            partition_list = [partition_by]
        else:
            partition_list = partition_by
        missing_cols = [col for col in partition_list if col not in batch.columns]
        if missing_cols:
            raise ValueError(f"Partition columns not found: {missing_cols}. Available: {list(batch.columns)}")

    # Validate order_by columns exist if specified
    if order_by is not None:
        if isinstance(order_by, str):
            order_list = [order_by]
        else:
            order_list = order_by
        missing_cols = [col for col in order_list if col not in batch.columns]
        if missing_cols:
            raise ValueError(f"Order columns not found: {missing_cols}. Available: {list(batch.columns)}")

    if not _CUDF_AVAILABLE or num_gpus == 0:
        # Fallback to pandas window functions
        result = batch.copy()
        if partition_by:
            grouped = result.groupby(partition_by)
            if order_by:
                result = result.sort_values(order_by)
            if window_func == "sum":
                result[f"{column}_{window_func}"] = grouped[column].transform("sum")
            elif window_func == "mean":
                result[f"{column}_{window_func}"] = grouped[column].transform("mean")
            elif window_func == "max":
                result[f"{column}_{window_func}"] = grouped[column].transform("max")
            elif window_func == "min":
                result[f"{column}_{window_func}"] = grouped[column].transform("min")
        return result

    gdf = None
    try:
        # Try zero-copy conversion first for performance
        try:
            gdf = cudf.DataFrame.from_pandas(batch, allow_copy=False)
        except (ValueError, TypeError):
            gdf = cudf.DataFrame.from_pandas(batch, allow_copy=True)
        if partition_by:
            grouped = gdf.groupby(partition_by)
            if order_by:
                gdf = gdf.sort_values(order_by)
            if window_func == "sum":
                gdf[f"{column}_{window_func}"] = grouped[column].transform("sum")
            elif window_func == "mean":
                gdf[f"{column}_{window_func}"] = grouped[column].transform("mean")
            elif window_func == "max":
                gdf[f"{column}_{window_func}"] = grouped[column].transform("max")
            elif window_func == "min":
                gdf[f"{column}_{window_func}"] = grouped[column].transform("min")
        
        # Convert to pandas for Ray Data compatibility
        result_pd = gdf.to_pandas()
        
        # Validate pandas conversion succeeded
        if not isinstance(result_pd, pd.DataFrame):
            raise RuntimeError(f"cuDF to_pandas() returned non-DataFrame: {type(result_pd)}")
        
        # Explicitly free GPU memory
        del gdf
        
        return result_pd
    except (ValueError, TypeError, RuntimeError) as e:
        logger.warning(f"GPU window function failed, falling back to CPU: {e}")
        # Cleanup on error
        if gdf is not None:
            del gdf
        result = batch.copy()
        if partition_by:
            grouped = result.groupby(partition_by)
            if order_by:
                result = result.sort_values(order_by)
            if window_func == "sum":
                result[f"{column}_{window_func}"] = grouped[column].transform("sum")
            elif window_func == "mean":
                result[f"{column}_{window_func}"] = grouped[column].transform("mean")
            elif window_func == "max":
                result[f"{column}_{window_func}"] = grouped[column].transform("max")
            elif window_func == "min":
                result[f"{column}_{window_func}"] = grouped[column].transform("min")
        return result
    finally:
        # Final cleanup
        if gdf is not None:
            del gdf


def gpu_drop_duplicates(
    batch: pd.DataFrame,
    subset: Optional[Union[str, List[str]]] = None,
    keep: str = "first",
    num_gpus: int = 1,
) -> pd.DataFrame:
    """GPU-accelerated duplicate removal using cuDF.

    Args:
        batch: Pandas DataFrame batch
        subset: Column(s) to check for duplicates
        keep: Which duplicates to keep ("first", "last", False)
        num_gpus: Number of GPUs to use

    Returns:
        DataFrame with duplicates removed
    """
    # Validate num_gpus parameter
    if num_gpus < 0:
        raise ValueError(f"num_gpus must be >= 0, got {num_gpus}")

    # Validate subset columns exist if specified
    if subset is not None:
        if isinstance(subset, str):
            subset_list = [subset]
        else:
            subset_list = subset
        missing_cols = [col for col in subset_list if col not in batch.columns]
        if missing_cols:
            raise ValueError(f"Subset columns not found in DataFrame: {missing_cols}. Available: {list(batch.columns)}")

    # Validate keep parameter
    valid_keep_values = {"first", "last", False}
    if keep not in valid_keep_values:
        raise ValueError(f"Invalid keep value '{keep}'. Must be one of: {valid_keep_values}")

    if not _CUDF_AVAILABLE or num_gpus == 0:
        return batch.drop_duplicates(subset=subset, keep=keep)

    gdf = None
    result_gdf = None
    try:
        # Try zero-copy conversion first for performance
        try:
            gdf = cudf.DataFrame.from_pandas(batch, allow_copy=False)
        except (ValueError, TypeError):
            gdf = cudf.DataFrame.from_pandas(batch, allow_copy=True)
        result_gdf = gdf.drop_duplicates(subset=subset, keep=keep)
        
        # Convert to pandas for Ray Data compatibility
        result_pd = result_gdf.to_pandas()
        
        # Validate pandas conversion succeeded
        if not isinstance(result_pd, pd.DataFrame):
            raise RuntimeError(f"cuDF to_pandas() returned non-DataFrame: {type(result_pd)}")
        
        # Explicitly free GPU memory
        del result_gdf
        del gdf
        
        return result_pd
    except (ValueError, TypeError, RuntimeError) as e:
        logger.warning(f"GPU drop_duplicates failed, falling back to CPU: {e}")
        # Cleanup on error
        if result_gdf is not None:
            del result_gdf
        if gdf is not None:
            del gdf
        return batch.drop_duplicates(subset=subset, keep=keep)
    finally:
        # Final cleanup
        if result_gdf is not None:
            del result_gdf
        if gdf is not None:
            del gdf


def gpu_fillna(
    batch: pd.DataFrame,
    value: Optional[Union[Any, Dict[str, Any]]] = None,
    method: Optional[str] = None,
    num_gpus: int = 1,
) -> pd.DataFrame:
    """GPU-accelerated fillna using cuDF.

    Args:
        batch: Pandas DataFrame batch
        value: Value to fill with
        method: Fill method ("ffill", "bfill")
        num_gpus: Number of GPUs to use

    Returns:
        DataFrame with NaN values filled
    """
    # Validate num_gpus parameter
    if num_gpus < 0:
        raise ValueError(f"num_gpus must be >= 0, got {num_gpus}")

    # Validate fillna parameters
    if method is not None and value is not None:
        raise ValueError("Cannot specify both 'method' and 'value' for fillna")
    if method is not None:
        valid_methods = {"ffill", "bfill", "pad", "backfill"}
        if method not in valid_methods:
            raise ValueError(f"Invalid fillna method '{method}'. Must be one of: {valid_methods}")

    if not _CUDF_AVAILABLE or num_gpus == 0:
        if method:
            return batch.fillna(method=method)
        else:
            return batch.fillna(value=value)

    gdf = None
    result_gdf = None
    try:
        # Try zero-copy conversion first for performance
        try:
            gdf = cudf.DataFrame.from_pandas(batch, allow_copy=False)
        except (ValueError, TypeError):
            gdf = cudf.DataFrame.from_pandas(batch, allow_copy=True)
        # cuDF fillna with method parameter may not support all pandas methods
        # Validate and use appropriate cuDF API
        if method:
            # cuDF supports 'ffill' and 'bfill' methods
            if method in {"ffill", "pad"}:
                result_gdf = gdf.fillna(method="ffill")
            elif method in {"bfill", "backfill"}:
                result_gdf = gdf.fillna(method="bfill")
            else:
                raise ValueError(f"cuDF does not support fillna method '{method}'")
        else:
            result_gdf = gdf.fillna(value=value)
        
        # Convert to pandas for Ray Data compatibility
        result_pd = result_gdf.to_pandas()
        
        # Validate pandas conversion succeeded
        if not isinstance(result_pd, pd.DataFrame):
            raise RuntimeError(f"cuDF to_pandas() returned non-DataFrame: {type(result_pd)}")
        
        # Explicitly free GPU memory
        del result_gdf
        del gdf
        
        return result_pd
    except (ValueError, TypeError, RuntimeError) as e:
        logger.warning(f"GPU fillna failed, falling back to CPU: {e}")
        # Cleanup on error
        if result_gdf is not None:
            del result_gdf
        if gdf is not None:
            del gdf
        if method:
            return batch.fillna(method=method)
        else:
            return batch.fillna(value=value)
    finally:
        # Final cleanup
        if result_gdf is not None:
            del result_gdf
        if gdf is not None:
            del gdf


def gpu_pivot(
    batch: pd.DataFrame,
    index: Union[str, List[str]],
    columns: str,
    values: str,
    num_gpus: int = 1,
) -> pd.DataFrame:
    """GPU-accelerated pivot operation using cuDF.

    Args:
        batch: Pandas DataFrame batch
        index: Column(s) to use as index
        columns: Column to pivot
        values: Column(s) to aggregate
        num_gpus: Number of GPUs to use

    Returns:
        Pivoted DataFrame
    """
    # Validate num_gpus parameter
    if num_gpus < 0:
        raise ValueError(f"num_gpus must be >= 0, got {num_gpus}")

    # Validate pivot columns exist
    if isinstance(index, str):
        index_list = [index]
    else:
        index_list = index
    missing_cols = [col for col in index_list if col not in batch.columns]
    if columns not in batch.columns:
        missing_cols.append(columns)
    if values not in batch.columns:
        missing_cols.append(values)
    if missing_cols:
        raise ValueError(f"Pivot columns not found in DataFrame: {missing_cols}. Available: {list(batch.columns)}")

    if not _CUDF_AVAILABLE or num_gpus == 0:
        return batch.pivot_table(index=index, columns=columns, values=values)

    gdf = None
    result_gdf = None
    try:
        # Try zero-copy conversion first for performance
        try:
            gdf = cudf.DataFrame.from_pandas(batch, allow_copy=False)
        except (ValueError, TypeError):
            gdf = cudf.DataFrame.from_pandas(batch, allow_copy=True)
        result_gdf = gdf.pivot_table(index=index, columns=columns, values=values)
        
        # Convert to pandas for Ray Data compatibility
        result_pd = result_gdf.to_pandas()
        
        # Validate pandas conversion succeeded
        if not isinstance(result_pd, pd.DataFrame):
            raise RuntimeError(f"cuDF to_pandas() returned non-DataFrame: {type(result_pd)}")
        
        # Explicitly free GPU memory
        del result_gdf
        del gdf
        
        return result_pd
    except (ValueError, TypeError, RuntimeError) as e:
        logger.warning(f"GPU pivot failed, falling back to CPU: {e}")
        # Cleanup on error
        if result_gdf is not None:
            del result_gdf
        if gdf is not None:
            del gdf
        return batch.pivot_table(index=index, columns=columns, values=values)
    finally:
        # Final cleanup
        if result_gdf is not None:
            del result_gdf
        if gdf is not None:
            del gdf


def gpu_concat(
    batches: List[pd.DataFrame],
    axis: int = 0,
    num_gpus: int = 1,
) -> pd.DataFrame:
    """GPU-accelerated concatenation using cuDF.

    Args:
        batches: List of DataFrames to concatenate
        axis: Concatenation axis (0=rows, 1=columns)
        num_gpus: Number of GPUs to use

    Returns:
        Concatenated DataFrame
    
    Raises:
        ValueError: If batches list is empty or contains invalid DataFrames
    """
    # Validate input
    if not batches:
        raise ValueError("Cannot concatenate empty list of batches")
    if not all(isinstance(b, pd.DataFrame) for b in batches):
        invalid_types = [type(b).__name__ for b in batches if not isinstance(b, pd.DataFrame)]
        raise ValueError(f"All batches must be pandas DataFrames, found: {invalid_types}")

    # Validate num_gpus parameter
    if num_gpus < 0:
        raise ValueError(f"num_gpus must be >= 0, got {num_gpus}")

    # Validate axis parameter
    if axis not in {0, 1}:
        raise ValueError(f"axis must be 0 (rows) or 1 (columns), got {axis}")

    if not _CUDF_AVAILABLE or num_gpus == 0:
        return pd.concat(batches, axis=axis)

    gdfs = None
    result_gdf = None
    try:
        # Try zero-copy conversion first for performance
        gdfs = []
        for batch in batches:
            try:
                gdf = cudf.DataFrame.from_pandas(batch, allow_copy=False)
            except (ValueError, TypeError):
                gdf = cudf.DataFrame.from_pandas(batch, allow_copy=True)
            gdfs.append(gdf)
        # cuDF concat may fail if DataFrames have incompatible schemas
        # Validate schema compatibility before concatenation
        if len(gdfs) > 1:
            # Check column compatibility for axis=0 (row concatenation)
            if axis == 0:
                first_cols = set(gdfs[0].columns)
                for i, gdf in enumerate(gdfs[1:], 1):
                    if set(gdf.columns) != first_cols:
                        raise ValueError(
                            f"DataFrame {i} has incompatible columns for axis=0 concat. "
                            f"Expected: {sorted(first_cols)}, got: {sorted(gdf.columns)}"
                        )
            # For axis=1 (column concatenation), check row count compatibility
            elif axis == 1:
                first_len = len(gdfs[0])
                for i, gdf in enumerate(gdfs[1:], 1):
                    if len(gdf) != first_len:
                        raise ValueError(
                            f"DataFrame {i} has incompatible row count for axis=1 concat. "
                            f"Expected: {first_len}, got: {len(gdf)}"
                        )
        result_gdf = cudf.concat(gdfs, axis=axis)
        
        # Convert to pandas for Ray Data compatibility
        result_pd = result_gdf.to_pandas()
        
        # Validate pandas conversion succeeded
        if not isinstance(result_pd, pd.DataFrame):
            raise RuntimeError(f"cuDF to_pandas() returned non-DataFrame: {type(result_pd)}")
        
        # Explicitly free GPU memory
        del result_gdf
        if gdfs:
            for gdf in gdfs:
                del gdf
        del gdfs
        
        return result_pd
    except (ValueError, TypeError, RuntimeError) as e:
        logger.warning(f"GPU concat failed, falling back to CPU: {e}")
        # Cleanup on error
        if result_gdf is not None:
            del result_gdf
        if gdfs:
            for gdf in gdfs:
                del gdf
        if gdfs is not None:
            del gdfs
        return pd.concat(batches, axis=axis)
    finally:
        # Final cleanup
        if result_gdf is not None:
            del result_gdf
        if gdfs:
            for gdf in gdfs:
                del gdf
        if gdfs is not None:
            del gdfs

