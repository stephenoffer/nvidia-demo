# GPU-Accelerated Operations in Ray Data

This directory contains GPU-accelerated utilities for Ray Data operations. It's critical to understand when to use batch-level GPU functions vs Ray Data's native functions for global operations.

## When to Use Batch-Level GPU Functions vs Ray Data Native Functions

### Use Batch-Level GPU Functions (`map_batches()`)

Use batch-level GPU functions when operations can be performed independently on each batch:

- **Transformations**: Element-wise operations, column transformations
- **Filtering**: Row-level filtering that doesn't require global state
- **Batch-level aggregations**: Aggregations that only need to see individual batches
- **Batch-level sorting**: Sorting within batches only

Example:
```python
# Correct: Batch-level transformation
dataset.map_batches(
    lambda batch: gpu_map_batches_transform(batch, transform_func, num_gpus=1),
    batch_format="pandas",
    batch_size=1000,
)
```

### Use Ray Data Native Functions (Global Operations)

**ALWAYS** use Ray Data's native functions for operations that require seeing all data or shuffling:

- **Global joins**: Joining two datasets
- **Global groupby**: Grouping across the entire dataset
- **Global aggregations**: Aggregations that need to see all data
- **Global sorting**: Sorting across the entire dataset
- **Shuffle operations**: Operations that require data redistribution

Example:
```python
# Correct: Global join using Ray Data native function
dataset1.join(dataset2, on="key", how="inner")

# Correct: Global groupby using Ray Data native function
dataset.groupby("key").agg({"value": "sum"})

# Correct: Global aggregation using Ray Data native function
dataset.aggregate(AggregateFnV2(...))
```

## GPU Acceleration for Global Operations

For GPU-accelerating Ray Data's native global operations, use one of these approaches:

### 1. AggregateFnV2 Classes

Use `AggregateFnV2` classes to accelerate global aggregations:

```python
from ray.data.aggregate import AggregateFnV2

class GPUSumAggregateFn(AggregateFnV2):
    def __init__(self):
        # Initialize GPU resources
        
    def accumulate(self, block):
        # Use GPU to accumulate from block
        import cudf
        gdf = cudf.DataFrame.from_pandas(block)
        return gdf.sum()
    
    def merge(self, accumulators):
        # Merge GPU accumulators
        import cudf
        return cudf.concat(accumulators).sum()
    
    def finalize(self, accumulator):
        # Convert GPU result back to CPU
        return accumulator.to_pandas()

# Use in Ray Data
dataset.aggregate(GPUSumAggregateFn())
```

### 2. Custom Operators (e.g., PolarsGPUJoin)

Replace underlying operators with GPU-accelerated versions:

```python
# Example: Replace PolarsJoin with PolarsGPUJoin
from ray.data._internal.operators.join_operator import JoinOperator
import polars as pl

class PolarsGPUJoin(JoinOperator):
    """GPU-accelerated join using Polars with cuDF backend."""
    
    def _execute(self, left, right, on, how):
        # Use Polars with GPU engine
        left_pl = pl.from_pandas(left)
        right_pl = pl.from_pandas(right)
        
        # Execute join on GPU
        result = left_pl.join(
            right_pl,
            on=on,
            how=how,
        ).collect(engine="gpu")  # Use GPU engine
        
        return result.to_pandas()
```

### 3. Ray Data Configuration

Configure Ray Data to use GPU-accelerated operators where available:

```python
from ray.data import DataContext

ctx = DataContext.get_current()
# Configure GPU-accelerated operators if available
ctx.use_push_based_shuffle = True  # For better shuffle performance
```

## Common Mistakes

### ❌ Wrong: Using `map_batches()` for Global Operations

```python
# WRONG: This only groups within each batch, not globally
dataset.map_batches(
    lambda batch: gpu_batch_groupby(batch, by=["key"], agg_dict={"value": "sum"}),
    batch_format="pandas",
)
# Result: Each batch is grouped independently, not across the dataset
```

### ✅ Correct: Using Ray Data Native Functions

```python
# CORRECT: Global groupby across entire dataset
dataset.groupby("key").agg({"value": "sum"})
```

### ❌ Wrong: Using Batch-Level Join for Global Join

```python
# WRONG: This doesn't work for joining two datasets
dataset1.map_batches(
    lambda batch: gpu_join(batch, dataset2.take(1000)),  # Can't join datasets this way
    batch_format="pandas",
)
```

### ✅ Correct: Using Ray Data Native Join

```python
# CORRECT: Global join between datasets
dataset1.join(dataset2, on="key", how="inner")
```

## Summary

- **Batch-level operations**: Use GPU functions with `map_batches()`
- **Global operations**: Use Ray Data native functions (`join()`, `groupby()`, `aggregate()`, etc.)
- **GPU acceleration for global ops**: Use `AggregateFnV2` classes or custom operators like `PolarsGPUJoin`

