"""GPU-accelerated utilities module.

Provides GPU-accelerated ETL operations, array operations, memory management,
and monitoring using cuDF, cuPy, and RAPIDS.
"""

from pipeline.utils.gpu.etl import (
    gpu_aggregate,
    gpu_concat,
    gpu_drop_duplicates,
    gpu_fillna,
    gpu_filter,
    gpu_join,
    gpu_map_batches_transform,
    gpu_pivot,
    gpu_sort,
    gpu_string_operations,
    gpu_window_function,
)
from pipeline.utils.gpu.arrays import (
    gpu_array_stats,
    gpu_normalize,
    gpu_remove_outliers,
)
from pipeline.utils.gpu.joins import (
    gpu_join_operator,
    gpu_union_operator,
)
from pipeline.utils.gpu.batch import (
    gpu_batch_aggregate,
    gpu_batch_groupby,
    gpu_batch_sort,
    gpu_batch_transform,
)

__all__ = [
    # ETL operations
    "gpu_map_batches_transform",
    "gpu_join",
    "gpu_aggregate",
    "gpu_filter",
    "gpu_sort",
    "gpu_string_operations",
    "gpu_window_function",
    "gpu_drop_duplicates",
    "gpu_fillna",
    "gpu_pivot",
    "gpu_concat",
    # Array operations
    "gpu_array_stats",
    "gpu_normalize",
    "gpu_remove_outliers",
    # Join operations
    "gpu_join_operator",
    "gpu_union_operator",
    # Batch operations
    "gpu_batch_aggregate",
    "gpu_batch_transform",
    "gpu_batch_sort",
    "gpu_batch_groupby",
]

