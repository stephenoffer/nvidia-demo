"""DataFrame-like API for Ray Data pipelines.

This module is organized into focused submodules:
- core: Core PipelineDataFrame class
- io: Input/output operations
- transformations: Data transformations
- aggregations: Grouping and aggregations
- joins: Join and union operations
- operations: Other operations (sort, limit, etc.)
- grouped: GroupedDataFrame class
- windowed: WindowedDataFrame class
- shared: Shared utilities
"""

from pipeline.api.dataframe.core import PipelineDataFrame as _PipelineDataFrameCore
from pipeline.api.dataframe.grouped import GroupedDataFrame
from pipeline.api.dataframe.windowed import WindowedDataFrame

# Import mixins
from pipeline.api.dataframe.io import IOMixin
from pipeline.api.dataframe.transformations import TransformationsMixin
from pipeline.api.dataframe.aggregations import AggregationsMixin
from pipeline.api.dataframe.aggregations_dataset import DatasetAggregationsMixin
from pipeline.api.dataframe.column_operations import ColumnOperationsMixin
from pipeline.api.dataframe.joins import JoinsMixin
from pipeline.api.dataframe.operations import OperationsMixin

# Compose PipelineDataFrame with all mixins using multiple inheritance
# Core class must be first so its __init__ and _create_dataframe are used
class PipelineDataFrame(
    _PipelineDataFrameCore,  # Core class (must be first)
    IOMixin,            # IO operations
    TransformationsMixin,  # Transformations
    AggregationsMixin,     # Aggregations
    DatasetAggregationsMixin,  # Dataset-level aggregations
    ColumnOperationsMixin,  # Column operations
    JoinsMixin,           # Joins
    OperationsMixin,       # Operations
):
    """Composed PipelineDataFrame with all functionality.
    
    This class combines the core PipelineDataFrame with all mixin classes
    to provide the complete DataFrame API.
    
    The core class is first in MRO so its __init__ and _create_dataframe
    methods are used. Mixins add their methods via multiple inheritance.
    """
    pass

# Re-export the composed class
__all__ = [
    "PipelineDataFrame",
    "GroupedDataFrame",
    "WindowedDataFrame",
]

