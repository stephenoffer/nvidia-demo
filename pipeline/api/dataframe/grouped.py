"""GroupedDataFrame class for aggregation operations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from ray.data import Dataset

if TYPE_CHECKING:
    from pipeline.api.dataframe import PipelineDataFrame
else:
    # Import core to avoid circular import
    from pipeline.api.dataframe.core import PipelineDataFrame as _PipelineDataFrameCore


def _get_pipeline_dataframe_class() -> type:
    """Internal: Get the composed PipelineDataFrame class.
    
    This function is used to get the fully composed PipelineDataFrame class
    at runtime, avoiding circular imports while ensuring the correct type is returned.
    """
    try:
        from pipeline.api.dataframe import PipelineDataFrame
        return PipelineDataFrame
    except ImportError:
        # Fallback to core if composed class not available
        return _PipelineDataFrameCore


class GroupedDataFrame:
    """Grouped DataFrame for aggregation operations.

    Inspired by Spark's GroupedData and Pandas' GroupBy.
    """

    def __init__(self, dataset: Dataset, keys: list[str]):
        """Initialize GroupedDataFrame.

        Args:
            dataset: Underlying Ray Data Dataset
            keys: Grouping keys
        """
        self._dataset = dataset
        self.keys = keys

    def agg(self, aggregations: dict[str, str]) -> PipelineDataFrame:
        """Apply aggregations.

        Inspired by Spark's agg() and Pandas' agg().

        Args:
            aggregations: Dictionary mapping column names to aggregation functions

        Returns:
            PipelineDataFrame with aggregated data

        Example:
            ```python
            df.groupby("episode_id").agg({
                "sensor_data": "mean",
                "timestamp": "max",
                "quality": "min",
            })
            ```
        """
        # Ray Data GroupedData uses aggregate() not agg()
        grouped = self._dataset.groupby(self.keys)
        aggregated = grouped.aggregate(aggregations)
        PipelineDataFrameClass = _get_pipeline_dataframe_class()
        return PipelineDataFrameClass(aggregated)

    def count(self) -> PipelineDataFrame:
        """Count rows per group.

        Returns:
            PipelineDataFrame with counts
        """
        grouped = self._dataset.groupby(self.keys)
        counted = grouped.count()
        PipelineDataFrameClass = _get_pipeline_dataframe_class()
        return PipelineDataFrameClass(counted)

    def sum(self, *columns: str) -> PipelineDataFrame:
        """Sum columns per group.

        Args:
            *columns: Column names to sum

        Returns:
            PipelineDataFrame with sums
        """
        aggregations = {col: "sum" for col in columns}
        return self.agg(aggregations)

    def mean(self, *columns: str) -> PipelineDataFrame:
        """Mean of columns per group.

        Args:
            *columns: Column names to compute mean

        Returns:
            PipelineDataFrame with means
        """
        aggregations = {col: "mean" for col in columns}
        return self.agg(aggregations)

    def max(self, *columns: str) -> PipelineDataFrame:
        """Max of columns per group.

        Args:
            *columns: Column names to compute max

        Returns:
            PipelineDataFrame with max values
        """
        aggregations = {col: "max" for col in columns}
        return self.agg(aggregations)

    def min(self, *columns: str) -> PipelineDataFrame:
        """Min of columns per group.

        Args:
            *columns: Column names to compute min

        Returns:
            PipelineDataFrame with min values
        """
        aggregations = {col: "min" for col in columns}
        return self.agg(aggregations)

    def first(self) -> PipelineDataFrame:
        """First row per group.

        Returns:
            PipelineDataFrame with first rows
        """
        grouped = self._dataset.groupby(self.keys)
        first_rows = grouped.map_groups(lambda group_df: group_df.take(1))
        PipelineDataFrameClass = _get_pipeline_dataframe_class()
        return PipelineDataFrameClass(first_rows)
    
    def last(self) -> PipelineDataFrame:
        """Last row per group.

        Returns:
            PipelineDataFrame with last rows
        """
        grouped = self._dataset.groupby(self.keys)
        last_rows = grouped.map_groups(lambda group_df: group_df.take(-1))
        PipelineDataFrameClass = _get_pipeline_dataframe_class()
        return PipelineDataFrameClass(last_rows)

