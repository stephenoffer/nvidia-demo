"""Aggregation operations for PipelineDataFrame."""

from __future__ import annotations

from pipeline.api.dataframe.grouped import GroupedDataFrame


class AggregationsMixin:
    """Mixin class for aggregation operations."""
    
    def groupby(
        self,
        *keys: str,
    ) -> GroupedDataFrame:
        """Group data by keys.

        Inspired by Spark's groupBy() and Pandas' groupby().

        Args:
            *keys: Column names to group by

        Returns:
            GroupedDataFrame for aggregation

        Raises:
            ValueError: If no keys provided

        Example:
            ```python
            df.groupby("episode_id").agg({"sensor_data": "mean"})
            ```
        """
        if not keys:
            raise ValueError("At least one grouping key must be provided")
        return GroupedDataFrame(self._dataset, list(keys))

