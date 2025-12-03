"""Dataset-level aggregation operations for PipelineDataFrame."""

from __future__ import annotations

from typing import Any, Optional

from ray.data import Dataset


class DatasetAggregationsMixin:
    """Mixin class for dataset-level aggregation operations.
    
    These are aggregations that operate on the entire dataset, not grouped data.
    Uses Ray Data's native aggregation methods for efficiency.
    """
    
    def max(self, column: str) -> float:
        """Get maximum value of a column.

        Uses Ray Data's native max() method.

        Args:
            column: Column name

        Returns:
            Maximum value

        Example:
            ```python
            max_quality = df.max("quality")
            ```
        """
        return self._dataset.max(column)
    
    def min(self, column: str) -> float:
        """Get minimum value of a column.

        Uses Ray Data's native min() method.

        Args:
            column: Column name

        Returns:
            Minimum value

        Example:
            ```python
            min_quality = df.min("quality")
            ```
        """
        return self._dataset.min(column)
    
    def mean(self, column: str) -> float:
        """Get mean value of a column.

        Uses Ray Data's native mean() method.

        Args:
            column: Column name

        Returns:
            Mean value

        Example:
            ```python
            avg_quality = df.mean("quality")
            ```
        """
        return self._dataset.mean(column)
    
    def sum(self, column: str) -> float:
        """Get sum of a column.

        Uses Ray Data's native sum() method.

        Args:
            column: Column name

        Returns:
            Sum value

        Example:
            ```python
            total = df.sum("value")
            ```
        """
        return self._dataset.sum(column)
    
    def std(self, column: str) -> float:
        """Get standard deviation of a column.

        Uses Ray Data's native std() method.

        Args:
            column: Column name

        Returns:
            Standard deviation

        Example:
            ```python
            std_dev = df.std("quality")
            ```
        """
        return self._dataset.std(column)
    
    def materialize(self) -> "PipelineDataFrame":
        """Materialize the dataset.

        Uses Ray Data's native materialize() to force execution and caching.

        Returns:
            Materialized PipelineDataFrame

        Example:
            ```python
            materialized_df = df.materialize()
            ```
        """
        materialized = self._dataset.materialize()
        return self._create_dataframe(materialized)
    
    def explain(self) -> str:
        """Explain the execution plan.

        Uses Ray Data's native explain() to show logical and physical plans.

        Returns:
            Execution plan explanation

        Example:
            ```python
            plan = df.filter(lambda x: x["a"] > 1).explain()
            print(plan)
            ```
        """
        return self._dataset.explain()
    
    def stats(self) -> dict[str, Any]:
        """Get dataset statistics.

        Uses Ray Data's native stats() method.

        Returns:
            Dictionary with dataset statistics

        Example:
            ```python
            stats = df.stats()
            print(f"Size: {stats['size_bytes']} bytes")
            ```
        """
        return self._dataset.stats()

