"""Column manipulation operations for PipelineDataFrame.

Uses Ray Data's native column operations for better performance.
"""

from __future__ import annotations

from typing import Any, Optional, Union

from ray.data import Dataset

from pipeline.api.dataframe.shared import validate_columns


class ColumnOperationsMixin:
    """Mixin class for column manipulation operations."""
    
    def drop(self, columns: Union[str, list[str]]) -> "PipelineDataFrame":
        """Drop columns from DataFrame.

        Pandas-style method name. Uses Ray Data's native drop_columns() for better performance.

        Args:
            columns: Column name(s) to drop

        Returns:
            New PipelineDataFrame without specified columns

        Raises:
            ValueError: If columns is empty

        Example:
            ```python
            df.drop("unused_col")
            df.drop(["col1", "col2"])
            ```
        """
        if isinstance(columns, str):
            columns = [columns]
        
        if not columns:
            raise ValueError("At least one column must be specified")
        
        # Use Ray Data's native drop_columns for better performance
        dropped = self._dataset.drop_columns(columns)
        return self._create_dataframe(dropped)

    def drop_columns(self, columns: Union[str, list[str]]) -> "PipelineDataFrame":
        """Drop columns (alias for drop())."""
        return self.drop(columns)
    
    def assign(self, **kwargs: Any) -> "PipelineDataFrame":
        """Assign new columns (Pandas-style).

        Args:
            **kwargs: Column name-value pairs

        Returns:
            New PipelineDataFrame with assigned columns

        Example:
            ```python
            df.assign(status="active", priority=1)
            ```
        """
        result = self._dataset
        for name, value in kwargs.items():
            result = result.add_column(name, value)
        return self._create_dataframe(result)
    
    def add_column(
        self,
        name: str,
        value: Any,
    ) -> "PipelineDataFrame":
        """Add a constant column (alias for assign()).

        Uses Ray Data's native add_column() for better performance.

        Args:
            name: Column name
            value: Constant value to add

        Returns:
            New PipelineDataFrame with added column

        Example:
            ```python
            df.add_column("status", "active")
            ```
        """
        return self.assign(**{name: value})

