"""Column manipulation operations for PipelineDataFrame.

Uses Ray Data's native column operations for better performance.
"""

from __future__ import annotations

from typing import Any, Optional, Union

from ray.data import Dataset

from pipeline.api.dataframe.shared import validate_columns


class ColumnOperationsMixin:
    """Mixin class for column manipulation operations."""
    
    def drop_columns(self, columns: Union[str, list[str]]) -> "PipelineDataFrame":
        """Drop columns from DataFrame.

        Uses Ray Data's native drop_columns() for better performance.

        Args:
            columns: Column name(s) to drop

        Returns:
            New PipelineDataFrame without specified columns

        Raises:
            ValueError: If columns is empty

        Example:
            ```python
            df.drop_columns("unused_col")
            df.drop_columns(["col1", "col2"])
            ```
        """
        if isinstance(columns, str):
            columns = [columns]
        
        if not columns:
            raise ValueError("At least one column must be specified")
        
        # Use Ray Data's native drop_columns for better performance
        dropped = self._dataset.drop_columns(columns)
        return self._create_dataframe(dropped)
    
    def add_column(
        self,
        name: str,
        value: Any,
    ) -> "PipelineDataFrame":
        """Add a constant column.

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
        added = self._dataset.add_column(name, value)
        return self._create_dataframe(added)

