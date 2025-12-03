"""Core PipelineDataFrame class."""

from __future__ import annotations

import logging
from typing import Any, Iterator, Optional

from ray.data import Dataset

logger = logging.getLogger(__name__)


class PipelineDataFrame:
    """DataFrame-like API wrapper around Ray Data Dataset.
    
    Provides a fluent, lazy-evaluation API inspired by Spark, Polars, and Pandas.
    Supports GPU acceleration and seamless integration with AI pipelines.
    
    Supports standard Python built-ins and operators:
    - len(df) - Get number of rows
    - iter(df) - Iterate over rows
    - bool(df) - Check if non-empty
    - value in df - Check membership
    - df + other - Union/concat DataFrames (like pd.concat)
    - df | other - Union DataFrames (alternative syntax)
    - df == other - Check if same dataset object
    - df["column"] - Column access
    - df.column - Attribute-style column access (like Pandas)
    - df[0] - Row indexing
    - df[0:10] - Row slicing (like Pandas)
    - df[["col1", "col2"]] - Select multiple columns
    - df.shape - Get (rows, columns) tuple
    - df.columns - Get list of column names
    - df.empty - Check if empty
    - df.copy() - Create a copy
    
    This is the core class. Methods are added via mixins in other modules.
    """

    def __init__(self, dataset: Dataset, name: Optional[str] = None):
        """Initialize PipelineDataFrame.

        Args:
            dataset: Underlying Ray Data Dataset
            name: Optional name for this DataFrame
        """
        self._dataset = dataset
        self._name = name
        self._cached: Optional[Dataset] = None
        self._is_cached = False

    def __repr__(self) -> str:
        """String representation of DataFrame.
        
        Follows Python conventions: shows essential information without
        triggering expensive operations.
        """
        name_str = f", name='{self._name}'" if self._name else ""
        try:
            # Try to get count for display (may fail for streaming datasets)
            # Use a quick check first to avoid expensive operations
            sample = self._dataset.take(1)
            if sample:
                try:
                    count = self.count()
                    return f"PipelineDataFrame(rows={count}{name_str})"
                except Exception:
                    return f"PipelineDataFrame(streaming=True{name_str})"
            else:
                return f"PipelineDataFrame(rows=0{name_str})"
        except Exception:
            return f"PipelineDataFrame(streaming=True{name_str})"
    
    def __str__(self) -> str:
        """String representation (same as repr for DataFrames).
        
        Follows Python convention where str() and repr() are often the same
        for data structures.
        """
        return self.__repr__()
    
    def __len__(self) -> int:
        """Get number of rows (supports len() built-in).
        
        Returns:
            Number of rows in the dataset
            
        Warning:
            This triggers a full dataset scan. For streaming datasets,
            this may be expensive or unavailable. Consider using
            take() or limit() for large datasets.
            
        Example:
            ```python
            df = PipelineDataFrame.from_paths("s3://bucket/data/")
            num_rows = len(df)  # Standard Python len() support
            ```
        """
        return self.count()
    
    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over rows (supports iter() and for loops).
        
        Yields:
            Row dictionaries
            
        Example:
            ```python
            df = PipelineDataFrame.from_paths("s3://bucket/data/")
            for row in df:  # Standard Python iteration
                print(row)
            ```
        """
        return iter(self._dataset.iter_rows())
    
    def __bool__(self) -> bool:
        """Check if DataFrame is non-empty (supports bool() and if statements).
        
        Returns:
            True if DataFrame has at least one row, False otherwise
            
        Warning:
            This may trigger a dataset scan. For streaming datasets,
            this checks if at least one row exists.
            
        Example:
            ```python
            df = PipelineDataFrame.from_paths("s3://bucket/data/")
            if df:  # Standard Python truthiness
                print("DataFrame has data")
            ```
        """
        try:
            # Try to get at least one row without full scan
            sample = self._dataset.take(1)
            return len(sample) > 0
        except Exception:
            # Fallback: try count (may be expensive)
            try:
                return self.count() > 0
            except Exception:
                # If count fails, assume non-empty (conservative)
                logger.warning("Could not determine if DataFrame is empty, assuming non-empty")
                return True
    
    def __contains__(self, value: Any) -> bool:
        """Check if value exists in DataFrame (supports 'in' operator).
        
        Args:
            value: Value to search for (can be dict for row matching, or any value for column search)
            
        Returns:
            True if value found, False otherwise
            
        Warning:
            This may trigger a dataset scan. For large datasets, consider
            using filter() or specific column operations instead.
            
        Example:
            ```python
            df = PipelineDataFrame.from_paths("s3://bucket/data/")
            if {"episode_id": "123"} in df:  # Check if row exists
                print("Episode found")
            if "some_value" in df["column_name"]:  # Check column values
                print("Value found")
            ```
        """
        # If value is a dict, check for matching row
        if isinstance(value, dict):
            for row in self._dataset.iter_rows():
                if all(row.get(k) == v for k, v in value.items()):
                    return True
            return False
        
        # Otherwise, search all column values
        for row in self._dataset.iter_rows():
            if value in row.values():
                return True
        return False

    @property
    def dataset(self) -> Dataset:
        """Get underlying Ray Data Dataset."""
        return self._dataset
    
    @property
    def shape(self) -> tuple[int, int]:
        """Get DataFrame shape (rows, columns) like Pandas.
        
        Returns:
            Tuple of (number_of_rows, number_of_columns)
            
        Warning:
            This triggers a full dataset scan for row count.
            For streaming datasets, this may be expensive.
            
        Example:
            ```python
            df = PipelineDataFrame.from_paths("s3://bucket/data/")
            rows, cols = df.shape  # (1000, 5) for example
            ```
        """
        try:
            # Get column count from schema or sample
            sample = self._dataset.take(1)
            num_cols = len(sample[0].keys()) if sample else 0
            
            # Get row count
            num_rows = self.count()
            
            return (num_rows, num_cols)
        except Exception:
            # Fallback: try to get from schema if available
            try:
                schema = self._dataset.schema()
                if hasattr(schema, "names"):
                    num_cols = len(schema.names)
                else:
                    num_cols = 0
                num_rows = self.count()
                return (num_rows, num_cols)
            except Exception:
                # If we can't determine, return (0, 0) or raise
                logger.warning("Could not determine DataFrame shape")
                return (0, 0)
    
    @property
    def columns(self) -> list[str]:
        """Get column names like Pandas.
        
        Returns:
            List of column names
            
        Example:
            ```python
            df = PipelineDataFrame.from_paths("s3://bucket/data/")
            col_names = df.columns  # ["episode_id", "image", "text", ...]
            ```
        """
        try:
            # Try to get from schema first (more efficient)
            schema = self._dataset.schema()
            if hasattr(schema, "names"):
                return list(schema.names)
            elif hasattr(schema, "field_names"):
                return list(schema.field_names())
        except Exception:
            pass
        
        # Fallback: sample first row
        try:
            sample = self._dataset.take(1)
            if sample:
                return list(sample[0].keys())
        except Exception:
            pass
        
        return []
    
    @property
    def empty(self) -> bool:
        """Check if DataFrame is empty like Pandas.
        
        Returns:
            True if DataFrame has no rows, False otherwise
            
        Example:
            ```python
            df = PipelineDataFrame.from_paths("s3://bucket/data/")
            if df.empty:
                print("DataFrame is empty")
            ```
        """
        return not self.__bool__()
    
    def copy(self) -> "PipelineDataFrame":
        """Create a copy of the DataFrame like Pandas.
        
        Returns:
            New PipelineDataFrame with same data
            
        Note:
            This creates a shallow copy. The underlying dataset
            is shared until operations modify it (lazy evaluation).
            
        Example:
            ```python
            df = PipelineDataFrame.from_paths("s3://bucket/data/")
            df_copy = df.copy()
            ```
        """
        return self._create_dataframe(self._dataset)
    
    def __getitem__(self, key: Any) -> Any:
        """Support indexing, slicing, and column access.
        
        Supports:
        - df["column"] - Get column values
        - df[0] - Get row at index
        - df[0:10] - Slice rows (like Pandas)
        - df[["col1", "col2"]] - Select multiple columns
        
        Args:
            key: Column name (str), index (int), slice, or list of column names
            
        Returns:
            Column values, row, sliced DataFrame, or DataFrame with selected columns
            
        Example:
            ```python
            df = PipelineDataFrame.from_paths("s3://bucket/data/")
            column = df["episode_id"]  # Get column
            first_row = df[0]  # Get first row
            first_10 = df[0:10]  # Get first 10 rows (slicing)
            selected = df[["col1", "col2"]]  # Select columns
            ```
        """
        if isinstance(key, str):
            # Column access - return column values
            return [row.get(key) for row in self._dataset.iter_rows()]
        elif isinstance(key, int):
            # Index access - return row at index
            rows = self._dataset.take(key + 1)
            if len(rows) > key:
                return rows[key]
            raise IndexError(f"Index {key} out of range")
        elif isinstance(key, slice):
            # Slicing - return sliced DataFrame (like Pandas)
            start = key.start or 0
            stop = key.stop
            step = key.step or 1
            
            if step != 1:
                # For non-unit step, need to collect and slice
                # This is expensive but necessary for step != 1
                if stop is None:
                    # Need to get all rows for full slice
                    all_rows = list(self._dataset.iter_rows())
                else:
                    all_rows = self._dataset.take(stop)
                sliced_rows = all_rows[start:stop:step]
                import ray.data
                sliced_dataset = ray.data.from_items(sliced_rows)
                return self._create_dataframe(sliced_dataset)
            else:
                # Unit step - use limit efficiently
                if stop is not None and start is not None:
                    # Both start and stop: limit to (stop - start) and skip start
                    if start >= stop:
                        # Empty slice
                        import ray.data
                        return self._create_dataframe(ray.data.from_items([]))
                    limited = self._dataset.limit(stop)
                    if start > 0:
                        # Use map_batches to skip rows efficiently
                        def _skip_rows(batch: dict[str, Any], skip_count: int = start) -> dict[str, Any]:
                            """Internal: Skip first skip_count rows from batch."""
                            if not batch:
                                return batch
                            keys = list(batch.keys())
                            if not keys:
                                return batch
                            num_items = len(batch[keys[0]])
                            if skip_count >= num_items:
                                # Skip entire batch
                                return {k: [] for k in keys}
                            # Skip first skip_count items
                            return {k: batch[k][skip_count:] for k in keys}
                        
                        # First, limit to stop, then skip start
                        # For simplicity, use take and skip pattern
                        # Note: Ray Data doesn't have skip(), so we use a workaround
                        rows = limited.take(stop)
                        if len(rows) > start:
                            sliced_rows = rows[start:]
                            import ray.data
                            return self._create_dataframe(ray.data.from_items(sliced_rows))
                        else:
                            # Empty result
                            import ray.data
                            return self._create_dataframe(ray.data.from_items([]))
                    return self._create_dataframe(limited)
                elif stop is not None:
                    # Only stop specified - limit to stop
                    return self._create_dataframe(self._dataset.limit(stop))
                elif start is not None and start > 0:
                    # Only start specified - skip first 'start' rows
                    # Ray Data doesn't have skip(), so we collect and slice
                    all_rows = list(self._dataset.iter_rows())
                    sliced_rows = all_rows[start:]
                    import ray.data
                    return self._create_dataframe(ray.data.from_items(sliced_rows))
                else:
                    # No limits - return self
                    return self
        elif isinstance(key, (list, tuple)):
            # Multiple column selection - return DataFrame with selected columns
            if not key:
                raise ValueError("Column list cannot be empty")
            if not all(isinstance(col, str) for col in key):
                raise TypeError("All column names must be strings")
            
            def _select_columns(row: dict[str, Any]) -> dict[str, Any]:
                """Internal: Select specified columns from row."""
                return {col: row.get(col) for col in key}
            
            selected = self._dataset.map(_select_columns)
            return self._create_dataframe(selected)
        else:
            raise TypeError(
                f"Index must be str (column name), int (row index), "
                f"slice (row slicing), or list (column selection), got {type(key)}"
            )
    
    def __getattr__(self, name: str) -> Any:
        """Support attribute access for columns (supports df.column_name).
        
        Allows accessing columns as attributes, like Pandas.
        Falls back to normal attribute access if column doesn't exist.
        
        Args:
            name: Column name or attribute name
            
        Returns:
            Column values if column exists, otherwise raises AttributeError
            
        Example:
            ```python
            df = PipelineDataFrame.from_paths("s3://bucket/data/")
            column = df.episode_id  # Attribute-style access
            # Equivalent to: df["episode_id"]
            ```
        """
        # Check if it's a valid attribute first
        if name.startswith("_") or name in dir(self):
            return super().__getattribute__(name)
        
        # Try to get as column
        try:
            # Check if column exists by sampling
            sample = self._dataset.take(1)
            if sample and name in sample[0]:
                return self[name]  # Use __getitem__ to get column
        except Exception:
            pass
        
        # Not a column, try normal attribute access
        return super().__getattribute__(name)
    
    def __add__(self, other: "PipelineDataFrame") -> "PipelineDataFrame":
        """Union/concat DataFrames (supports + operator).
        
        Concatenates rows from both DataFrames, similar to pd.concat() or Spark's union().
        This is the standard Python way to combine DataFrames.
        
        Args:
            other: Other PipelineDataFrame to union with
            
        Returns:
            New PipelineDataFrame with unioned data
            
        Raises:
            TypeError: If other is not a PipelineDataFrame
            
        Example:
            ```python
            df1 = PipelineDataFrame.from_paths("s3://bucket/data1/")
            df2 = PipelineDataFrame.from_paths("s3://bucket/data2/")
            combined = df1 + df2  # Standard Python + operator for concat
            # Equivalent to: df1.union(df2)
            ```
        """
        if not isinstance(other, type(self)):
            return NotImplemented
        
        return self.union(other)
    
    def __or__(self, other: "PipelineDataFrame") -> "PipelineDataFrame":
        """Union DataFrames (supports | operator as alternative to +).
        
        Alternative syntax for union operation. Useful for chaining multiple unions.
        
        Args:
            other: Other PipelineDataFrame to union with
            
        Returns:
            New PipelineDataFrame with unioned data
            
        Example:
            ```python
            df1 = PipelineDataFrame.from_paths("s3://bucket/data1/")
            df2 = PipelineDataFrame.from_paths("s3://bucket/data2/")
            df3 = PipelineDataFrame.from_paths("s3://bucket/data3/")
            combined = df1 | df2 | df3  # Chain unions
            ```
        """
        if not isinstance(other, type(self)):
            return NotImplemented
        
        return self.union(other)
    
    def __eq__(self, other: object) -> bool:
        """Check if DataFrames are equal (same dataset reference).
        
        Note: This checks if the underlying datasets are the same object,
        not if they contain the same data. For data comparison, use
        a custom comparison function.
        
        Args:
            other: Object to compare with
            
        Returns:
            True if same dataset object, False otherwise
        """
        if not isinstance(other, type(self)):
            return NotImplemented
        
        return self._dataset is other._dataset
    
    def __ne__(self, other: object) -> bool:
        """Check if DataFrames are not equal.
        
        Args:
            other: Object to compare with
            
        Returns:
            True if different dataset objects, False otherwise
        """
        return not self.__eq__(other)
    
    # Internal helper method for mixins
    def _create_dataframe(self, dataset: Dataset) -> "PipelineDataFrame":
        """Internal: Create new PipelineDataFrame instance.
        
        This method is used by mixins to create new instances while preserving
        the name and other attributes. Uses type(self) to ensure the same
        class type is returned (including mixins).
        
        Args:
            dataset: New Ray Data Dataset
            
        Returns:
            New PipelineDataFrame instance of the same type
        """
        return type(self)(dataset, name=self._name)

