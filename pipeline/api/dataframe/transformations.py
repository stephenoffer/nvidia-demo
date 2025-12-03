"""Data transformation operations for PipelineDataFrame."""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional, Union

from ray.data import Dataset

from pipeline.api.dataframe.expressions import Expr, _EXPRESSIONS_AVAILABLE, col as _col
from pipeline.api.dataframe.shared import batch_to_items, items_to_batch, validate_batch_structure

logger = logging.getLogger(__name__)


class TransformationsMixin:
    """Mixin class for data transformation operations."""
    
    def filter(self, predicate: Union[Callable[[dict], bool], Expr] = None, *, expr: Optional[Expr] = None, fn: Optional[Callable[[dict], bool]] = None) -> "PipelineDataFrame":
        """Filter rows based on predicate.

        Inspired by Spark's filter() and Polars' filter().

        Args:
            predicate: Function that returns True to keep row, or Ray Data expression.
                      Expressions are preferred for better performance.
            expr: Ray Data expression (keyword-only, preferred for performance).
            fn: Function predicate (keyword-only, for backward compatibility).

        Returns:
            New PipelineDataFrame with filtered data

        Raises:
            TypeError: If predicate is not callable or an expression
            ValueError: If multiple predicate arguments provided

        Example:
            ```python
            # Using lambda function (works but less efficient)
            df.filter(lambda x: x["quality"] > 0.8)
            
            # Using expression (preferred, more efficient)
            from pipeline.api.dataframe.expressions import col
            df.filter(expr=col("quality") > 0.8)
            ```
        """
        # Determine which predicate to use
        if expr is not None:
            if predicate is not None or fn is not None:
                raise ValueError("Cannot specify both expr and predicate/fn")
            if not _EXPRESSIONS_AVAILABLE:
                raise ImportError("Ray Data expressions API not available")
            if not isinstance(expr, Expr):
                raise TypeError(f"expr must be a Ray Data expression, got {type(expr)}")
            filtered = self._dataset.filter(expr=expr)
        elif fn is not None:
            if predicate is not None:
                raise ValueError("Cannot specify both predicate and fn")
            if not callable(fn):
                raise TypeError("fn must be callable")
            filtered = self._dataset.filter(fn=fn)
        elif predicate is not None:
            # Check if it's an expression
            if _EXPRESSIONS_AVAILABLE and isinstance(predicate, Expr):
                filtered = self._dataset.filter(expr=predicate)
            elif callable(predicate):
                # Use function-based filter (works but less efficient)
                filtered = self._dataset.filter(fn=predicate)
            else:
                raise TypeError(
                    f"predicate must be callable or Ray Data expression, got {type(predicate)}"
                )
        else:
            raise ValueError("Must provide predicate, expr, or fn")
        
        return self._create_dataframe(filtered)
    
    def map(
        self,
        func: Callable[[dict], dict],
        **map_kwargs: Any,
    ) -> "PipelineDataFrame":
        """Apply function to each row.

        Inspired by Spark's map() and Polars' map_rows().

        Args:
            func: Function to apply to each row
            **map_kwargs: Additional arguments for map_batches

        Returns:
            New PipelineDataFrame with transformed data

        Raises:
            TypeError: If func is not callable
            ValueError: If func returns invalid result

        Example:
            ```python
            df.map(lambda x: {**x, "processed": True})
            ```
        """
        if not callable(func):
            raise TypeError("func must be callable")
        
        def _map_batch(batch: dict[str, Any]) -> dict[str, Any]:
            """Internal: Map batch."""
            if not batch:
                return {}
            
            items = batch_to_items(batch)
            if not items:
                return {}
            
            # Apply function to each item
            transformed_items = []
            for item in items:
                try:
                    result = func(item)
                    if not isinstance(result, dict):
                        raise ValueError(f"func must return dict, got {type(result)}")
                    transformed_items.append(result)
                except Exception as e:
                    logger.warning(f"Failed to transform item: {e}")
                    continue
            
            if not transformed_items:
                return {}
            
            return items_to_batch(transformed_items)
        
        mapped = self._dataset.map_batches(_map_batch, **map_kwargs)
        return self._create_dataframe(mapped)
    
    def map_batches(
        self,
        func: Callable[[dict], dict],
        batch_size: int = 1000,
        use_gpu: bool = False,
        **map_kwargs: Any,
    ) -> "PipelineDataFrame":
        """Apply function to batches.

        Inspired by Ray Data's map_batches() with GPU support.

        Args:
            func: Function to apply to each batch
            batch_size: Batch size
            use_gpu: Use GPU acceleration
            **map_kwargs: Additional arguments for map_batches

        Returns:
            New PipelineDataFrame with transformed data
        """
        kwargs = {
            "batch_size": batch_size,
            **map_kwargs,
        }
        
        if use_gpu:
            kwargs.setdefault("ray_remote_args", {})["num_gpus"] = 1
        
        mapped = self._dataset.map_batches(func, **kwargs)
        return self._create_dataframe(mapped)
    
    def map_sql(
        self,
        query: str,
        table_name: str = "batch",
        batch_format: str = "pandas",
        **map_kwargs: Any,
    ) -> "PipelineDataFrame":
        """Apply SQL query to each batch using DuckDB.
        
        Executes SQL queries on each batch of data using DuckDB's efficient
        in-process SQL engine. This enables SQL-based transformations on
        distributed data without loading entire datasets into memory.
        
        Args:
            query: SQL query to execute (table name defaults to 'batch')
            table_name: Name of the table in the SQL query (default: "batch")
            batch_format: Batch format for Ray Data ("pandas", "pyarrow", "numpy")
            **map_kwargs: Additional arguments for map_batches
        
        Returns:
            New PipelineDataFrame with SQL-transformed data
        
        Raises:
            ImportError: If DuckDB is not installed
            ValueError: If query is empty or invalid
            RuntimeError: If SQL execution fails
        
        Example:
            ```python
            # Filter rows
            df.map_sql("SELECT * FROM batch WHERE quality > 0.8")
            
            # Aggregate
            df.map_sql("SELECT episode_id, AVG(quality) as avg_quality FROM batch GROUP BY episode_id")
            
            # Transform columns
            df.map_sql("SELECT *, UPPER(name) as upper_name FROM batch")
            ```
        """
        try:
            import duckdb
        except ImportError:
            raise ImportError(
                "DuckDB is required for map_sql. Install it with: pip install duckdb"
            )
        
        if not query or not isinstance(query, str):
            raise ValueError("query must be a non-empty string")
        
        if not table_name or not isinstance(table_name, str):
            raise ValueError("table_name must be a non-empty string")
        
        def _sql_batch(batch: Any) -> Any:
            """Internal: Execute SQL query on batch."""
            if batch is None or (isinstance(batch, dict) and not batch):
                return batch
            
            try:
                # Create in-memory DuckDB connection
                con = duckdb.connect(database=":memory:")
                
                # Register batch as table
                df = self._prepare_batch_for_sql(batch, batch_format)
                con.register(table_name, df)
                
                # Execute SQL query
                result = con.execute(query).fetchdf()
                
                # Close connection
                con.close()
                
                # Return in appropriate format
                return self._format_sql_result(result, batch_format)
                    
            except Exception as e:
                logger.error(f"SQL query failed on batch: {e}")
                logger.error(f"Query: {query}")
                raise RuntimeError(f"SQL execution failed: {e}") from e
        
        # Use pandas batch format by default for DuckDB compatibility
        kwargs = {
            "batch_format": batch_format,
            **map_kwargs,
        }
        
        mapped = self._dataset.map_batches(_sql_batch, **kwargs)
        return self._create_dataframe(mapped)
    
    def flat_map(
        self,
        func: Callable[[dict], list[dict]],
        **map_kwargs: Any,
    ) -> "PipelineDataFrame":
        """Apply function that returns multiple rows per input row.

        Inspired by Spark's flatMap().

        Args:
            func: Function that returns list of rows
            **map_kwargs: Additional arguments for map_batches

        Returns:
            New PipelineDataFrame with flattened data

        Raises:
            TypeError: If func is not callable
            ValueError: If func returns invalid result

        Example:
            ```python
            df.flat_map(lambda x: [x, {**x, "augmented": True}])
            ```
        """
        if not callable(func):
            raise TypeError("func must be callable")
        
        def _flat_map_batch(batch: dict[str, Any]) -> dict[str, Any]:
            """Internal: Flat map batch."""
            if not batch:
                return {}
            
            items = batch_to_items(batch)
            if not items:
                return {}
            
            # Apply function and flatten
            flattened_items = []
            for item in items:
                try:
                    result_items = func(item)
                    if not isinstance(result_items, (list, tuple)):
                        result_items = [result_items]
                    
                    for result_item in result_items:
                        if not isinstance(result_item, dict):
                            raise ValueError(f"func must return list of dicts, got {type(result_item)}")
                        flattened_items.append(result_item)
                except Exception as e:
                    logger.warning(f"Failed to flat_map item: {e}")
                    continue
            
            if not flattened_items:
                return {}
            
            return items_to_batch(flattened_items)
        
        mapped = self._dataset.map_batches(_flat_map_batch, **map_kwargs)
        return self._create_dataframe(mapped)
    
    def select(self, *columns: str) -> "PipelineDataFrame":
        """Select specific columns.

        Inspired by Spark's select() and Polars' select().
        Uses Ray Data's native select_columns() for better performance.

        Args:
            *columns: Column names to select

        Returns:
            New PipelineDataFrame with selected columns

        Raises:
            ValueError: If no columns specified or columns don't exist

        Example:
            ```python
            df.select("image", "sensor_data", "timestamp")
            ```
        """
        if not columns:
            raise ValueError("At least one column must be specified")
        
        columns_list = list(columns)
        
        # Validate columns exist
        from pipeline.api.dataframe.shared import validate_columns
        validate_columns(self._dataset, columns_list)
        
        # Use Ray Data's native select_columns for better performance
        selected = self._dataset.select_columns(columns_list)
        return self._create_dataframe(selected)
    
    def with_column(
        self,
        name: str,
        expr: Union[Callable[[dict], Any], Expr],
    ) -> "PipelineDataFrame":
        """Add or replace column.

        Inspired by Spark's withColumn(). Uses Ray Data's native with_column()
        when expressions are available, otherwise falls back to map().

        Args:
            name: Column name
            expr: Expression (Ray Data Expr) or function that computes column value.
                  Expressions are preferred for better performance.

        Returns:
            New PipelineDataFrame with new column

        Raises:
            ValueError: If name is empty or invalid
            TypeError: If expr is not callable or an expression

        Example:
            ```python
            # Using expression (preferred, uses native Ray Data with_column)
            from pipeline.api.dataframe.expressions import col
            df.with_column("quality_score", col("quality") * 100)
            
            # Using function (fallback, uses map)
            df.with_column("quality_score", lambda x: x["quality"] * 100)
            ```
        """
        if not name or not isinstance(name, str):
            raise ValueError(f"name must be a non-empty string, got {name}")
        
        # Try to use Ray Data's native with_column if expression is provided
        if _EXPRESSIONS_AVAILABLE and isinstance(expr, Expr):
            try:
                result = self._dataset.with_column(name, expr)
                return self._create_dataframe(result)
            except Exception as e:
                logger.warning(f"Failed to use native with_column, falling back to map: {e}")
                # Fall through to function-based implementation
        
        # Fallback to function-based implementation
        if not callable(expr):
            raise TypeError(
                f"expr must be callable or Ray Data expression, got {type(expr)}"
            )
        
        def _add_column(row: dict[str, Any]) -> dict[str, Any]:
            """Internal: Add column to row."""
            try:
                value = expr(row)
                return {**row, name: value}
            except Exception as e:
                logger.warning(f"Failed to compute column {name}: {e}")
                return {**row, name: None}
        
        return self.map(_add_column)
    
    def rename(self, columns: dict[str, str]) -> "PipelineDataFrame":
        """Rename columns.

        Inspired by Pandas' rename(). Uses Ray Data's native rename_columns()
        for better performance.

        Args:
            columns: Dictionary mapping old names to new names

        Returns:
            New PipelineDataFrame with renamed columns

        Example:
            ```python
            df.rename({"old_name": "new_name", "old_name2": "new_name2"})
            ```
        """
        # Use Ray Data's native rename_columns for better performance
        renamed = self._dataset.rename_columns(columns)
        return self._create_dataframe(renamed)
    
    # Internal helper methods
    def _prepare_batch_for_sql(self, batch: Any, batch_format: str) -> Any:
        """Internal: Prepare batch for SQL processing."""
        import pandas as pd
        
        if batch_format == "pandas":
            if isinstance(batch, dict):
                return pd.DataFrame(batch)
            elif isinstance(batch, pd.DataFrame):
                return batch
            else:
                raise ValueError(f"Expected pandas DataFrame or dict, got {type(batch)}")
        elif batch_format == "pyarrow":
            import pyarrow as pa
            if isinstance(batch, pa.Table):
                return batch.to_pandas()
            else:
                if isinstance(batch, dict):
                    return pd.DataFrame(batch)
                else:
                    return pd.DataFrame(batch)
        else:
            # Default: convert to pandas
            if isinstance(batch, dict):
                return pd.DataFrame(batch)
            else:
                return pd.DataFrame(batch)
    
    def _format_sql_result(self, result: Any, batch_format: str) -> Any:
        """Internal: Format SQL result for Ray Data."""
        if batch_format == "pyarrow":
            import pyarrow as pa
            return pa.Table.from_pandas(result)
        elif batch_format == "numpy":
            return result.to_dict(orient="list")
        else:
            # Return as pandas DataFrame (Ray Data will handle conversion)
            return result
    

