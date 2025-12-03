"""General operations for PipelineDataFrame (sort, limit, distinct, etc.)."""

from __future__ import annotations

import hashlib
import json
import logging
import math
from typing import Any, Optional, Union

from ray.data import Dataset

from pipeline.api.dataframe.shared import batch_to_items, items_to_batch, validate_columns

logger = logging.getLogger(__name__)


class OperationsMixin:
    """Mixin class for general operations."""
    
    def head(self, n: int = 10) -> list[dict[str, Any]]:
        """Get first n rows (alias for take).

        Inspired by Pandas' head().

        Args:
            n: Number of rows to return

        Returns:
            List of row dictionaries

        Example:
            ```python
            sample = df.head(10)
            ```
        """
        return self.take(n)
    
    def tail(self, n: int = 10) -> list[dict[str, Any]]:
        """Get last n rows.

        Inspired by Pandas' tail().

        Args:
            n: Number of rows to return

        Returns:
            List of row dictionaries

        Warning:
            This requires loading the entire dataset to find the last n rows.
            For large datasets, consider using limit() and sort() instead.

        Example:
            ```python
            last_rows = df.tail(10)
            ```
        """
        if not isinstance(n, int) or n <= 0:
            raise ValueError(f"n must be a positive integer, got {n}")
        
        # Sort by a stable key (if available) or use limit
        # For simplicity, we'll collect and take last n
        # In production, you'd want to optimize this
        all_rows = self.collect()
        return all_rows[-n:] if len(all_rows) > n else all_rows
    
    def sort(
        self,
        by: Union[str, list[str]],
        ascending: Union[bool, list[bool]] = True,
    ) -> "PipelineDataFrame":
        """Sort by column(s).

        Inspired by Spark's orderBy() and Pandas' sort_values().

        Args:
            by: Column name(s) to sort by
            ascending: Sort order(s)

        Returns:
            New PipelineDataFrame with sorted data

        Example:
            ```python
            df.sort("timestamp", ascending=True)
            df.sort(["episode_id", "timestamp"], ascending=[True, False])
            ```
        """
        if isinstance(by, str):
            by = [by]
        if isinstance(ascending, bool):
            ascending = [ascending] * len(by)
        
        sorted_ds = self._dataset.sort(by, descending=[not a for a in ascending])
        return self._create_dataframe(sorted_ds)
    
    def limit(self, n: int) -> "PipelineDataFrame":
        """Limit number of rows.

        Inspired by Spark's limit() and SQL LIMIT.

        Args:
            n: Number of rows to keep

        Returns:
            New PipelineDataFrame with limited rows

        Raises:
            ValueError: If n is not positive

        Example:
            ```python
            df.limit(1000)
            ```
        """
        if not isinstance(n, int) or n <= 0:
            raise ValueError(f"n must be a positive integer, got {n}")
        
        limited = self._dataset.limit(n)
        return self._create_dataframe(limited)
    
    def distinct(self, *columns: Optional[str]) -> "PipelineDataFrame":
        """Remove duplicate rows.

        Inspired by Spark's distinct().

        Args:
            *columns: Column names to check for duplicates (None = all columns)

        Returns:
            New PipelineDataFrame with distinct rows

        Example:
            ```python
            df.distinct()  # All columns
            df.distinct("episode_id")  # Specific columns
            ```
        """
        if columns and columns[0] is not None:
            # Validate columns exist
            validate_columns(self._dataset, list(columns))
            
            # Group by columns and take first
            grouped = self.groupby(*columns)
            distinct = grouped.first()
            return distinct
        else:
            # Ray Data doesn't have drop_duplicates - implement via map_batches
            # Use hash-based deduplication for streaming compatibility
            seen = set()
            
            def _deduplicate_batch(batch: dict[str, Any]) -> dict[str, Any]:
                """Internal: Remove duplicates from batch using hash of row."""
                if not batch:
                    return {}
                
                items = batch_to_items(batch)
                if not items:
                    return {}
                
                # Deduplicate using hash
                unique_items = []
                for item in items:
                    # Create hash of item (sorted for consistency)
                    item_str = json.dumps(item, sort_keys=True)
                    item_hash = hashlib.md5(item_str.encode()).hexdigest()
                    
                    if item_hash not in seen:
                        seen.add(item_hash)
                        unique_items.append(item)
                
                if not unique_items:
                    return {}
                
                return items_to_batch(unique_items)
            
            distinct = self._dataset.map_batches(_deduplicate_batch)
            return self._create_dataframe(distinct)
    
    def drop_duplicates(
        self,
        subset: Optional[list[str]] = None,
    ) -> "PipelineDataFrame":
        """Drop duplicate rows.

        Inspired by Pandas' drop_duplicates(). Uses Ray Data's native unique()
        when subset is provided, otherwise uses distinct().

        Args:
            subset: Column names to check for duplicates (None = all columns)

        Returns:
            New PipelineDataFrame without duplicates

        Example:
            ```python
            df.drop_duplicates()  # All columns
            df.drop_duplicates(subset=["episode_id"])  # Specific columns
            ```
        """
        if subset:
            # Use Ray Data's native unique() for better performance
            unique = self._dataset.unique(subset)
            return self._create_dataframe(unique)
        else:
            return self.distinct()
    
    def dropna(
        self,
        subset: Optional[list[str]] = None,
    ) -> "PipelineDataFrame":
        """Drop rows with null values.

        Inspired by Pandas' dropna().

        Args:
            subset: Column names to check (None = all columns)

        Returns:
            New PipelineDataFrame without nulls

        Example:
            ```python
            df.dropna()  # All columns
            df.dropna(subset=["sensor_data", "image"])  # Specific columns
            ```
        """
        def _is_not_null(row: dict[str, Any]) -> bool:
            """Internal: Check if row has no nulls in subset."""
            if subset:
                for col in subset:
                    if col not in row:
                        continue
                    val = row[col]
                    # Check for None, NaN, and empty strings
                    if val is None:
                        return False
                    if isinstance(val, float) and math.isnan(val):
                        return False
                    if isinstance(val, str) and not val.strip():
                        return False
                return True
            else:
                for v in row.values():
                    if v is None:
                        return False
                    if isinstance(v, float) and math.isnan(v):
                        return False
                    if isinstance(v, str) and not v.strip():
                        return False
                return True
        
        filtered = self._dataset.filter(_is_not_null)
        return self._create_dataframe(filtered)
    
    def fillna(
        self,
        value: Any,
        subset: Optional[list[str]] = None,
    ) -> "PipelineDataFrame":
        """Fill null values with specified value.

        Inspired by Pandas' fillna().

        Args:
            value: Value to fill nulls with
            subset: Column names to fill (None = all columns)

        Returns:
            New PipelineDataFrame with filled nulls

        Example:
            ```python
            df.fillna(0)  # Fill all nulls with 0
            df.fillna(0, subset=["sensor_data"])  # Fill specific columns
            ```
        """
        def _fill_nulls(row: dict[str, Any]) -> dict[str, Any]:
            """Internal: Fill nulls in row."""
            filled = row.copy()
            if subset:
                for col in subset:
                    if col in filled:
                        if filled[col] is None or (isinstance(filled[col], float) and math.isnan(filled[col])):
                            filled[col] = value
            else:
                for key in filled:
                    if filled[key] is None or (isinstance(filled[key], float) and math.isnan(filled[key])):
                        filled[key] = value
            return filled
        
        return self.map(_fill_nulls)
    
    def sample(
        self,
        fraction: Optional[float] = None,
        n: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> "PipelineDataFrame":
        """Sample rows from DataFrame.

        Inspired by Spark's sample() and Pandas' sample().

        Args:
            fraction: Fraction of rows to sample (0.0-1.0)
            n: Number of rows to sample
            seed: Random seed

        Returns:
            New PipelineDataFrame with sampled data

        Raises:
            ValueError: If both fraction and n specified, or neither specified,
                or values are invalid

        Warning:
            If n is specified, this calls count() which may be expensive for
            streaming datasets.

        Example:
            ```python
            df.sample(fraction=0.1)  # 10% sample
            df.sample(n=1000)  # 1000 rows
            ```
        """
        if fraction is not None and n is not None:
            raise ValueError("Cannot specify both fraction and n")
        
        if fraction is not None:
            if not isinstance(fraction, (int, float)) or fraction < 0.0 or fraction > 1.0:
                raise ValueError(f"fraction must be between 0.0 and 1.0, got {fraction}")
            sampled = self._dataset.random_sample(fraction=fraction, seed=seed)
        elif n is not None:
            if not isinstance(n, int) or n <= 0:
                raise ValueError(f"n must be a positive integer, got {n}")
            # Avoid count() for streaming datasets - use limit after sampling
            # Sample a larger fraction to ensure we get enough rows
            sampled = self._dataset.random_sample(fraction=min(1.0, max(0.1, n / 1000)), seed=seed).limit(n)
        else:
            raise ValueError("Must specify either fraction or n")
        
        return self._create_dataframe(sampled)
    
    def cache(self, storage_level: str = "memory") -> "PipelineDataFrame":
        """Cache DataFrame in memory or disk.

        Inspired by Spark's cache() and persist().

        Args:
            storage_level: Storage level ("memory", "disk", "gpu")

        Returns:
            Self (for method chaining)

        Raises:
            ValueError: If storage_level is invalid

        Note:
            Currently this is a no-op marker. Ray Data handles caching
            automatically. For disk caching, use checkpoint() instead.

        Example:
            ```python
            df.cache()  # Cache in memory
            df.cache("disk")  # Cache on disk
            ```
        """
        valid_levels = {"memory", "disk", "gpu"}
        if storage_level not in valid_levels:
            raise ValueError(f"storage_level must be one of {valid_levels}, got {storage_level}")
        
        # Mark as cached (Ray Data handles actual caching)
        self._cached = self._dataset
        self._is_cached = True
        
        if storage_level == "gpu":
            logger.info("Marked DataFrame for GPU caching")
        elif storage_level == "disk":
            logger.info("Marked DataFrame for disk caching (use checkpoint() for actual disk caching)")
        else:
            logger.info("Marked DataFrame for memory caching")
        
        return self
    
    def persist(self, **kwargs: Any) -> "PipelineDataFrame":
        """Persist DataFrame (alias for cache).

        Inspired by Spark's persist().

        Args:
            **kwargs: Storage options

        Returns:
            Self (for method chaining)
        """
        return self.cache(**kwargs)
    
    def repartition(self, num_partitions: int) -> "PipelineDataFrame":
        """Repartition DataFrame.

        Inspired by Spark's repartition().

        Args:
            num_partitions: Number of partitions

        Returns:
            New PipelineDataFrame with repartitioned data

        Raises:
            ValueError: If num_partitions is not positive

        Example:
            ```python
            df.repartition(100)
            ```
        """
        if not isinstance(num_partitions, int) or num_partitions <= 0:
            raise ValueError(f"num_partitions must be a positive integer, got {num_partitions}")
        
        repartitioned = self._dataset.repartition(num_partitions)
        return self._create_dataframe(repartitioned)
    
    def coalesce(self, num_partitions: int) -> "PipelineDataFrame":
        """Coalesce partitions.

        Inspired by Spark's coalesce().

        Args:
            num_partitions: Number of partitions

        Returns:
            New PipelineDataFrame with coalesced partitions

        Raises:
            ValueError: If num_partitions is not positive

        Note:
            Ray Data doesn't have coalesce(). This uses repartition() which
            may shuffle data. For true coalescing (no shuffle), consider
            using repartition() only when reducing partitions.

        Example:
            ```python
            df.coalesce(10)
            ```
        """
        if not isinstance(num_partitions, int) or num_partitions <= 0:
            raise ValueError(f"num_partitions must be a positive integer, got {num_partitions}")
        
        # Ray Data doesn't have coalesce - use repartition instead
        # Note: This may shuffle data, unlike Spark's coalesce
        coalesced = self._dataset.repartition(num_partitions)
        return self._create_dataframe(coalesced)
    
    def repartition_by_hash(
        self,
        columns: list[str],
        num_partitions: Optional[int] = None,
    ) -> "PipelineDataFrame":
        """Repartition by hash of columns.

        Inspired by Spark's repartition() with hash partitioning.

        Args:
            columns: Column names to hash partition by
            num_partitions: Number of partitions (None = auto)

        Returns:
            New PipelineDataFrame with hash-partitioned data

        Raises:
            ValueError: If columns is empty or invalid

        Example:
            ```python
            df.repartition_by_hash(["episode_id"], num_partitions=100)
            ```
        """
        if not columns or not isinstance(columns, list):
            raise ValueError("columns must be a non-empty list")
        
        validate_columns(self._dataset, columns)
        
        if num_partitions is not None:
            repartitioned = self._dataset.repartition(num_partitions, shuffle=True, partition_by_hash=columns)
        else:
            repartitioned = self._dataset.repartition(shuffle=True, partition_by_hash=columns)
        
        return self._create_dataframe(repartitioned)
    
    def checkpoint(
        self,
        path: str,
        format: str = "parquet",
    ) -> "PipelineDataFrame":
        """Checkpoint DataFrame to disk.

        Inspired by Spark's checkpoint() and Ray Data's materialization.

        Args:
            path: Checkpoint path
            format: Checkpoint format ("parquet", "json")

        Returns:
            New PipelineDataFrame from checkpoint

        Raises:
            ValueError: If path is empty or format is invalid
            OSError: If checkpoint fails

        Example:
            ```python
            df.checkpoint("s3://bucket/checkpoint/")
            ```
        """
        if not path or not isinstance(path, str):
            raise ValueError(f"path must be a non-empty string, got {path}")
        
        if format not in {"parquet", "json"}:
            raise ValueError(f"format must be 'parquet' or 'json', got {format}")
        
        # Write checkpoint
        if format == "parquet":
            self._dataset.write_parquet(path)
        else:
            self._dataset.write_json(path)
        
        # Read back from checkpoint
        import ray.data
        if format == "parquet":
            checkpointed = ray.data.read_parquet(path)
        else:
            checkpointed = ray.data.read_json(path)
        
        return self._create_dataframe(checkpointed)
    
    def collect(self) -> list[dict[str, Any]]:
        """Collect all rows into a list.

        Inspired by Spark's collect() and Polars' collect().

        Returns:
            List of row dictionaries

        Warning:
            This loads all data into memory. For large datasets, use take() or
            write to disk instead.

        Raises:
            MemoryError: If dataset is too large to fit in memory

        Example:
            ```python
            rows = df.collect()
            ```
        """
        try:
            return list(self._dataset.iter_rows())
        except MemoryError:
            logger.error("Dataset too large to collect into memory. Use take() or write to disk instead.")
            raise
        except Exception as e:
            logger.error(f"Failed to collect rows: {e}")
            raise
    
    def take(self, n: int) -> list[dict[str, Any]]:
        """Take first n rows.

        Inspired by Spark's take() and Polars' head().

        Args:
            n: Number of rows to take

        Returns:
            List of row dictionaries

        Raises:
            ValueError: If n is not positive

        Example:
            ```python
            sample = df.take(10)
            ```
        """
        if not isinstance(n, int) or n <= 0:
            raise ValueError(f"n must be a positive integer, got {n}")
        
        return list(self._dataset.take(n))
    
    def count(self) -> int:
        """Count number of rows.

        Inspired by Spark's count().

        Returns:
            Number of rows

        Warning:
            This triggers full dataset scan. For streaming datasets, consider
            using limit() or sample() instead.

        Example:
            ```python
            num_rows = df.count()
            ```
        """
        try:
            return self._dataset.count()
        except Exception as e:
            logger.error(f"Failed to count rows: {e}")
            raise
    
    def show(self, n: int = 20) -> None:
        """Show first n rows (for debugging).

        Inspired by Spark's show().

        Args:
            n: Number of rows to show

        Example:
            ```python
            df.show(10)
            ```
        """
        rows = self.take(n)
        for i, row in enumerate(rows):
            print(f"Row {i}: {row}")
    
    def to_dataset(self) -> Dataset:
        """Get underlying Ray Data Dataset.

        Returns:
            Ray Data Dataset

        Example:
            ```python
            dataset = df.to_dataset()
            ```
        """
        return self._dataset
    

