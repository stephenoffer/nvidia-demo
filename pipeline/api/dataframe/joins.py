"""Join and union operations for PipelineDataFrame."""

from __future__ import annotations

import logging
from typing import Union

from pipeline.api.dataframe.shared import validate_columns

logger = logging.getLogger(__name__)


class JoinsMixin:
    """Mixin class for join and union operations."""
    
    def join(
        self,
        other: "PipelineDataFrame",
        on: Union[str, list[str]],
        how: str = "inner",
    ) -> "PipelineDataFrame":
        """Join with another DataFrame.

        Inspired by Spark's join() and Pandas' merge().

        Args:
            other: Other PipelineDataFrame to join with
            on: Column name(s) to join on
            how: Join type ("inner", "left", "right", "outer")

        Returns:
            New PipelineDataFrame with joined data

        Raises:
            TypeError: If other is not a PipelineDataFrame
            ValueError: If on is empty or how is invalid

        Example:
            ```python
            df1.join(df2, on="episode_id", how="inner")
            ```
        """
        if not isinstance(other, type(self)):
            raise TypeError(f"other must be a {type(self).__name__}, got {type(other)}")
        
        if isinstance(on, str):
            on = [on]
        
        if not on:
            raise ValueError("on must specify at least one column")
        
        valid_joins = {"inner", "left", "right", "outer"}
        if how not in valid_joins:
            raise ValueError(f"how must be one of {valid_joins}, got {how}")
        
        # Validate join keys exist
        self._validate_join_keys(self._dataset, other._dataset, on)
        
        joined = self._dataset.join(
            other._dataset,
            keys=on,
            how=how,
        )
        return self._create_dataframe(joined)
    
    def union(self, *others: "PipelineDataFrame") -> "PipelineDataFrame":
        """Union with other DataFrames (concatenate rows).

        Inspired by Spark's union() and Pandas' pd.concat().
        Combines rows from multiple DataFrames into one.

        Args:
            *others: Other PipelineDataFrames to union with

        Returns:
            New PipelineDataFrame with unioned data

        Example:
            ```python
            # Method chaining
            df1.union(df2, df3)
            
            # Or use + operator
            combined = df1 + df2 + df3
            
            # Or use | operator
            combined = df1 | df2 | df3
            ```
        """
        if not others:
            return self
        
        datasets = [self._dataset] + [df._dataset for df in others]
        
        # Use Ray Data's union for better performance (streaming-compatible)
        if len(datasets) == 1:
            return self._create_dataframe(datasets[0])
        elif len(datasets) == 2:
            # Simple union
            unioned = datasets[0].union(datasets[1])
            return self._create_dataframe(unioned)
        else:
            # Multiple datasets - use Ray Data's union class method
            import ray.data
            unioned = ray.data.union(*datasets)
            return self._create_dataframe(unioned)
    
    # Internal helper methods
    def _validate_join_keys(
        self,
        left_dataset,
        right_dataset,
        keys: list[str],
    ) -> None:
        """Internal: Validate join keys exist in both datasets."""
        try:
            sample1 = left_dataset.take(1)
            sample2 = right_dataset.take(1)
            if sample1 and sample2:
                keys1 = set(sample1[0].keys())
                keys2 = set(sample2[0].keys())
                missing1 = set(keys) - keys1
                missing2 = set(keys) - keys2
                if missing1:
                    raise ValueError(f"Join keys not found in left DataFrame: {missing1}")
                if missing2:
                    raise ValueError(f"Join keys not found in right DataFrame: {missing2}")
        except Exception:
            logger.debug("Could not validate join keys, proceeding anyway")
    

