"""WindowedDataFrame class for window function operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
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


class WindowedDataFrame:
    """Windowed DataFrame for window functions.

    Inspired by Spark's window functions.
    """

    def __init__(
        self,
        dataset: Dataset,
        partition_by: list[str],
        order_by: list[str],
    ):
        """Initialize WindowedDataFrame.

        Args:
            dataset: Underlying Ray Data Dataset
            partition_by: Columns to partition by
            order_by: Columns to order by
        """
        self._dataset = dataset
        self.partition_by = partition_by
        self.order_by = order_by

    def row_number(self) -> PipelineDataFrame:
        """Add row number within each partition.

        Returns:
            PipelineDataFrame with row_number column
        """
        def _add_row_number(batch: dict[str, Any]) -> dict[str, Any]:
            """Internal: Add row number to batch."""
            if not batch:
                return {}
            
            # Convert to pandas for easier window operations
            df = pd.DataFrame(batch)
            
            # Group by partition columns and add row number
            if self.partition_by:
                df['row_number'] = df.groupby(self.partition_by).cumcount() + 1
            else:
                df['row_number'] = range(1, len(df) + 1)
            
            # Convert back to dict format
            return df.to_dict(orient="list")
        
        windowed = self._dataset.map_batches(_add_row_number, batch_format="pandas")
        PipelineDataFrameClass = _get_pipeline_dataframe_class()
        return PipelineDataFrameClass(windowed)
    
    def add_row_number(self) -> PipelineDataFrame:
        """Add row number (alias for row_number).

        Returns:
            PipelineDataFrame with row_number column
        """
        return self.row_number()

