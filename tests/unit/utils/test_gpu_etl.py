"""Unit tests for GPU ETL operations."""

import pandas as pd
import pytest

from pipeline.utils.gpu.etl import (
    gpu_aggregate,
    gpu_filter,
    gpu_join,
    gpu_map_batches_transform,
    gpu_sort,
)


class TestGPUMapBatchesTransform:
    """Test GPU batch transformation."""

    def test_cpu_fallback(self):
        """Test CPU fallback when GPU unavailable."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        
        def transform(x):
            x["c"] = x["a"] + x["b"]
            return x
        
        result = gpu_map_batches_transform(df, transform, num_gpus=0)
        assert "c" in result.columns
        assert list(result["c"]) == [5, 7, 9]


class TestGPUJoin:
    """Test GPU join operations."""

    def test_cpu_fallback(self):
        """Test CPU fallback for joins."""
        left = pd.DataFrame({"id": [1, 2, 3], "value": [10, 20, 30]})
        right = pd.DataFrame({"id": [2, 3, 4], "other": [200, 300, 400]})
        
        result = gpu_join(left, right, on=["id"], how="inner", num_gpus=0)
        assert len(result) == 2
        assert list(result["id"]) == [2, 3]


class TestGPUFilter:
    """Test GPU filter operations."""

    def test_cpu_fallback(self):
        """Test CPU fallback for filtering."""
        df = pd.DataFrame({"value": [1, 2, 3, 4, 5]})
        
        def filter_func(x):
            return x["value"] > 2
        
        result = gpu_filter(df, filter_func, num_gpus=0)
        assert len(result) == 3
        assert all(result["value"] > 2)


class TestGPUSort:
    """Test GPU sort operations."""

    def test_cpu_fallback(self):
        """Test CPU fallback for sorting."""
        df = pd.DataFrame({"value": [3, 1, 4, 2, 5]})
        
        result = gpu_sort(df, by=["value"], ascending=True, num_gpus=0)
        assert list(result["value"]) == [1, 2, 3, 4, 5]


class TestGPUAggregate:
    """Test GPU aggregation operations."""

    def test_cpu_fallback(self):
        """Test CPU fallback for aggregation."""
        df = pd.DataFrame({"group": ["A", "A", "B", "B"], "value": [1, 2, 3, 4]})
        
        result = gpu_aggregate(
            df,
            agg_dict={"value": "sum"},
            groupby=["group"],
            num_gpus=0,
        )
        assert len(result) == 2
        assert set(result["group"]) == {"A", "B"}

