"""Performance benchmarks for pipeline components."""

import time

import pytest
import ray
from ray.data import Dataset

from pipeline.utils.gpu.etl import gpu_filter, gpu_sort


@pytest.mark.benchmark
class TestPerformanceBenchmarks:
    """Performance benchmarks."""

    @pytest.fixture(scope="class")
    def large_dataset(self):
        """Create large dataset for benchmarking."""
        if not ray.is_initialized():
            ray.init(num_cpus=4, num_gpus=0)
        
        data = [{"id": i, "value": float(i)} for i in range(100000)]
        return ray.data.from_items(data)

    def test_filter_performance(self, benchmark, large_dataset: Dataset):
        """Benchmark filter performance."""
        def filter_func(x):
            return x["value"] > 50000
        
        def run_filter():
            # Use map_batches with batch_format="pandas" instead of materializing with .to_pandas()
            def filter_batch(batch):
                return gpu_filter(batch, filter_func, num_gpus=0)
            
            filtered_dataset = large_dataset.map_batches(
                filter_batch,
                batch_format="pandas",
            )
            # Materialize only for assertion - use iter_batches for actual processing
            result_list = []
            for batch in filtered_dataset.iter_batches(batch_size=10000, prefetch_batches=0):
                result_list.append(batch)
            if result_list:
                return result_list[0]
            return None
        
        result = benchmark(run_filter)
        assert result is not None and len(result) > 0

    def test_sort_performance(self, benchmark, large_dataset: Dataset):
        """Benchmark sort performance."""
        def run_sort():
            # Use map_batches with batch_format="pandas" instead of materializing with .to_pandas()
            def sort_batch(batch):
                return gpu_sort(batch, by=["value"], num_gpus=0)
            
            sorted_dataset = large_dataset.map_batches(
                sort_batch,
                batch_format="pandas",
            )
            # Materialize only for assertion - use iter_batches for actual processing
            result_list = []
            for batch in sorted_dataset.iter_batches(batch_size=10000, prefetch_batches=0):
                result_list.append(batch)
            if result_list:
                return result_list[0]
            return None
        
        result = benchmark(run_sort)
        assert result is not None and len(result) == 100000

