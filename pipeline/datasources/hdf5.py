"""Custom datasource for HDF5 files.

HDF5 is commonly used for storing large-scale scientific and robotics data.
Follows Ray Data FileBasedDatasource API.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, Iterator, List, Optional, Union

import numpy as np
from ray.data._internal.arrow_block import ArrowBlockBuilder
from ray.data.block import Block

from pipeline.datasources.base import FileBasedDatasource
from pipeline.exceptions import ConfigurationError, DataSourceError, ConfigurationError
from pipeline.utils.decorators import handle_errors, log_execution_time

if TYPE_CHECKING:
    import pyarrow

logger = logging.getLogger(__name__)

# Constants
_MAX_DATASET_SIZE_BYTES = 100 * 1024 * 1024  # 100MB
_MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024 * 1024  # 10GB
_MAX_ARRAY_SIZE_FOR_LIST = 1_000_000  # Don't convert arrays > 1M elements to list


class HDF5Datasource(FileBasedDatasource):
    """Custom datasource for reading HDF5 files.

    HDF5 (Hierarchical Data Format) is widely used in robotics for storing
    large-scale sensor data, trajectories, and simulation results.
    """

    def __init__(
        self,
        paths: Union[str, List[str]],
        datasets: Optional[List[str]] = None,
        max_datasets: Optional[int] = None,
        **kwargs: Any,
    ):
        """Initialize HDF5 datasource.

        Args:
            paths: HDF5 file path(s) or directory path(s)
            datasets: List of dataset names to read (None = all datasets)
            max_datasets: Maximum number of datasets to read (None = unlimited)
            **kwargs: Additional FileBasedDatasource options

        Raises:
            ValueError: If parameters are invalid
        """
        super().__init__(paths=paths, **kwargs)
        
        # Validate parameters
        if datasets is not None and not isinstance(datasets, list):
            raise ConfigurationError(f"datasets must be a list, got {type(datasets)}")
        if datasets is not None and len(datasets) == 0:
            raise ConfigurationError("datasets cannot be an empty list")
        
        if max_datasets is not None and max_datasets <= 0:
            raise ConfigurationError(f"max_datasets must be positive, got {max_datasets}")
        
        self.datasets = datasets
        self.max_datasets = max_datasets

    @log_execution_time
    @handle_errors(error_class=DataSourceError)
    def _read_stream(
        self, f: "pyarrow.NativeFile", path: str
    ) -> Iterator[Block]:
        """Read HDF5 file and yield dataset blocks.

        Args:
            f: pyarrow.NativeFile handle (unused, HDF5 requires direct file access)
            path: Path to HDF5 file

        Yields:
            Block objects (pyarrow.Table) with HDF5 dataset data

        Raises:
            DataSourceError: If reading fails

        Note:
            HDF5 requires direct file access, so the file handle is not used.
            For cloud storage, files must be copied locally first.
        """
        self._validate_file_handle(f, path)
        
        # Validate file exists (for local files)
        if not path.startswith(("s3://", "gs://", "hdfs://", "abfss://")):
            if not os.path.exists(path):
                raise DataSourceError(f"HDF5 file does not exist: {path}")
            
            if not os.path.isfile(path):
                raise DataSourceError(f"HDF5 path is not a file: {path}")
            
            # Check file size
            file_size = os.path.getsize(path)
            if file_size > _MAX_FILE_SIZE_BYTES:
                raise DataSourceError(
                    f"HDF5 file {path} is {file_size} bytes, "
                    f"exceeds maximum size of {_MAX_FILE_SIZE_BYTES}"
                )
        
        try:
            import h5py
        except ImportError:
            raise DataSourceError(
                "h5py library not installed. Install with: pip install h5py"
            ) from None

        hf = None
        try:
            hf = h5py.File(path, "r")
            
            # Get datasets to read
            if self.datasets:
                datasets_to_read = [d for d in self.datasets if d in hf]
                missing_datasets = set(self.datasets) - set(datasets_to_read)
                if missing_datasets:
                    logger.warning(
                        f"Datasets not found in {path}: {missing_datasets}"
                    )
            else:
                datasets_to_read = list(hf.keys())
            
            # Apply max_datasets limit
            if self.max_datasets is not None:
                datasets_to_read = datasets_to_read[:self.max_datasets]
            
            if not datasets_to_read:
                logger.warning(f"No datasets found in HDF5 file: {path}")
                return
            
            logger.info(f"Reading {len(datasets_to_read)} datasets from {path}")

            for dataset_name in datasets_to_read:
                try:
                    if dataset_name not in hf:
                        logger.warning(f"Dataset '{dataset_name}' not found in {path}")
                        continue

                    dataset = hf[dataset_name]
                    
                    if not isinstance(dataset, h5py.Dataset):
                        logger.debug(f"Skipping non-dataset '{dataset_name}' (type: {type(dataset)})")
                        continue

                    dataset_size = dataset.size * dataset.dtype.itemsize

                    if dataset_size > _MAX_DATASET_SIZE_BYTES:
                        logger.warning(
                            f"Dataset '{dataset_name}' in {path} is {dataset_size} bytes, "
                            f"processing in chunks"
                        )
                        yield from self._read_dataset_chunked(
                            dataset, dataset_name, path, _MAX_DATASET_SIZE_BYTES
                        )
                    else:
                        try:
                            data = dataset[:]
                            builder = ArrowBlockBuilder()
                            
                            # Convert data to serializable format
                            data_serializable = self._convert_to_serializable(data)
                            
                            # Get attributes safely
                            attrs = {}
                            try:
                                attrs = dict(dataset.attrs)
                            except Exception as e:
                                logger.warning(f"Failed to read attributes for {dataset_name}: {e}")
                            
                            # Return clean data without metadata wrapping
                            # Ray Data handles path via include_paths parameter
                            item = {
                                "dataset_name": dataset_name,
                                "shape": list(dataset.shape),
                                "dtype": str(dataset.dtype),
                                "data": data_serializable,
                                "attributes": attrs,
                            }
                            builder.add(item)
                            yield builder.build()
                        except Exception as e:
                            logger.error(f"Failed to read dataset '{dataset_name}': {e}", exc_info=True)
                            # Raise exception instead of wrapping in block
                            raise DataSourceError(
                                f"Failed to read dataset '{dataset_name}' from {path}: {e}"
                            ) from e

                except Exception as e:
                    logger.error(f"Error processing dataset '{dataset_name}': {e}", exc_info=True)
                    # Continue with next dataset
                    continue

        except (OSError, IOError, ValueError, KeyError) as e:
            logger.error(f"Error reading HDF5 file {path}: {e}", exc_info=True)
            # Re-raise exception instead of wrapping in block
            raise DataSourceError(f"Failed to read HDF5 file {path}: {e}") from e
        finally:
            if hf is not None:
                try:
                    hf.close()
                except Exception as e:
                    logger.warning(f"Error closing HDF5 file: {e}")

    def _convert_to_serializable(self, data: Any) -> Any:
        """Convert numpy array to serializable format.

        Args:
            data: NumPy array or scalar

        Returns:
            Serializable data (list, scalar, or string representation)
        """
        if isinstance(data, np.ndarray):
            # For large arrays, don't convert to list (too slow)
            if data.size > _MAX_ARRAY_SIZE_FOR_LIST:
                return {
                    "type": "array",
                    "shape": list(data.shape),
                    "dtype": str(data.dtype),
                    "size": data.size,
                    "data_truncated": True,
                }
            
            # Convert to list for small arrays
            try:
                return data.tolist()
            except Exception as e:
                logger.warning(f"Failed to convert array to list: {e}")
                return str(data)
        elif isinstance(data, (np.integer, np.floating)):
            return float(data) if isinstance(data, np.floating) else int(data)
        elif isinstance(data, np.bool_):
            return bool(data)
        else:
            return data

    def _read_dataset_chunked(
        self, dataset: Any, dataset_name: str, path: str, max_chunk_size: int
    ) -> Iterator[Block]:
        """Read large HDF5 dataset in chunks.

        Args:
            dataset: HDF5 dataset object
            dataset_name: Name of the dataset
            path: File path
            max_chunk_size: Maximum chunk size in bytes

        Yields:
            Block objects with chunked dataset data
        """
        if dataset is None:
            raise DataSourceError(f"Dataset '{dataset_name}' is None")
        
        shape = dataset.shape
        dtype_size = dataset.dtype.itemsize

        if len(shape) == 0:
            # Scalar dataset
            builder = ArrowBlockBuilder()
            try:
                scalar_value = dataset[()]
                attrs = {}
                try:
                    attrs = dict(dataset.attrs)
                except Exception:
                    pass
                
                # Return clean data without metadata wrapping
                builder.add(
                    {
                        "dataset_name": dataset_name,
                        "shape": [],
                        "dtype": str(dataset.dtype),
                        "data": self._convert_to_serializable(scalar_value),
                        "attributes": attrs,
                    }
                )
                yield builder.build()
            except Exception as e:
                logger.error(f"Failed to read scalar dataset '{dataset_name}': {e}")
            return

        # Calculate chunking parameters
        try:
            elements_per_chunk = max_chunk_size // dtype_size
            if elements_per_chunk == 0:
                elements_per_chunk = 1  # At least 1 element
            
            total_elements = np.prod(shape)
            chunk_size = min(elements_per_chunk, total_elements)

            first_dim_size = shape[0]
            if first_dim_size == 0:
                logger.warning(f"Dataset '{dataset_name}' has zero-size first dimension")
                return
            
            chunk_first_dim = min(chunk_size // max(1, np.prod(shape[1:])), first_dim_size)
            if chunk_first_dim == 0:
                chunk_first_dim = 1  # At least 1 element
            
            # Get attributes once
            attrs = {}
            try:
                attrs = dict(dataset.attrs)
            except Exception:
                pass

            for chunk_start in range(0, first_dim_size, chunk_first_dim):
                chunk_end = min(chunk_start + chunk_first_dim, first_dim_size)
                
                try:
                    if len(shape) == 1:
                        chunk_data = dataset[chunk_start:chunk_end]
                    else:
                        indices = tuple(
                            slice(chunk_start, chunk_end) if i == 0 else slice(None)
                            for i in range(len(shape))
                        )
                        chunk_data = dataset[indices]

                    builder = ArrowBlockBuilder()
                    # Return clean data without metadata wrapping
                    item = {
                        "dataset_name": dataset_name,
                        "shape": list(chunk_data.shape),
                        "dtype": str(dataset.dtype),
                        "data": self._convert_to_serializable(chunk_data),
                        "attributes": attrs,
                        "chunk_index": chunk_start // chunk_first_dim,
                        "chunk_range": [chunk_start, chunk_end],
                    }
                    builder.add(item)
                    yield builder.build()
                except Exception as e:
                    logger.error(
                        f"Failed to read chunk [{chunk_start}:{chunk_end}] "
                        f"of dataset '{dataset_name}': {e}"
                    )
                    # Raise exception instead of silently continuing
                    raise DataSourceError(
                        f"Failed to read chunk [{chunk_start}:{chunk_end}] "
                        f"of dataset '{dataset_name}' from {path}: {e}"
                    ) from e
        except Exception as e:
            logger.error(f"Failed to chunk dataset '{dataset_name}': {e}", exc_info=True)
            raise DataSourceError(f"Failed to chunk dataset '{dataset_name}': {e}") from e


def test() -> None:
    """Test HDF5 datasource with example data."""
    from pathlib import Path
    
    # Initialize Ray if not already initialized
    try:
        import ray
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
    except Exception:
        pass
    
    # Test data directory
    test_data_dir = Path(__file__).parent.parent.parent / "examples" / "data"
    test_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Create test HDF5 file
    test_file = test_data_dir / "test_data.h5"
    
    try:
        import h5py
        import numpy as np
        
        # Create a simple HDF5 file
        with h5py.File(test_file, "w") as f:
            f.create_dataset("sensor_data", data=np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
            f.create_dataset("timestamps", data=np.array([100, 200, 300, 400, 500]))
        
        logger.info(f"Testing HDF5Datasource with {test_file}")
        
        # Test HDF5 datasource
        datasource = HDF5Datasource(paths=str(test_file))
        dataset = ray.data.read_datasource(datasource)
        
        # Verify dataset
        count = 0
        for batch in dataset.iter_batches(batch_size=10, prefetch_batches=0):
            count += len(batch) if hasattr(batch, '__len__') else 1
        
        logger.info(f"HDF5Datasource test passed: loaded {count} datasets")
        
        # Cleanup
        if test_file.exists():
            test_file.unlink()
    except ImportError:
        logger.warning("HDF5 test skipped: h5py library not installed")
    except Exception as e:
        logger.error(f"HDF5Datasource test failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    test()
