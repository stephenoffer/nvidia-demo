"""Input/Output operations for PipelineDataFrame."""

from __future__ import annotations

import logging
from typing import Any, Optional, Union

from ray.data import Dataset

logger = logging.getLogger(__name__)


class IOMixin:
    """Mixin class for input/output operations."""
    
    @classmethod
    def from_paths(
        cls,
        paths: Union[str, list[str]],
        format: Optional[str] = None,
        **read_kwargs: Any,
    ) -> "PipelineDataFrame":
        """Create DataFrame from file paths.
        
        Automatically detects the appropriate reader (native Ray Data or custom datasource)
        based on file extension or directory structure. Supports explicit format specification.
        
        Args:
            paths: Single path or list of paths
            format: Optional explicit format name (e.g., "parquet", "mcap", "rosbag")
            **read_kwargs: Additional arguments for Ray Data read functions
        
        Returns:
            PipelineDataFrame instance
        """
        import ray.data
        from pipeline.utils.data.reader_registry import ReaderRegistry
        
        if not paths:
            raise ValueError("paths cannot be empty")
        
        if isinstance(paths, str):
            paths = [paths]
        
        if not isinstance(paths, list) or not all(isinstance(p, str) for p in paths):
            raise ValueError("paths must be a string or list of strings")
        
        # Validate local paths exist (skip for remote paths)
        for path in paths:
            if not path or not isinstance(path, str):
                raise ValueError(f"Invalid path: {path}")
            
            # Check local paths exist
            if not path.startswith(("s3://", "gs://", "hdfs://", "abfss://", "http://", "https://")):
                import os
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Path does not exist: {path}")
        
        # Initialize reader registry
        ReaderRegistry.initialize()
        
        # Detect and read datasets
        datasets = []
        errors = []
        
        for path in paths:
            try:
                # Detect appropriate reader
                reader = ReaderRegistry.detect_reader(path, format=format)
                
                if reader is None:
                    logger.warning(f"Could not detect reader for {path}, using binary files reader")
                    try:
                        reader = ray.data.read_binary_files
                    except AttributeError:
                        errors.append(f"No reader available for {path}")
                        continue
                
                # Read dataset using detected reader
                try:
                    ds = reader(path, **read_kwargs)
                    datasets.append(ds)
                    logger.debug(f"Successfully read {path} using {reader.__name__}")
                except Exception as e:
                    if format:
                        errors.append(f"Failed to read {path} as {format}: {e}")
                        logger.error(f"Failed to read {path} with format {format}: {e}", exc_info=True)
                        continue
                    
                    # Try fallback readers
                    logger.debug(f"Reader {reader.__name__} failed for {path}: {e}, trying fallbacks")
                    
                    fallback_readers = [
                        ("parquet", ray.data.read_parquet),
                        ("json", ray.data.read_json),
                        ("csv", ray.data.read_csv),
                        ("binary", ray.data.read_binary_files),
                    ]
                    
                    success = False
                    for fmt_name, fallback_reader in fallback_readers:
                        try:
                            ds = fallback_reader(path, **read_kwargs)
                            datasets.append(ds)
                            logger.info(f"Successfully read {path} using fallback reader: {fmt_name}")
                            success = True
                            break
                        except Exception:
                            continue
                    
                    if not success:
                        errors.append(f"Failed to read {path} with all readers: {e}")
                        logger.error(f"Failed to read {path} with all readers: {e}", exc_info=True)
                        continue
                        
            except Exception as e:
                errors.append(f"Failed to process {path}: {e}")
                logger.error(f"Failed to process path {path}: {e}", exc_info=True)
                continue
        
        if errors and not datasets:
            raise ValueError(f"Failed to read all paths. Errors: {'; '.join(errors)}")
        
        if errors:
            logger.warning(f"Some paths failed to load: {'; '.join(errors)}")
        
        if not datasets:
            raise ValueError("No datasets were successfully loaded")
        
        if len(datasets) == 1:
            return cls(datasets[0])
        else:
            # Use Dataset.union() class method (streaming-compatible)
            combined = Dataset.union(*datasets)
            return cls(combined)
    
    @classmethod
    def from_dataset(cls, dataset: Dataset) -> "PipelineDataFrame":
        """Create DataFrame from Ray Data Dataset.
        
        Args:
            dataset: Ray Data Dataset
        
        Returns:
            PipelineDataFrame instance
        """
        if not isinstance(dataset, Dataset):
            raise TypeError(f"dataset must be a Ray Data Dataset, got {type(dataset)}")
        
        return cls(dataset)
    
    def to_pipeline(
        self,
        output_path: str,
        **pipeline_kwargs: Any,
    ) -> Any:
        """Convert DataFrame to Pipeline for further processing.
        
        Args:
            output_path: Output path for pipeline results
            **pipeline_kwargs: Additional pipeline configuration
        
        Returns:
            Pipeline instance
        """
        from pipeline.api.helpers import convert_dataframe_to_pipeline
        return convert_dataframe_to_pipeline(self, output_path, **pipeline_kwargs)
    
    def write(
        self,
        path: str,
        format: Optional[str] = None,
        **write_kwargs: Any,
    ) -> None:
        """Write DataFrame to file(s) with automatic format detection.
        
        Args:
            path: Output path (file or directory)
            format: Optional explicit format name
            **write_kwargs: Additional arguments for write functions
        """
        from pipeline.utils.data.writer_registry import WriterRegistry
        
        if not path or not isinstance(path, str):
            raise ValueError(f"path must be a non-empty string, got {path}")
        
        WriterRegistry.initialize()
        writer = WriterRegistry.detect_writer(path, format=format)
        
        if writer is None:
            logger.warning(f"Could not detect writer for {path}, using parquet writer")
            writer = WriterRegistry.get_writer("parquet")
        
        if writer is None:
            raise RuntimeError(f"No writer available for path {path} and format {format}")
        
        try:
            writer(self._dataset, path, **write_kwargs)
            logger.info(f"Successfully wrote data to {path}")
        except Exception as e:
            logger.error(f"Failed to write to {path}: {e}")
            raise RuntimeError(f"Failed to write to {path}: {e}") from e
    
    # Specific write methods (delegated to write() or direct Ray Data calls)
    def write_parquet(self, path: str, **write_kwargs: Any) -> None:
        """Write to Parquet files."""
        if not path or not isinstance(path, str):
            raise ValueError(f"path must be a non-empty string, got {path}")
        self._dataset.write_parquet(path, **write_kwargs)
    
    def write_json(self, path: str, **write_kwargs: Any) -> None:
        """Write to JSON files."""
        if not path or not isinstance(path, str):
            raise ValueError(f"path must be a non-empty string, got {path}")
        self._dataset.write_json(path, **write_kwargs)
    
    def write_csv(self, path: str, **write_kwargs: Any) -> None:
        """Write to CSV files."""
        if not path or not isinstance(path, str):
            raise ValueError(f"path must be a non-empty string, got {path}")
        self._dataset.write_csv(path, **write_kwargs)
    
    def write_numpy(self, path: str, **write_kwargs: Any) -> None:
        """Write to NumPy files."""
        if not path or not isinstance(path, str):
            raise ValueError(f"path must be a non-empty string, got {path}")
        self._dataset.write_numpy(path, **write_kwargs)
    
    def write_tfrecords(self, path: str, **write_kwargs: Any) -> None:
        """Write to TFRecord files."""
        if not path or not isinstance(path, str):
            raise ValueError(f"path must be a non-empty string, got {path}")
        self._dataset.write_tfrecords(path, **write_kwargs)
    
    def write_images(self, path: str, **write_kwargs: Any) -> None:
        """Write images to directory."""
        if not path or not isinstance(path, str):
            raise ValueError(f"path must be a non-empty string, got {path}")
        self._dataset.write_images(path, **write_kwargs)

