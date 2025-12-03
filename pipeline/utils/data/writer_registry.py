"""Writer registry for automatic detection and selection of data writers.

Supports both native Ray Data writers and custom datasources.
Provides automatic format detection and explicit writer specification.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class WriterRegistry:
    """Registry for data writers (native Ray Data and custom datasources).
    
    Maps file extensions and format names to appropriate writers.
    Supports automatic detection and explicit specification.
    """
    
    # Native Ray Data writers mapped by extension
    _native_writers: dict[str, str] = {}  # Maps extension to method name
    
    # Custom writers mapped by extension
    _custom_writers: dict[str, Callable] = {}
    
    # Format names mapped to writers (for explicit specification)
    _format_writers: dict[str, Callable] = {}
    
    @classmethod
    def _initialize_native_writers(cls) -> None:
        """Initialize native Ray Data writers."""
        if cls._native_writers:
            return  # Already initialized
        
        # Parquet formats
        cls._native_writers[".parquet"] = "write_parquet"
        cls._native_writers[".pq"] = "write_parquet"
        
        # CSV formats
        cls._native_writers[".csv"] = "write_csv"
        cls._native_writers[".tsv"] = "write_csv"
        
        # JSON formats
        cls._native_writers[".json"] = "write_json"
        cls._native_writers[".jsonl"] = "write_json"
        
        # NumPy formats
        cls._native_writers[".npy"] = "write_numpy"
        cls._native_writers[".npz"] = "write_numpy"
        
        # TFRecord formats
        cls._native_writers[".tfrecord"] = "write_tfrecords"
        cls._native_writers[".tfrecords"] = "write_tfrecords"
        
        # Image formats (write_images writes to directory)
        cls._native_writers[".jpg"] = "write_images"
        cls._native_writers[".jpeg"] = "write_images"
        cls._native_writers[".png"] = "write_images"
        cls._native_writers[".tiff"] = "write_images"
        cls._native_writers[".tif"] = "write_images"
        cls._native_writers[".bmp"] = "write_images"
        cls._native_writers[".gif"] = "write_images"
        cls._native_writers[".webp"] = "write_images"
        
        # Register format names
        cls._format_writers["parquet"] = lambda dataset, path, **kwargs: dataset.write_parquet(path, **kwargs)
        cls._format_writers["csv"] = lambda dataset, path, **kwargs: dataset.write_csv(path, **kwargs)
        cls._format_writers["json"] = lambda dataset, path, **kwargs: dataset.write_json(path, **kwargs)
        cls._format_writers["jsonl"] = lambda dataset, path, **kwargs: dataset.write_json(path, **kwargs)
        cls._format_writers["numpy"] = lambda dataset, path, **kwargs: dataset.write_numpy(path, **kwargs)
        cls._format_writers["tfrecords"] = lambda dataset, path, **kwargs: dataset.write_tfrecords(path, **kwargs)
        cls._format_writers["images"] = lambda dataset, path, **kwargs: dataset.write_images(path, **kwargs)
        cls._format_writers["sql"] = lambda dataset, path, **kwargs: dataset.write_sql(path, **kwargs)
        cls._format_writers["bigquery"] = lambda dataset, path, **kwargs: dataset.write_bigquery(path, **kwargs)
        cls._format_writers["clickhouse"] = lambda dataset, path, **kwargs: dataset.write_clickhouse(path, **kwargs)
        cls._format_writers["mongo"] = lambda dataset, path, **kwargs: dataset.write_mongo(path, **kwargs)
        cls._format_writers["snowflake"] = lambda dataset, path, **kwargs: dataset.write_snowflake(path, **kwargs)
        cls._format_writers["iceberg"] = lambda dataset, path, **kwargs: dataset.write_iceberg(path, **kwargs)
        cls._format_writers["lance"] = lambda dataset, path, **kwargs: dataset.write_lance(path, **kwargs)
        cls._format_writers["webdataset"] = lambda dataset, path, **kwargs: dataset.write_webdataset(path, **kwargs)
        
        logger.debug("Initialized native Ray Data writers")
    
    @classmethod
    def _initialize_custom_writers(cls) -> None:
        """Initialize custom datasource writers."""
        if cls._custom_writers:
            return  # Already initialized
        
        # Note: Most custom datasources are read-only (MCAP, ROS bags, etc.)
        # Custom writers would be implemented here for formats that need special handling
        
        # HDF5 format (custom writer needed)
        try:
            def write_hdf5(dataset, path: str, **kwargs: Any) -> None:
                """Write dataset to HDF5 format."""
                import h5py
                import numpy as np
                import pandas as pd
                
                # Convert dataset to pandas in batches
                h5_file = h5py.File(path, 'w')
                try:
                    batch_size = kwargs.get('batch_size', 10000)
                    batch_idx = 0
                    
                    for batch in dataset.iter_batches(batch_size=batch_size, batch_format="pandas"):
                        if isinstance(batch, dict):
                            batch = pd.DataFrame(batch)
                        
                        # Write each column as a dataset
                        for col in batch.columns:
                            group_name = f"batch_{batch_idx}/{col}"
                            data = batch[col].values
                            
                            # Handle different data types
                            if isinstance(data[0], (list, np.ndarray)):
                                # Variable-length arrays
                                h5_file.create_dataset(
                                    group_name,
                                    data=data,
                                    compression="gzip",
                                    compression_opts=4,
                                )
                            else:
                                # Scalar values
                                h5_file.create_dataset(
                                    group_name,
                                    data=data,
                                    compression="gzip",
                                    compression_opts=4,
                                )
                        
                        batch_idx += 1
                finally:
                    h5_file.close()
            
            cls._custom_writers[".h5"] = write_hdf5
            cls._custom_writers[".hdf5"] = write_hdf5
            cls._format_writers["hdf5"] = write_hdf5
        except ImportError:
            logger.debug("HDF5 writer not available (h5py not installed)")
        
        # MessagePack format (custom writer)
        try:
            import msgpack
            
            def write_msgpack(dataset, path: str, **kwargs: Any) -> None:
                """Write dataset to MessagePack format."""
                import os
                
                # Create directory if needed
                os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
                
                with open(path, 'wb') as f:
                    batch_size = kwargs.get('batch_size', 10000)
                    for batch in dataset.iter_batches(batch_size=batch_size):
                        if isinstance(batch, dict):
                            # Convert dict batch to list of dicts
                            keys = list(batch.keys())
                            num_items = len(batch[keys[0]]) if keys else 0
                            items = [
                                {key: batch[key][i] for key in keys}
                                for i in range(num_items)
                            ]
                        else:
                            items = batch if isinstance(batch, list) else batch.to_dict('records')
                        
                        for item in items:
                            msgpack.pack(item, f)
            
            cls._custom_writers[".msgpack"] = write_msgpack
            cls._custom_writers[".mpk"] = write_msgpack
            cls._format_writers["msgpack"] = write_msgpack
        except ImportError:
            logger.debug("MessagePack writer not available (msgpack not installed)")
        
        # Protobuf format (custom writer - requires schema)
        try:
            def write_protobuf(dataset, path: str, **kwargs: Any) -> None:
                """Write dataset to Protobuf format."""
                import os
                
                # Protobuf writing requires a schema definition
                # This is a placeholder - users should provide their own protobuf writer
                logger.warning(
                    "Protobuf writing requires a schema definition. "
                    "Please use a custom writer with your protobuf schema."
                )
                raise NotImplementedError("Protobuf writing requires schema definition")
            
            cls._custom_writers[".pb"] = write_protobuf
            cls._custom_writers[".protobuf"] = write_protobuf
            cls._format_writers["protobuf"] = write_protobuf
        except Exception:
            logger.debug("Protobuf writer not available")
        
        logger.debug("Initialized custom datasource writers")
    
    @classmethod
    def initialize(cls) -> None:
        """Initialize all writers (native and custom)."""
        cls._initialize_native_writers()
        cls._initialize_custom_writers()
    
    @classmethod
    def detect_writer(
        cls,
        path: str,
        format: Optional[str] = None,
    ) -> Optional[Callable]:
        """Detect appropriate writer for a path.
        
        Args:
            path: File or directory path
            format: Optional explicit format name (overrides auto-detection)
        
        Returns:
            Writer function or None if not found
        """
        cls.initialize()
        
        # Explicit format specification takes precedence
        if format:
            format_lower = format.lower()
            if format_lower in cls._format_writers:
                writer = cls._format_writers[format_lower]
                logger.debug(f"Using explicit format writer: {format}")
                return writer
            else:
                logger.warning(f"Unknown format '{format}', falling back to auto-detection")
        
        # Auto-detect from extension
        path_lower = path.lower()
        
        # Check for directory patterns first
        if "/parquet/" in path_lower or path_lower.endswith("/parquet"):
            return cls._format_writers.get("parquet")
        if "/json/" in path_lower or path_lower.endswith("/json"):
            return cls._format_writers.get("json")
        if "/csv/" in path_lower or path_lower.endswith("/csv"):
            return cls._format_writers.get("csv")
        if "/images/" in path_lower or path_lower.endswith("/images"):
            return cls._format_writers.get("images")
        if "/numpy/" in path_lower or path_lower.endswith("/numpy"):
            return cls._format_writers.get("numpy")
        
        # Check file extensions (try longest first for multi-part extensions)
        extensions = [
            ".tar.gz", ".tar.bz2", ".tar.bz", ".tar.xz",
            ".tgz", ".tbz2", ".tbz",
        ]
        for ext in extensions:
            if path_lower.endswith(ext):
                writer = cls._custom_writers.get(ext) or cls._get_native_writer(ext)
                if writer:
                    logger.debug(f"Detected writer from extension: {ext}")
                    return writer
        
        # Check single-part extensions
        from pathlib import Path
        path_obj = Path(path)
        suffix = path_obj.suffix.lower()
        
        # Try custom writers first (they override native for same extension)
        if suffix in cls._custom_writers:
            logger.debug(f"Detected custom writer from extension: {suffix}")
            return cls._custom_writers[suffix]
        
        # Try native writers
        if suffix in cls._native_writers:
            logger.debug(f"Detected native writer from extension: {suffix}")
            return cls._get_native_writer(suffix)
        
        # Fallback: try parquet writer
        logger.debug(f"No specific writer found for {suffix}, using parquet writer")
        return cls._format_writers.get("parquet")
    
    @classmethod
    def _get_native_writer(cls, extension: str) -> Optional[Callable]:
        """Get native Ray Data writer function for extension.
        
        Args:
            extension: File extension
        
        Returns:
            Writer function or None
        """
        method_name = cls._native_writers.get(extension)
        if not method_name:
            return None
        
        # Return a lambda that calls the method on the dataset
        def writer(dataset, path: str, **kwargs: Any) -> None:
            method = getattr(dataset, method_name)
            return method(path, **kwargs)
        
        return writer
    
    @classmethod
    def get_writer(cls, format: str) -> Optional[Callable]:
        """Get writer by format name.
        
        Args:
            format: Format name (e.g., "parquet", "json", "hdf5")
        
        Returns:
            Writer function or None if not found
        """
        cls.initialize()
        return cls._format_writers.get(format.lower())
    
    @classmethod
    def list_formats(cls) -> dict[str, list[str]]:
        """List all supported formats.
        
        Returns:
            Dictionary with 'native' and 'custom' format lists
        """
        cls.initialize()
        
        native_formats = [
            "parquet", "csv", "json", "jsonl", "numpy", "tfrecords",
            "images", "sql", "bigquery", "clickhouse", "mongo",
            "snowflake", "iceberg", "lance", "webdataset",
        ]
        
        custom_formats = [
            fmt for fmt in cls._format_writers.keys()
            if fmt not in native_formats
        ]
        
        return {
            "native": sorted(native_formats),
            "custom": sorted(custom_formats),
        }
    
    @classmethod
    def register_custom_writer(
        cls,
        extension: str,
        writer: Callable,
        format_name: Optional[str] = None,
    ) -> None:
        """Register a custom writer.
        
        Args:
            extension: File extension (e.g., ".custom")
            writer: Writer function that takes (dataset, path, **kwargs)
            format_name: Optional format name for explicit specification
        """
        cls.initialize()
        cls._custom_writers[extension.lower()] = writer
        if format_name:
            cls._format_writers[format_name.lower()] = writer
        logger.info(f"Registered custom writer for extension {extension}")


# Convenience functions
def detect_writer(path: str, format: Optional[str] = None) -> Optional[Callable]:
    """Detect appropriate writer for a path.
    
    Args:
        path: File or directory path
        format: Optional explicit format name
    
    Returns:
        Writer function or None if not found
    """
    return WriterRegistry.detect_writer(path, format)


def get_writer(format: str) -> Optional[Callable]:
    """Get writer by format name.
    
    Args:
        format: Format name
    
    Returns:
        Writer function or None if not found
    """
    return WriterRegistry.get_writer(format)


def list_formats() -> dict[str, list[str]]:
    """List all supported formats.
    
    Returns:
        Dictionary with 'native' and 'custom' format lists
    """
    return WriterRegistry.list_formats()

