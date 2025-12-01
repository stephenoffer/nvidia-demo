"""Custom datasource for binary data files.

Binary formats are commonly used in robotics for efficient storage
of sensor data, images, and custom data structures.
Follows Ray Data FileBasedDatasource API.
"""

from __future__ import annotations

import logging
import struct
from typing import TYPE_CHECKING, Any, Iterator, Optional, Union

from ray.data._internal.arrow_block import ArrowBlockBuilder
from ray.data.block import Block

from pipeline.datasources.base import FileBasedDatasource
from pipeline.exceptions import ConfigurationError, DataSourceError
from pipeline.utils.decorators import handle_errors, log_execution_time

if TYPE_CHECKING:
    import pyarrow

logger = logging.getLogger(__name__)

# Constants
_MAX_CHUNK_SIZE_BYTES = 100 * 1024 * 1024  # 100MB
_MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024 * 1024  # 10GB
_MAX_RECORDS_PER_BLOCK = 10000


class BinaryDatasource(FileBasedDatasource):
    """Custom datasource for reading binary data files.

    Supports reading binary files with configurable parsing formats.
    Common in robotics for sensor data, custom protocols, and raw data.
    """

    def __init__(
        self,
        paths: Union[str, list[str]],
        format_string: Optional[str] = None,
        record_size: Optional[int] = None,
        max_records: Optional[int] = None,
        **kwargs: Any,
    ):
        """Initialize binary datasource.

        Args:
            paths: Binary file path(s) or directory path(s)
            format_string: Optional struct format string for parsing
            record_size: Optional fixed record size in bytes
            max_records: Maximum number of records to read (None = unlimited)
            **kwargs: Additional FileBasedDatasource options

        Raises:
            ValueError: If parameters are invalid
        """
        super().__init__(paths=paths, **kwargs)
        
        # Validate parameters
        if format_string is not None and not isinstance(format_string, str):
            raise ConfigurationError(f"format_string must be str, got {type(format_string)}")
        if format_string is not None and not format_string.strip():
            raise ConfigurationError("format_string cannot be empty")
        
        if record_size is not None and not isinstance(record_size, int):
            raise ValueError(f"record_size must be int, got {type(record_size)}")
        if record_size is not None and record_size <= 0:
            raise ValueError(f"record_size must be positive, got {record_size}")
        
        if max_records is not None and max_records <= 0:
            raise ValueError(f"max_records must be positive, got {max_records}")
        
        # Validate format_string if provided
        if format_string:
            try:
                struct.calcsize(format_string)
            except struct.error as e:
                raise ValueError(f"Invalid struct format_string '{format_string}': {e}")
        
        self.format_string = format_string
        self.record_size = record_size
        self.max_records = max_records

    @log_execution_time
    @handle_errors(error_class=DataSourceError)
    def _read_stream(
        self, f: "pyarrow.NativeFile", path: str
    ) -> Iterator[Block]:
        """Read binary file and yield data blocks.

        Args:
            f: pyarrow.NativeFile handle for the file (supports S3, HDFS, etc.)
            path: Path to binary file

        Yields:
            Block objects with binary data

        Raises:
            DataSourceError: If reading fails
        """
        self._validate_file_handle(f, path)
        
        data = f.readall()
        
        if not data:
            # Empty file - return empty block following Ray Data pattern
            builder = ArrowBlockBuilder()
            builder.add({"bytes": b""})
            yield builder.build()
            return
            
            data_size = len(data)
            
            # Check file size
            if data_size > _MAX_FILE_SIZE_BYTES:
                raise DataSourceError(
                    f"Binary file {path} is {data_size} bytes, "
                    f"exceeds maximum size of {_MAX_FILE_SIZE_BYTES}"
                )

            if data_size > _MAX_CHUNK_SIZE_BYTES and (self.format_string or self.record_size):
                yield from self._process_chunked(
                    data, data_size, _MAX_CHUNK_SIZE_BYTES, path
                )
                return

            builder = ArrowBlockBuilder()
            record_count = 0

            if self.format_string and self.record_size:
                struct_size = struct.calcsize(self.format_string)
                
                # Validate struct_size matches record_size
                if struct_size != self.record_size:
                    logger.warning(
                        f"Struct size ({struct_size}) != record_size ({self.record_size}), "
                        f"using struct_size"
                    )
                
                num_records = len(data) // struct_size
                
                for i in range(num_records):
                    # Check max_records limit
                    if self.max_records is not None and record_count >= self.max_records:
                        logger.info(f"Reached max_records limit ({self.max_records})")
                        break
                    
                    offset = i * struct_size
                    if offset + struct_size > len(data):
                        logger.warning(f"Incomplete record at offset {offset}")
                        break
                    
                    try:
                        record_data = struct.unpack(
                            self.format_string, data[offset : offset + struct_size]
                        )
                        # Return clean data without metadata wrapping
                        builder.add({"data": record_data})
                        record_count += 1
                        
                        # Yield block periodically
                        if builder.num_rows() >= _MAX_RECORDS_PER_BLOCK:
                            yield builder.build()
                            builder = ArrowBlockBuilder()
                    except struct.error as e:
                        logger.warning(f"Failed to unpack record {i}: {e}")
                        continue
                        
            elif self.record_size:
                num_records = len(data) // self.record_size
                
                for i in range(num_records):
                    # Check max_records limit
                    if self.max_records is not None and record_count >= self.max_records:
                        logger.info(f"Reached max_records limit ({self.max_records})")
                        break
                    
                    offset = i * self.record_size
                    if offset + self.record_size > len(data):
                        logger.warning(f"Incomplete record at offset {offset}")
                        break
                    
                    record_data = data[offset : offset + self.record_size]
                    # Return clean data without metadata wrapping
                    builder.add({"bytes": record_data})
                    record_count += 1
                    
                    # Yield block periodically
                    if builder.num_rows() >= _MAX_RECORDS_PER_BLOCK:
                        yield builder.build()
                        builder = ArrowBlockBuilder()
            else:
                # Unstructured binary - return raw bytes following Ray Data binary datasource pattern
                if data_size > _MAX_CHUNK_SIZE_BYTES:
                    logger.warning(
                        f"Binary file {path} is {data_size} bytes, truncating to {_MAX_CHUNK_SIZE_BYTES}"
                    )
                    data = data[:_MAX_CHUNK_SIZE_BYTES]

                # Follow Ray Data binary datasource pattern: single column "bytes"
                builder.add({"bytes": data})

            if builder.num_rows() > 0:
                yield builder.build()
            
            logger.info(f"Processed {record_count} records from binary file: {path}")

    def _process_chunked(
        self, data: bytes, data_size: int, chunk_size: int, path: str
    ) -> Iterator[Block]:
        """Process large binary files in chunks.

        Args:
            data: Binary data
            data_size: Total size of data
            chunk_size: Size of each chunk
            path: File path

        Yields:
            Block objects with chunked data
        """
        if not (self.format_string or self.record_size):
            logger.warning(
                f"Cannot chunk unstructured binary file {path}, processing entire file"
            )
            builder = ArrowBlockBuilder()
            builder.add({"bytes": data[:chunk_size]})
            yield builder.build()
            return

        struct_size = (
            struct.calcsize(self.format_string)
            if self.format_string
            else self.record_size
        )
        
        if struct_size == 0:
            raise DataSourceError("struct_size cannot be zero")

        record_count = 0
        
        for chunk_start in range(0, data_size, chunk_size):
            chunk_end = min(chunk_start + chunk_size, data_size)
            chunk_data = data[chunk_start:chunk_end]

            builder = ArrowBlockBuilder()
            num_records = len(chunk_data) // struct_size

            for i in range(num_records):
                # Check max_records limit
                if self.max_records is not None and record_count >= self.max_records:
                    logger.info(f"Reached max_records limit ({self.max_records})")
                    if builder.num_rows() > 0:
                        yield builder.build()
                    return
                
                offset = i * struct_size
                if offset + struct_size > len(chunk_data):
                    break
                
                try:
                    if self.format_string:
                        record_data = struct.unpack(
                            self.format_string, chunk_data[offset : offset + struct_size]
                        )
                        # Return clean data without metadata wrapping
                        builder.add({"data": record_data})
                    else:
                        record_data = chunk_data[offset : offset + struct_size]
                        # Return clean data without metadata wrapping
                        builder.add({"bytes": record_data})
                    record_count += 1
                    
                    # Yield block periodically
                    if builder.num_rows() >= _MAX_RECORDS_PER_BLOCK:
                        yield builder.build()
                        builder = ArrowBlockBuilder()
                except struct.error as e:
                    logger.warning(f"Failed to unpack record: {e}")
                    continue

            if builder.num_rows() > 0:
                yield builder.build()


def test() -> None:
    """Test binary datasource with example data."""
    import struct
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
    
    # Create test binary file with structured data
    test_file = test_data_dir / "test_binary.bin"
    # Write some structured binary data (float, int, float)
    format_string = "fif"  # float, int, float
    with open(test_file, "wb") as f:
        for i in range(10):
            data = struct.pack(format_string, float(i), i, float(i * 2))
            f.write(data)
    
    logger.info(f"Testing BinaryDatasource with {test_file}")
    
    try:
        # Test binary datasource
        datasource = BinaryDatasource(
            paths=str(test_file),
            format_string=format_string,
            record_size=struct.calcsize(format_string),
        )
        dataset = ray.data.read_datasource(datasource)
        
        # Verify dataset
        count = 0
        for batch in dataset.iter_batches(batch_size=10, prefetch_batches=0):
            count += len(batch) if hasattr(batch, '__len__') else 1
        
        logger.info(f"BinaryDatasource test passed: loaded {count} records")
        
        # Cleanup
        if test_file.exists():
            test_file.unlink()
    except Exception as e:
        logger.error(f"BinaryDatasource test failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    test()
