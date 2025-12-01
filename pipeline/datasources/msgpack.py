"""Custom datasource for MessagePack files.

MessagePack is a binary serialization format that's more compact than JSON,
commonly used in robotics for efficient data storage and transmission.
Follows Ray Data FileBasedDatasource API.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Iterator, Optional, Union

from ray.data._internal.arrow_block import ArrowBlockBuilder
from ray.data.block import Block

from pipeline.datasources.base import FileBasedDatasource
from pipeline.exceptions import DataSourceError, ConfigurationError
from pipeline.utils.decorators import handle_errors, log_execution_time

if TYPE_CHECKING:
    import pyarrow

logger = logging.getLogger(__name__)

# Constants
_DEFAULT_CHUNK_SIZE = 64 * 1024  # 64KB
_MAX_RECORDS_PER_BLOCK = 1000
_MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024 * 1024  # 10GB


class MessagePackDatasource(FileBasedDatasource):
    """Custom datasource for reading MessagePack files.

    MessagePack is a binary serialization format that's more compact
    than JSON, commonly used in robotics for efficient data storage.
    """

    def __init__(
        self,
        paths: Union[str, list[str]],
        max_records: Optional[int] = None,
        **kwargs: Any,
    ):
        """Initialize MessagePack datasource.

        Args:
            paths: MessagePack file path(s) or directory path(s)
            max_records: Maximum number of records to read (None = unlimited)
            **kwargs: Additional FileBasedDatasource options

        Raises:
            ValueError: If parameters are invalid
        """
        super().__init__(paths=paths, **kwargs)
        
        # Validate parameters
        if max_records is not None and max_records <= 0:
            raise ConfigurationError(f"max_records must be positive, got {max_records}")
        
        self.max_records = max_records

    @log_execution_time
    @handle_errors(error_class=DataSourceError)
    def _read_stream(
        self, f: "pyarrow.NativeFile", path: str
    ) -> Iterator[Block]:
        """Read MessagePack file and yield data blocks.

        Args:
            f: pyarrow.NativeFile handle for the file (supports S3, HDFS, etc.)
            path: Path to MessagePack file

        Yields:
            Block objects with deserialized MessagePack data

        Raises:
            DataSourceError: If reading fails
        """
        self._validate_file_handle(f, path)
        
        try:
            import msgpack  # https://msgpack.org/
        except ImportError:
            raise DataSourceError(
                "msgpack library not installed. Install with: pip install msgpack"
            ) from None

        builder = ArrowBlockBuilder()

        # Use provided file handle to support S3, HDFS, etc.
        # Read in chunks to avoid loading entire file into memory
        unpacker = msgpack.Unpacker(raw=False)

        record_count = 0
        total_bytes_read = 0

        while True:
            # Check file size limit
            if total_bytes_read >= _MAX_FILE_SIZE_BYTES:
                raise DataSourceError(
                    f"MessagePack file {path} exceeds {_MAX_FILE_SIZE_BYTES} bytes"
                )
            
            chunk = f.read(_DEFAULT_CHUNK_SIZE)
            if not chunk:
                break
            
            total_bytes_read += len(chunk)
            unpacker.feed(chunk)
            
            try:
                for data in unpacker:
                    # Check max_records limit
                    if self.max_records is not None and record_count >= self.max_records:
                        logger.info(f"Reached max_records limit ({self.max_records})")
                        if builder.num_rows() > 0:
                            yield builder.build()
                        return
                    
                    # Return clean data without metadata wrapping
                    builder.add({"data": data})
                    record_count += 1

                    # Yield block periodically to avoid large blocks
                    if builder.num_rows() >= _MAX_RECORDS_PER_BLOCK:
                        yield builder.build()
                        builder = ArrowBlockBuilder()
            except msgpack.exceptions.OutOfData:
                # Need more data, continue reading
                continue
            except msgpack.exceptions.ExtraData as e:
                logger.warning(f"Extra data in MessagePack file: {e}")
                break
            except msgpack.exceptions.UnpackException as e:
                raise DataSourceError(f"Failed to unpack MessagePack data: {e}") from e

        if builder.num_rows() > 0:
            yield builder.build()
        
        logger.info(f"Processed {record_count} records from MessagePack file: {path}")


def test() -> None:
    """Test MessagePack datasource with example data."""
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
    
    # Create test MessagePack file
    test_file = test_data_dir / "test_data.msgpack"
    
    try:
        import msgpack
    except ImportError:
        logger.warning("MessagePack test skipped: msgpack library not installed")
        return
    
    test_data = [
        {"sensor_id": 1, "value": 42.5, "timestamp": 1234567890},
        {"sensor_id": 2, "value": 43.2, "timestamp": 1234567891},
        {"sensor_id": 3, "value": 44.1, "timestamp": 1234567892},
    ]
    
    with open(test_file, "wb") as f:
        for item in test_data:
            msgpack.pack(item, f)
    
    logger.info(f"Testing MessagePackDatasource with {test_file}")
    
    try:
        # Test MessagePack datasource
        datasource = MessagePackDatasource(paths=str(test_file))
        dataset = ray.data.read_datasource(datasource)
        
        # Verify dataset
        count = 0
        for batch in dataset.iter_batches(batch_size=10, prefetch_batches=0):
            count += len(batch) if hasattr(batch, '__len__') else 1
        
        logger.info(f"MessagePackDatasource test passed: loaded {count} records")
        
        # Cleanup
        if test_file.exists():
            test_file.unlink()
    except Exception as e:
        logger.error(f"MessagePackDatasource test failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    test()
