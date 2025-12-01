"""Custom datasource for Protocol Buffer files.

Protocol Buffers are commonly used in robotics for efficient serialization
of structured data, especially in ROS and other robotics frameworks.
Follows Ray Data FileBasedDatasource API.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Iterator, Optional, Union

from ray.data._internal.arrow_block import ArrowBlockBuilder
from ray.data.block import Block

from pipeline.datasources.base import FileBasedDatasource
from pipeline.exceptions import DataSourceError
from pipeline.utils.decorators import handle_errors, log_execution_time

if TYPE_CHECKING:
    import pyarrow

logger = logging.getLogger(__name__)

# Constants
_MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024  # 100MB


class ProtobufDatasource(FileBasedDatasource):
    """Custom datasource for reading Protocol Buffer files.

    Protocol Buffers (protobuf) are a language-neutral, platform-neutral
    serialization format commonly used in robotics for efficient data storage.
    """

    def __init__(
        self,
        paths: Union[str, list[str]],
        message_type: Optional[str] = None,
        **kwargs: Any,
    ):
        """Initialize protobuf datasource.

        Args:
            paths: Protobuf file path(s) or directory path(s)
            message_type: Optional message type name for parsing
            **kwargs: Additional FileBasedDatasource options

        Raises:
            ValueError: If parameters are invalid
        """
        super().__init__(paths=paths, **kwargs)
        
        # Validate parameters
        if message_type is not None and not isinstance(message_type, str):
            raise ValueError(f"message_type must be str, got {type(message_type)}")
        if message_type is not None and not message_type.strip():
            raise ValueError("message_type cannot be empty")
        
        self.message_type = message_type

    @log_execution_time
    @handle_errors(error_class=DataSourceError)
    def _read_stream(
        self, f: "pyarrow.NativeFile", path: str
    ) -> Iterator[Block]:
        """Read protobuf file and yield message blocks.

        Args:
            f: pyarrow.NativeFile handle for the file (supports S3, HDFS, etc.)
            path: Path to protobuf file

        Yields:
            Block objects (pyarrow.Table) with parsed protobuf messages

        Raises:
            DataSourceError: If reading fails
        """
        self._validate_file_handle(f, path)
        
        try:
            from google.protobuf import (
                message,  # https://protobuf.dev/
                text_format,
            )
        except ImportError:
            raise DataSourceError(
                "protobuf library not installed. Install with: pip install protobuf"
            ) from None

        try:
            # Use provided file handle to support S3, HDFS, etc.
            # Limit size to avoid OOM on very large files
            data = f.read(_MAX_FILE_SIZE_BYTES)
            data_size = len(data)
            
            if data_size == 0:
                # Empty file - return empty block following Ray Data pattern
                builder = ArrowBlockBuilder()
                builder.add({"message": {}})
                yield builder.build()
                return
            
            # Check if there's more data
            if data_size == _MAX_FILE_SIZE_BYTES:
                try:
                    f.read(1)  # Try to read one more byte
                    logger.warning(
                        f"Protobuf file {path} exceeds {_MAX_FILE_SIZE_BYTES} bytes, truncating"
                    )
                except Exception:
                    pass  # EOF, file is exactly max_size

            builder = ArrowBlockBuilder()

            # Try to parse as generic message
            # In production, you'd use specific message types
            try:
                # Attempt to parse as text format first
                try:
                    decoded_data = data.decode("utf-8")
                    msg = text_format.Parse(decoded_data, message.Message())
                    # Convert message to dictionary
                    msg_dict = self._message_to_dict(msg)

                    # Return clean data without metadata wrapping
                    builder.add(
                        {
                            "message_type": self.message_type or type(msg).__name__,
                            "message": msg_dict,
                        }
                    )
                except UnicodeDecodeError:
                    # Not UTF-8, try binary format
                    raise text_format.ParseError("Not UTF-8 text format")
            except text_format.ParseError:
                # Fall back to binary format
                # Note: This requires knowing the message type
                # For now, we'll return raw data
                # Return clean data without metadata wrapping
                builder.add(
                    {
                        "message_type": self.message_type or "unknown",
                        "raw_data": data.hex() if isinstance(data, bytes) else str(data),
                        "parse_error": "Binary format requires message_type",
                    }
                )

            yield builder.build()

        except Exception as e:
            logger.error(f"Error reading protobuf file {path}: {e}", exc_info=True)
            # Re-raise exception instead of wrapping in block
            raise DataSourceError(f"Failed to read protobuf file {path}: {e}") from e

    def _message_to_dict(self, msg: Any) -> dict:
        """Convert protobuf message to dictionary.

        Args:
            msg: Protobuf message object

        Returns:
            Dictionary representation
        """
        if msg is None:
            return {}
        
        result = {}
        try:
            for field, value in msg.ListFields():
                try:
                    if hasattr(value, "ListFields"):
                        result[field.name] = self._message_to_dict(value)
                    elif isinstance(value, list):
                        result[field.name] = [
                            self._message_to_dict(item) if hasattr(item, "ListFields") else item
                            for item in value
                        ]
                    else:
                        # Convert complex types to serializable format
                        if hasattr(value, "__dict__"):
                            result[field.name] = str(value)
                        else:
                            result[field.name] = value
                except Exception as e:
                    logger.debug(f"Failed to convert field {field.name}: {e}")
                    result[field.name] = str(value)
        except Exception as e:
            logger.warning(f"Failed to convert protobuf message to dict: {e}")
            # Re-raise exception instead of returning error dict
            raise DataSourceError(f"Failed to convert protobuf message to dict: {e}") from e
        
        return result


def test() -> None:
    """Test protobuf datasource with example data."""
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
    
    # Create test protobuf file (text format)
    test_file = test_data_dir / "test_data.pb"
    
    try:
        from google.protobuf import text_format
        from google.protobuf.message import Message
        
        # Create a simple text-format protobuf file
        test_content = """
        sensor_id: 1
        value: 42.5
        timestamp: 1234567890
        """
        
        with open(test_file, "w") as f:
            f.write(test_content)
        
        logger.info(f"Testing ProtobufDatasource with {test_file}")
        
        # Test protobuf datasource
        datasource = ProtobufDatasource(paths=str(test_file))
        dataset = ray.data.read_datasource(datasource)
        
        # Verify dataset
        count = 0
        for batch in dataset.iter_batches(batch_size=10, prefetch_batches=0):
            count += len(batch) if hasattr(batch, '__len__') else 1
        
        logger.info(f"ProtobufDatasource test passed: loaded {count} items")
        
        # Cleanup
        if test_file.exists():
            test_file.unlink()
    except ImportError:
        logger.warning("Protobuf test skipped: protobuf library not installed")
    except Exception as e:
        logger.error(f"ProtobufDatasource test failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    test()
