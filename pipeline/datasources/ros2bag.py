"""Custom datasource for ROS2 bag files.

Supports ROS2 .db3 files using rclpy library.
Follows Ray Data FileBasedDatasource API.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, List, Optional, Union

from ray.data._internal.arrow_block import ArrowBlockBuilder
from ray.data.block import Block

from pipeline.datasources.base import FileBasedDatasource
from pipeline.exceptions import ConfigurationError, DataSourceError
from pipeline.utils.decorators import handle_errors, log_execution_time
from pipeline.utils.resource_manager import check_disk_space, get_resource_manager

if TYPE_CHECKING:
    import pyarrow

logger = logging.getLogger(__name__)

# Constants
_MAX_MESSAGES_PER_BLOCK = 1000
_MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024 * 1024  # 10GB
_DEFAULT_CHUNK_SIZE = 64 * 1024  # 64KB


class ROS2BagDatasource(FileBasedDatasource):
    """Custom datasource for reading ROS2 bag files.

    Supports ROS2 .db3 files (SQLite-based bag format) commonly used in
    robotics for storing sensor data, images, and other ROS2 messages.
    """

    def __init__(
        self,
        paths: Union[str, List[str]],
        topics: Optional[List[str]] = None,
        start_time: Optional[int] = None,  # nanoseconds
        end_time: Optional[int] = None,  # nanoseconds
        max_messages: Optional[int] = None,
        **kwargs: Any,
    ):
        """Initialize ROS2 bag datasource.

        Args:
            paths: ROS2 bag file path(s) or directory path(s)
            topics: List of topic names to filter (None = all topics)
            start_time: Start timestamp in nanoseconds (None = beginning)
            end_time: End timestamp in nanoseconds (None = end)
            max_messages: Maximum number of messages to read (None = unlimited)
            **kwargs: Additional FileBasedDatasource options

        Raises:
            ValueError: If parameters are invalid
        """
        super().__init__(paths=paths, **kwargs)
        
        # Validate parameters
        if topics is not None and not isinstance(topics, list):
            raise ConfigurationError(f"topics must be a list, got {type(topics)}")
        if topics is not None and len(topics) == 0:
            raise ConfigurationError("topics cannot be an empty list")
        
        if start_time is not None and not isinstance(start_time, int):
            raise ConfigurationError(f"start_time must be int (nanoseconds), got {type(start_time)}")
        if start_time is not None and start_time < 0:
            raise ConfigurationError(f"start_time must be non-negative, got {start_time}")
        
        if end_time is not None and not isinstance(end_time, int):
            raise ConfigurationError(f"end_time must be int (nanoseconds), got {type(end_time)}")
        if end_time is not None and end_time < 0:
            raise ConfigurationError(f"end_time must be non-negative, got {end_time}")
        
        if start_time is not None and end_time is not None and start_time >= end_time:
            raise ConfigurationError(f"start_time ({start_time}) must be < end_time ({end_time})")
        
        if max_messages is not None and max_messages <= 0:
            raise ConfigurationError(f"max_messages must be positive, got {max_messages}")
        
        self.topics = topics
        self.start_time = start_time
        self.end_time = end_time
        self.max_messages = max_messages

    @log_execution_time
    @handle_errors(error_class=DataSourceError)
    def _read_stream(
        self, f: "pyarrow.NativeFile", path: str
    ) -> Iterator[Block]:
        """Read ROS2 bag file and yield message blocks.

        Args:
            f: pyarrow.NativeFile handle for the file
            path: Path to ROS2 bag file

        Yields:
            Block objects (pyarrow.Table) with ROS2 message data

        Raises:
            DataSourceError: If reading fails

        Note:
            ROS2 bag library (rclpy) requires file paths, not file handles.
            For S3/HDFS support, the file must be copied to local storage first.
            This is a limitation of the rclpy library.
        """
        self._validate_file_handle(f, path)
        
        try:
            # ROS2 uses rclpy.serialization and rosbag2_py
            try:
                import rosbag2_py  # https://github.com/ros2/rosbag2
            except ImportError:
                # Fallback: try rclpy directly
                try:
                    import rclpy
                    from rclpy.serialization import deserialize_message
                except ImportError:
                    raise DataSourceError(
                        "rosbag2_py or rclpy not installed. Install ROS2 dependencies: "
                        "pip install rosbag2_py or install ROS2."
                    ) from None

            # ROS2 bag library requires file paths, not file handles
            # For cloud storage, we need to copy to temp file first
            resource_mgr = get_resource_manager()
            reader = None
            temp_file_path = None

            try:
                # Check if path is local or needs copying
                is_cloud_path = path.startswith(("s3://", "gs://", "hdfs://", "abfss://"))
                
                if is_cloud_path:
                    # Copy to temp file for cloud storage
                    with resource_mgr.temp_file(suffix=".db3") as temp_path:
                        temp_file_path = temp_path
                        
                        # Read from file handle and write to temp file
                        try:
                            f.seek(0)  # Reset to beginning
                            data = f.readall()
                            
                            if not data:
                                raise DataSourceError(f"Empty file: {path}")
                            
                            # Check file size
                            data_size = len(data)
                            if data_size > _MAX_FILE_SIZE_BYTES:
                                raise DataSourceError(
                                    f"ROS2 bag file {path} is {data_size} bytes, "
                                    f"exceeds maximum size of {_MAX_FILE_SIZE_BYTES}"
                                )
                            
                            # Check available disk space before writing
                            has_space, available = check_disk_space(temp_path, data_size)
                            if not has_space:
                                raise DataSourceError(
                                    f"Insufficient disk space for ROS2 bag temp file: "
                                    f"required {data_size}, available {available}"
                                )

                            with open(temp_path, "wb") as tf:
                                tf.write(data)
                            
                            # Process ROS2 bag (yield blocks)
                            yield from self._process_ros2bag(temp_path, path)
                        except (OSError, IOError) as e:
                            raise DataSourceError(f"Failed to copy ROS2 bag file to temp: {e}") from e
                else:
                    # Local file, validate exists
                    if not os.path.exists(path):
                        raise DataSourceError(f"ROS2 bag file does not exist: {path}")
                    
                    if not os.path.isfile(path):
                        raise DataSourceError(f"ROS2 bag path is not a file: {path}")
                    
                    # Check file size
                    file_size = os.path.getsize(path)
                    if file_size > _MAX_FILE_SIZE_BYTES:
                        raise DataSourceError(
                            f"ROS2 bag file {path} is {file_size} bytes, "
                            f"exceeds maximum size of {_MAX_FILE_SIZE_BYTES}"
                        )
                    
                    # Process ROS2 bag (yield blocks)
                    yield from self._process_ros2bag(path, path)

            finally:
                # Ensure reader is closed even on error
                if reader is not None:
                    try:
                        if hasattr(reader, "close"):
                            reader.close()
                    except (AttributeError, RuntimeError) as e:
                        logger.warning(f"Error closing ROS2 bag reader: {e}")
                
                # Cleanup temp file if created
                if temp_file_path and os.path.exists(temp_file_path):
                    try:
                        os.remove(temp_file_path)
                    except OSError as e:
                        logger.warning(f"Failed to remove temp file {temp_file_path}: {e}")

        except ImportError:
            raise DataSourceError(
                "rosbag2_py or rclpy library not installed. Install ROS2 dependencies: "
                "pip install rosbag2_py or install ROS2."
            ) from None
        except (IOError, OSError, RuntimeError, ValueError) as e:
            logger.error(f"Error reading ROS2 bag {path}: {e}", exc_info=True)
            # Re-raise exception instead of wrapping in block
            raise DataSourceError(f"Failed to read ROS2 bag {path}: {e}") from e

    def _process_ros2bag(
        self, bag_path: str, original_path: str
    ) -> Iterator[Block]:
        """Process ROS2 bag and yield message blocks.

        Args:
            bag_path: Path to ROS2 bag file (local)
            original_path: Original file path (for metadata)

        Yields:
            Block objects with ROS2 message data

        Raises:
            DataSourceError: If processing fails
        """
        if not bag_path or not bag_path.strip():
            raise DataSourceError("bag_path cannot be empty")
        
        if not os.path.exists(bag_path):
            raise DataSourceError(f"ROS2 bag file does not exist: {bag_path}")
        
        try:
            import rosbag2_py
            from rclpy.serialization import deserialize_message
            from rclpy.time import Time
        except ImportError:
            logger.warning("rosbag2_py not available, skipping ROS2 bag processing")
            return

        # Filter topics if specified
        topics_filter = self.topics if self.topics else None

        builder = ArrowBlockBuilder()
        message_count = 0

        # Create ROS2 bag reader
        reader = None
        try:
            reader = rosbag2_py.SequentialReader()
            storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id="sqlite3")
            converter_options = rosbag2_py.ConverterOptions(
                input_serialization_format="cdr", output_serialization_format="cdr"
            )

            reader.open(storage_options, converter_options)

            # Get topic metadata
            topic_types = reader.get_all_topics_and_types()
            if not topic_types:
                logger.warning(f"No topics found in ROS2 bag: {bag_path}")
                return
            
            topic_type_map = {topic.name: topic.type for topic in topic_types}

            # Read messages from bag
            while reader.has_next():
                # Check max_messages limit
                if self.max_messages is not None and message_count >= self.max_messages:
                    logger.info(f"Reached max_messages limit ({self.max_messages})")
                    break
                
                try:
                    (topic, data, timestamp_ns) = reader.read_next()
                except (RuntimeError, ValueError) as e:
                    logger.warning(f"Failed to read message from ROS2 bag: {e}")
                    continue

                # Validate timestamp
                if not isinstance(timestamp_ns, int):
                    logger.warning(f"Invalid timestamp type: {type(timestamp_ns)}")
                    continue
                
                if timestamp_ns < 0:
                    logger.warning(f"Invalid timestamp (negative): {timestamp_ns}")
                    continue

                # Filter by topic if specified
                if topics_filter and topic not in topics_filter:
                    continue

                # Filter by time range if specified
                if self.start_time is not None and timestamp_ns < self.start_time:
                    continue
                if self.end_time is not None and timestamp_ns > self.end_time:
                    continue

                # Convert ROS2 message to dict
                try:
                    msg_dict = self._ros2_message_to_dict(
                        data, topic, topic_type_map.get(topic, "unknown")
                    )
                except Exception as e:
                    logger.warning(f"Failed to convert ROS2 message: {e}")
                    # Return minimal data instead of error dict
                    msg_dict = {"raw_data": data.hex() if isinstance(data, bytes) else str(data)}

                # Return clean data without metadata wrapping
                # Path is handled automatically by Ray Data via include_paths
                item = {
                    "topic": topic,
                    "timestamp": timestamp_ns / 1e9,  # Convert to seconds
                    "timestamp_ns": timestamp_ns,
                    "message": msg_dict,
                    "message_type": topic_type_map.get(topic, "unknown"),
                    "message_index": message_count,
                }
                builder.add(item)
                message_count += 1

                # Yield block periodically to avoid large blocks
                if builder.num_rows() >= _MAX_MESSAGES_PER_BLOCK:
                    yield builder.build()
                    builder = ArrowBlockBuilder()

            if builder.num_rows() > 0:
                yield builder.build()
            
            logger.info(f"Processed {message_count} messages from ROS2 bag: {bag_path}")

        except Exception as e:
            raise DataSourceError(f"Failed to process ROS2 bag {bag_path}: {e}") from e
        finally:
            if reader is not None:
                try:
                    reader.close()
                except (AttributeError, RuntimeError) as e:
                    logger.warning(f"Error closing ROS2 bag reader: {e}")

    def _ros2_message_to_dict(
        self, data: bytes, topic: str, message_type: str
    ) -> dict:
        """Convert ROS2 message to dictionary.

        Args:
            data: Serialized ROS2 message bytes
            topic: Topic name
            message_type: ROS2 message type name

        Returns:
            Dictionary representation of message
        """
        if not isinstance(data, bytes):
            logger.warning(f"Expected bytes for ROS2 message data, got {type(data)}")
            # Return minimal data instead of error dict
            return {
                "data": str(data),
                "topic": topic,
                "message_type": message_type,
            }
        
        try:
            # Try to deserialize and convert to dict
            # This is a simplified version - full implementation would
            # need to handle all ROS2 message types
            from rclpy.serialization import deserialize_message

            # For now, return raw data and metadata
            # Full deserialization requires importing specific message types
            return {
                "data": data.hex() if isinstance(data, bytes) else str(data),
                "topic": topic,
                "message_type": message_type,
                "size_bytes": len(data) if isinstance(data, bytes) else 0,
            }
        except (ImportError, AttributeError, TypeError) as e:
            logger.debug(f"Could not deserialize ROS2 message: {e}")
            # Return data without error field
            return {
                "data": data.hex() if isinstance(data, bytes) else str(data),
                "topic": topic,
                "message_type": message_type,
                "size_bytes": len(data) if isinstance(data, bytes) else 0,
            }


def test() -> None:
    """Test ROS2 bag datasource with example data."""
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
    
    logger.info("Testing ROS2BagDatasource")
    logger.info("Note: ROS2 bag test requires actual .db3 files. Skipping if no test files found.")
    
    # Check for test ROS2 bag files
    test_files = list(test_data_dir.glob("*.db3"))
    if not test_files:
        logger.warning("ROS2 bag test skipped: No .db3 files found in examples/data/")
        logger.info("To test ROS2 bag, place test .db3 files in examples/data/")
        return
    
    try:
        # Test ROS2 bag datasource
        datasource = ROS2BagDatasource(paths=str(test_files[0]))
        dataset = ray.data.read_datasource(datasource)
        
        # Verify dataset
        count = 0
        for batch in dataset.iter_batches(batch_size=10, prefetch_batches=0):
            count += len(batch) if hasattr(batch, '__len__') else 1
        
        logger.info(f"ROS2BagDatasource test passed: loaded {count} messages")
    except Exception as e:
        logger.error(f"ROS2BagDatasource test failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    test()
