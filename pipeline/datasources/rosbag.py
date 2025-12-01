"""Custom datasource for ROS bag files.

Supports ROS1 .bag files using rosbag library.
Follows Ray Data FileBasedDatasource API.
"""

from __future__ import annotations

import logging
import os
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


class ROSBagDatasource(FileBasedDatasource):
    """Custom datasource for reading ROS bag files.

    Supports ROS1 .bag files commonly used in robotics for storing
    sensor data, images, and other ROS messages.
    """

    def __init__(
        self,
        paths: Union[str, List[str]],
        topics: Optional[List[str]] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        max_messages: Optional[int] = None,
        **kwargs: Any,
    ):
        """Initialize ROS bag datasource.

        Args:
            paths: ROS bag file path(s) or directory path(s)
            topics: List of topic names to filter (None = all topics)
            start_time: Start timestamp in seconds (None = beginning)
            end_time: End timestamp in seconds (None = end)
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
        
        if start_time is not None and not isinstance(start_time, (int, float)):
            raise ConfigurationError(f"start_time must be numeric (seconds), got {type(start_time)}")
        if start_time is not None and start_time < 0:
            raise ConfigurationError(f"start_time must be non-negative, got {start_time}")
        
        if end_time is not None and not isinstance(end_time, (int, float)):
            raise ConfigurationError(f"end_time must be numeric (seconds), got {type(end_time)}")
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
        """Read ROS bag file and yield message blocks.

        Args:
            f: pyarrow.NativeFile handle for the file
            path: Path to ROS bag file

        Yields:
            Block objects (pyarrow.Table) with ROS message data

        Raises:
            DataSourceError: If reading fails

        Note:
            ROS bag library requires file paths, not file handles. For S3/HDFS
            support, the file must be copied to local storage first. This is a
            limitation of the rosbag library.
        """
        self._validate_file_handle(f, path)
        
        try:
            import rosbag  # https://wiki.ros.org/rosbag
        except ImportError:
            raise DataSourceError(
                "rosbag library not installed. Install ROS dependencies."
            ) from None

        # ROS bag library requires file paths, not file handles
        # For cloud storage, we need to copy to temp file first
        # This is a limitation of the rosbag library
        resource_mgr = get_resource_manager()
        bag = None
        temp_file_path = None

        try:
            # Check if path is local or needs copying
            is_cloud_path = path.startswith(("s3://", "gs://", "hdfs://", "abfss://"))
            
            if is_cloud_path:
                # Copy to temp file for cloud storage
                # Use resource manager to ensure cleanup
                with resource_mgr.temp_file(suffix=".bag") as temp_path:
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
                                f"ROS bag file {path} is {data_size} bytes, "
                                f"exceeds maximum size of {_MAX_FILE_SIZE_BYTES}"
                            )
                        
                        # Check available disk space before writing
                        has_space, available = check_disk_space(temp_path, data_size)
                        if not has_space:
                            raise DataSourceError(
                                f"Insufficient disk space for ROS bag temp file: "
                                f"required {data_size}, available {available}"
                            )

                        with open(temp_path, "wb") as tf:
                            tf.write(data)

                        bag = rosbag.Bag(temp_path)
                        
                        # Process bag (yield blocks)
                        yield from self._process_rosbag(bag, path)
                    except (OSError, IOError) as e:
                        raise DataSourceError(f"Failed to copy ROS bag file to temp: {e}") from e
            else:
                # Local file, validate exists
                if not os.path.exists(path):
                    raise DataSourceError(f"ROS bag file does not exist: {path}")
                
                if not os.path.isfile(path):
                    raise DataSourceError(f"ROS bag path is not a file: {path}")
                
                # Check file size
                file_size = os.path.getsize(path)
                if file_size > _MAX_FILE_SIZE_BYTES:
                    raise DataSourceError(
                        f"ROS bag file {path} is {file_size} bytes, "
                        f"exceeds maximum size of {_MAX_FILE_SIZE_BYTES}"
                    )
                
                # Local file, use directly
                bag = rosbag.Bag(path)
                # Process bag (yield blocks)
                yield from self._process_rosbag(bag, path)

        except (IOError, OSError, RuntimeError, ValueError) as e:
            logger.error(f"Error reading ROS bag {path}: {e}", exc_info=True)
            # Re-raise exception instead of wrapping in block
            raise DataSourceError(f"Failed to read ROS bag {path}: {e}") from e
        finally:
            # Ensure bag is closed even on error
            if bag is not None:
                try:
                    bag.close()
                except (AttributeError, RuntimeError) as e:
                    logger.warning(f"Error closing ROS bag: {e}")
            
            # Cleanup temp file if created
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                except OSError as e:
                    logger.warning(f"Failed to remove temp file {temp_file_path}: {e}")

    def _process_rosbag(
        self, bag: Any, path: str
    ) -> Iterator[Block]:
        """Process ROS bag and yield message blocks.

        Args:
            bag: Open ROS bag object
            path: Original file path

        Yields:
            Block objects with ROS message data

        Raises:
            DataSourceError: If processing fails
        """
        if bag is None:
            raise DataSourceError("bag cannot be None")
        
        if not path or not path.strip():
            raise DataSourceError("path cannot be empty")
        
        # Filter topics if specified
        topics_filter = self.topics if self.topics else None

        builder = ArrowBlockBuilder()
        message_count = 0

        try:
            # Read messages from bag
            for topic, msg, t in bag.read_messages(
                topics=topics_filter,
                start_time=self.start_time,
                end_time=self.end_time,
            ):
                # Check max_messages limit
                if self.max_messages is not None and message_count >= self.max_messages:
                    logger.info(f"Reached max_messages limit ({self.max_messages})")
                    break
                
                # Validate timestamp
                try:
                    timestamp_sec = t.to_sec()
                    if timestamp_sec < 0:
                        logger.warning(f"Invalid timestamp (negative): {timestamp_sec}")
                        continue
                except (AttributeError, RuntimeError) as e:
                    logger.warning(f"Failed to convert timestamp: {e}")
                    continue
                
                # Convert ROS message to dict
                try:
                    msg_dict = self._ros_message_to_dict(msg)
                except Exception as e:
                    logger.warning(f"Failed to convert ROS message: {e}")
                    # Return minimal data instead of error dict
                    msg_dict = {"raw_data": str(msg)}

                # Return clean data without metadata wrapping
                # Path is handled automatically by Ray Data via include_paths
                item = {
                    "topic": topic,
                    "timestamp": timestamp_sec,
                    "message": msg_dict,
                    "message_type": type(msg).__name__ if hasattr(msg, "__class__") else "unknown",
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
            
            logger.info(f"Processed {message_count} messages from ROS bag: {path}")
        except Exception as e:
            raise DataSourceError(f"Failed to process ROS bag {path}: {e}") from e

    def _ros_message_to_dict(self, msg: Any) -> dict:
        """Convert ROS message to dictionary.

        Args:
            msg: ROS message object

        Returns:
            Dictionary representation of message

        Raises:
            DataSourceError: If message conversion fails
        """
        if msg is None:
            raise DataSourceError("ROS message is None")
        
        try:
            if hasattr(msg, "__slots__"):
                result = {}
                for slot in msg.__slots__:
                    try:
                        value = getattr(msg, slot)
                        # Convert complex types to serializable format
                        if hasattr(value, "__dict__"):
                            result[slot] = str(value)
                        else:
                            result[slot] = value
                    except (AttributeError, TypeError) as e:
                        logger.debug(f"Failed to get slot {slot}: {e}")
                        result[slot] = None
                return result
            elif hasattr(msg, "__dict__"):
                return {k: str(v) if not isinstance(v, (str, int, float, bool, type(None))) else v 
                        for k, v in msg.__dict__.items()}
            else:
                return {"data": str(msg)}
        except (AttributeError, TypeError) as e:
            logger.debug(f"Failed to convert ROS message to dict: {e}")
            # Re-raise as DataSourceError instead of returning error dict
            raise DataSourceError(f"Failed to convert ROS message to dict: {e}") from e


def test() -> None:
    """Test ROS bag datasource with example data."""
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
    
    logger.info("Testing ROSBagDatasource")
    logger.info("Note: ROS bag test requires actual .bag files. Skipping if no test files found.")
    
    # Check for test ROS bag files
    test_files = list(test_data_dir.glob("*.bag"))
    if not test_files:
        logger.warning("ROS bag test skipped: No .bag files found in examples/data/")
        logger.info("To test ROS bag, place test .bag files in examples/data/")
        return
    
    try:
        # Test ROS bag datasource
        datasource = ROSBagDatasource(paths=str(test_files[0]))
        dataset = ray.data.read_datasource(datasource)
        
        # Verify dataset
        count = 0
        for batch in dataset.iter_batches(batch_size=10, prefetch_batches=0):
            count += len(batch) if hasattr(batch, '__len__') else 1
        
        logger.info(f"ROSBagDatasource test passed: loaded {count} messages")
    except Exception as e:
        logger.error(f"ROSBagDatasource test failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    test()
