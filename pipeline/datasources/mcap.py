"""MCAP datasource using Ray Data's native MCAP support.

MCAP is a container format for robotics data, commonly used with ROS2.
Ray Data provides native support via ray.data.read_mcap() with predicate
pushdown optimization for efficient filtering.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import ray  # https://docs.ray.io/
from ray.data import Dataset  # https://docs.ray.io/en/latest/data/data.html

from pipeline.exceptions import ConfigurationError, DataSourceError

# TimeRange may not be available in all Ray versions - handle gracefully
try:
    from ray.data.datasource import TimeRange  # type: ignore[attr-defined]
    _TIMERANGE_AVAILABLE = True
except ImportError:
    # TimeRange not available - use tuple instead
    _TIMERANGE_AVAILABLE = False
    TimeRange = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)


def read_mcap(
    paths: Union[str, List[str]],
    topics: Optional[Union[List[str], Set[str]]] = None,
    time_range: Optional[Union[Tuple[int, int], TimeRange]] = None,
    message_types: Optional[Union[List[str], Set[str]]] = None,
    include_metadata: bool = True,
    include_paths: bool = True,
    max_messages: Optional[int] = None,
    **kwargs: Any,
) -> Dataset:
    """Read MCAP files using Ray Data's native MCAP support.

    MCAP (Message Capture) is a container format for storing robotics
    time-series data, commonly used with ROS2. Ray Data provides native
    support with predicate pushdown optimization for efficient filtering.

    Args:
        paths: Single file/directory or list of file/directory paths
        topics: Optional list/set of topic names to filter
        time_range: Optional time range tuple (start_ns, end_ns) or TimeRange object
        message_types: Optional list/set of message type names to filter
        include_metadata: Whether to include MCAP metadata fields (default: True)
        include_paths: Whether to include file paths in output (default: True)
        max_messages: Maximum number of messages to read (None = unlimited)
        **kwargs: Additional arguments passed to ray.data.read_mcap()

    Returns:
        Ray Dataset with MCAP message records

    Raises:
        ValueError: If parameters are invalid
        DataSourceError: If reading fails

    Example:
        ```python
        # Read all MCAP files
        dataset = read_mcap("s3://bucket/mcap-data/")

        # Read with topic and time filtering (using tuple if TimeRange not available)
        dataset = read_mcap(
            "s3://bucket/mcap-data/",
            topics={"/camera/image_raw", "/lidar/points"},
            time_range=(1000000000, 5000000000),  # (start_ns, end_ns)
        )
        ```
    """
    # Validate parameters
    if not paths:
        raise ValueError("paths cannot be empty")
    
    if isinstance(paths, str):
        if not paths.strip():
            raise ValueError("paths string cannot be empty")
    elif isinstance(paths, list):
        if len(paths) == 0:
            raise ValueError("paths list cannot be empty")
        for p in paths:
            if not isinstance(p, str) or not p.strip():
                raise ValueError(f"Invalid path in list: {p}")
    else:
        raise ValueError(f"paths must be str or list[str], got {type(paths)}")
    
    if topics is not None:
        if isinstance(topics, (list, set)):
            if len(topics) == 0:
                raise ConfigurationError("topics cannot be empty")
            for topic in topics:
                if not isinstance(topic, str) or not topic.strip():
                    raise ConfigurationError(f"Invalid topic: {topic}")
        else:
            raise ConfigurationError(f"topics must be list or set, got {type(topics)}")
    
    if time_range is not None:
        if isinstance(time_range, tuple):
            if len(time_range) != 2:
                raise ValueError(f"time_range tuple must have 2 elements, got {len(time_range)}")
            start_ns, end_ns = time_range
            if not isinstance(start_ns, int) or not isinstance(end_ns, int):
                raise ConfigurationError(f"time_range elements must be int (nanoseconds)")
            if start_ns < 0 or end_ns < 0:
                raise ConfigurationError(f"time_range values must be non-negative")
            if start_ns >= end_ns:
                raise ConfigurationError(f"time_range start ({start_ns}) must be < end ({end_ns})")
        elif _TIMERANGE_AVAILABLE and isinstance(time_range, TimeRange):
            pass  # Valid TimeRange object
        else:
            raise ConfigurationError(f"time_range must be tuple or TimeRange, got {type(time_range)}")
    
    if message_types is not None:
        if isinstance(message_types, (list, set)):
            if len(message_types) == 0:
                raise ConfigurationError("message_types cannot be empty")
            for msg_type in message_types:
                if not isinstance(msg_type, str) or not msg_type.strip():
                    raise ConfigurationError(f"Invalid message_type: {msg_type}")
        else:
            raise ConfigurationError(f"message_types must be list or set, got {type(message_types)}")
    
    if max_messages is not None and max_messages <= 0:
        raise ConfigurationError(f"max_messages must be positive, got {max_messages}")
    
    logger.info(f"Reading MCAP files from {paths}")

    try:
        # Use Ray Data's native MCAP reader
        # See: https://docs.ray.io/en/latest/data/api/doc/ray.data.read_mcap.html
        dataset = ray.data.read_mcap(
            paths=paths,
            topics=topics,
            time_range=time_range,
            message_types=message_types,
            include_metadata=include_metadata,
            include_paths=include_paths,
            **kwargs,
        )
    except Exception as e:
        raise DataSourceError(f"Failed to read MCAP files: {e}") from e

    # Apply max_messages limit if specified
    if max_messages is not None:
        def limit_messages(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            """Limit messages in batch."""
            return batch[:max_messages]
        
        dataset = dataset.map_batches(
            limit_messages,
            batch_size=max_messages,
            batch_format="pandas",
        )

    # Add format metadata using optimized batch processing
    # Use larger batch size for metadata addition (lightweight operation)
    def add_metadata(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add format metadata to batch."""
        if not batch:
            return batch
        
        return [
            {
                **item,
                "format": "mcap",
                "data_type": item.get("data_type", "sensor"),
            }
            for item in batch
        ]

    # Use larger batch size for lightweight metadata addition
    from pipeline.utils.constants import _LARGE_BATCH_SIZE

    return dataset.map_batches(
        add_metadata,
        batch_size=_LARGE_BATCH_SIZE,
        batch_format="pandas",  # Specify batch format for consistency
    )


def test() -> None:
    """Test MCAP datasource with example data."""
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
    
    logger.info("Testing MCAP datasource")
    logger.info("Note: MCAP test requires actual MCAP files. Skipping if no test files found.")
    
    # Check for test MCAP files
    test_files = list(test_data_dir.glob("*.mcap"))
    if not test_files:
        logger.warning("MCAP test skipped: No .mcap files found in examples/data/")
        logger.info("To test MCAP, place test MCAP files in examples/data/")
        return
    
    try:
        # Test MCAP datasource using Ray Data's native reader
        dataset = read_mcap(paths=[str(f) for f in test_files[:1]])
        
        # Verify dataset
        count = 0
        for batch in dataset.iter_batches(batch_size=10, prefetch_batches=0):
            count += len(batch) if hasattr(batch, '__len__') else 1
        
        logger.info(f"MCAP datasource test passed: loaded {count} messages")
    except Exception as e:
        logger.error(f"MCAP datasource test failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    test()
