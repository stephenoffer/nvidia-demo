"""Custom datasource for Velodyne LIDAR data files.

Velodyne LIDAR sensors commonly output data in PCAP format or
custom binary formats for point cloud data.
Follows Ray Data FileBasedDatasource API.
"""

from __future__ import annotations

import logging
import os
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
_MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024  # 100MB
_MAX_PACKETS_PER_BLOCK = 10000


class VelodyneDatasource(FileBasedDatasource):
    """Custom datasource for reading Velodyne LIDAR data files.

    Supports Velodyne PCAP files and VLP-16/VLP-32 data formats
    commonly used in robotics for 3D point cloud data.
    """

    def __init__(
        self,
        paths: Union[str, list[str]],
        max_packets: Optional[int] = None,
        **kwargs: Any,
    ):
        """Initialize Velodyne datasource.

        Args:
            paths: Velodyne data file path(s) or directory path(s)
            max_packets: Maximum number of packets to read (None = unlimited)
            **kwargs: Additional FileBasedDatasource options

        Raises:
            ValueError: If parameters are invalid
        """
        super().__init__(paths=paths, **kwargs)
        
        # Validate parameters
        if max_packets is not None and max_packets <= 0:
            raise ConfigurationError(f"max_packets must be positive, got {max_packets}")
        
        self.max_packets = max_packets

    @log_execution_time
    @handle_errors(error_class=DataSourceError)
    def _read_stream(
        self, f: "pyarrow.NativeFile", path: str
    ) -> Iterator[Block]:
        """Read Velodyne file and yield point cloud blocks.

        Args:
            f: pyarrow.NativeFile handle (Velodyne requires direct file access)
            path: Path to Velodyne data file

        Yields:
            Block objects (pyarrow.Table) with LIDAR point cloud data

        Raises:
            DataSourceError: If reading fails

        Note:
            Velodyne libraries require direct file access. For cloud storage,
            files must be copied locally first.
        """
        self._validate_file_handle(f, path)
        
        # Validate file exists (for local files)
        if not path.startswith(("s3://", "gs://", "hdfs://", "abfss://")):
            if not os.path.exists(path):
                raise DataSourceError(f"Velodyne file does not exist: {path}")
            
            if not os.path.isfile(path):
                raise DataSourceError(f"Velodyne path is not a file: {path}")
        
        path_lower = path.lower()

        try:
            if path_lower.endswith(".pcap"):
                yield from self._read_pcap(path)
            else:
                # Try to read as Velodyne binary format
                yield from self._read_velodyne_binary(path)

        except Exception as e:
            logger.error(f"Error reading Velodyne file {path}: {e}", exc_info=True)
            # Re-raise exception instead of wrapping in block
            raise DataSourceError(f"Failed to read Velodyne file {path}: {e}") from e

    def _read_pcap(self, path: str) -> Iterator[Block]:
        """Read Velodyne PCAP file.

        Args:
            path: Path to PCAP file

        Yields:
            Block objects with LIDAR packets

        Raises:
            DataSourceError: If reading fails
        """
        if not os.path.exists(path):
            raise DataSourceError(f"PCAP file does not exist: {path}")
        
        try:
            import dpkt  # https://github.com/kbandla/dpkt
        except ImportError:
            raise DataSourceError(
                "dpkt library not installed. Install with: pip install dpkt"
            ) from None

        try:
            builder = ArrowBlockBuilder()
            packet_count = 0

            with open(path, "rb") as pcap_file:
                try:
                    pcap = dpkt.pcap.Reader(pcap_file)
                except Exception as e:
                    raise DataSourceError(f"Invalid PCAP file {path}: {e}") from e

                for timestamp, buf in pcap:
                    # Check max_packets limit
                    if self.max_packets is not None and packet_count >= self.max_packets:
                        logger.info(f"Reached max_packets limit ({self.max_packets})")
                        break
                    
                    # Validate timestamp
                    if not isinstance(timestamp, (int, float)):
                        logger.warning(f"Invalid timestamp type: {type(timestamp)}")
                        continue
                    
                    # Validate buffer
                    if not isinstance(buf, bytes):
                        logger.warning(f"Invalid buffer type: {type(buf)}")
                        continue
                    
                    # Parse Velodyne packet (simplified)
                    # In production, use proper Velodyne packet parsing
                    # Return clean data without metadata wrapping
                    builder.add(
                        {
                            "packet_index": packet_count,
                            "timestamp": float(timestamp),
                            "packet_size": len(buf),
                            "packet_data": buf.hex(),  # Hex representation
                        }
                    )
                    packet_count += 1
                    
                    # Yield block periodically
                    if builder.num_rows() >= _MAX_PACKETS_PER_BLOCK:
                        yield builder.build()
                        builder = ArrowBlockBuilder()

            if builder.num_rows() > 0:
                yield builder.build()
            
            logger.info(f"Processed {packet_count} packets from PCAP file: {path}")

        except Exception as e:
            raise DataSourceError(f"Failed to read PCAP file {path}: {e}") from e

    def _read_velodyne_binary(self, path: str) -> Iterator[Block]:
        """Read Velodyne binary format file.

        Args:
            path: Path to binary file

        Yields:
            Block objects with point cloud data

        Raises:
            DataSourceError: If reading fails

        Note:
            This method reads entire file into memory. For very large files,
            consider using chunked reading or streaming.
        """
        if not os.path.exists(path):
            raise DataSourceError(f"Velodyne binary file does not exist: {path}")
        
        try:
            import numpy as np  # https://numpy.org/
        except ImportError:
            raise DataSourceError("numpy is required for Velodyne binary reading") from None

        try:
            # Check file size to avoid OOM
            file_size = os.path.getsize(path)
            
            if file_size > _MAX_FILE_SIZE_BYTES:
                logger.warning(
                    f"Velodyne file {path} is {file_size} bytes, "
                    f"truncating to {_MAX_FILE_SIZE_BYTES} bytes"
                )
                # Read only first max_size bytes
                data = np.fromfile(path, dtype=np.uint8, count=_MAX_FILE_SIZE_BYTES)
                truncated = True
            else:
                # Read binary file
                data = np.fromfile(path, dtype=np.uint8)
                truncated = False

            # Velodyne VLP-16/VLP-32 format parsing (simplified)
            # In production, use proper Velodyne format parser
            # For large arrays, don't convert to list (use Arrow arrays)
            builder = ArrowBlockBuilder()
            
            # Only convert small arrays to list
            if len(data) < 1_000_000:
                data_serializable = data.tolist()
            else:
                # For large arrays, store metadata only
                data_serializable = {
                    "type": "array",
                    "shape": list(data.shape),
                    "dtype": str(data.dtype),
                    "size": len(data),
                    "data_truncated": True,
                }
            
            # Return clean data without metadata wrapping
            builder.add(
                {
                    "size": len(data),
                    "data": data_serializable,
                    "truncated": truncated,
                }
            )
            yield builder.build()

        except Exception as e:
            raise DataSourceError(f"Failed to read Velodyne binary file {path}: {e}") from e


def test() -> None:
    """Test Velodyne datasource with example data."""
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
    
    logger.info("Testing VelodyneDatasource")
    logger.info("Note: Velodyne test requires actual PCAP or binary files. Skipping if no test files found.")
    
    # Check for test Velodyne files
    test_files = list(test_data_dir.glob("*.pcap"))
    if not test_files:
        # Try binary files
        test_files = list(test_data_dir.glob("velodyne*.bin"))
    
    if not test_files:
        logger.warning("Velodyne test skipped: No .pcap or velodyne*.bin files found in examples/data/")
        logger.info("To test Velodyne, place test files (.pcap or velodyne*.bin) in examples/data/")
        return
    
    try:
        # Test Velodyne datasource
        datasource = VelodyneDatasource(paths=str(test_files[0]))
        dataset = ray.data.read_datasource(datasource)
        
        # Verify dataset
        count = 0
        for batch in dataset.iter_batches(batch_size=10, prefetch_batches=0):
            count += len(batch) if hasattr(batch, '__len__') else 1
        
        logger.info(f"VelodyneDatasource test passed: loaded {count} packets")
    except Exception as e:
        logger.error(f"VelodyneDatasource test failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    test()
