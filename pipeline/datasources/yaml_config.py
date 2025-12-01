"""Custom datasource for YAML configuration files.

YAML is widely used in robotics for configuration files, robot parameters,
and calibration data.
Follows Ray Data FileBasedDatasource API.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Iterator, Union

from ray.data._internal.arrow_block import ArrowBlockBuilder
from ray.data.block import Block

from pipeline.datasources.base import FileBasedDatasource
from pipeline.exceptions import DataSourceError
from pipeline.utils.decorators import handle_errors, log_execution_time

if TYPE_CHECKING:
    import pyarrow

logger = logging.getLogger(__name__)

# Constants
_MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10MB


class YAMLConfigDatasource(FileBasedDatasource):
    """Custom datasource for reading YAML configuration files.

    YAML is commonly used in robotics for configuration files, robot
    parameters, calibration data, and launch files.
    """

    def __init__(
        self,
        paths: Union[str, list[str]],
        **kwargs: Any,
    ):
        """Initialize YAML config datasource.

        Args:
            paths: YAML file path(s) or directory path(s)
            **kwargs: Additional FileBasedDatasource options
        """
        super().__init__(paths=paths, **kwargs)

    @log_execution_time
    @handle_errors(error_class=DataSourceError)
    def _read_stream(
        self, f: "pyarrow.NativeFile", path: str
    ) -> Iterator[Block]:
        """Read YAML file and yield configuration blocks.

        Args:
            f: pyarrow.NativeFile handle for the file (supports S3, HDFS, etc.)
            path: Path to YAML file

        Yields:
            Block objects (pyarrow.Table) with YAML configuration data

        Raises:
            DataSourceError: If reading fails
        """
        self._validate_file_handle(f, path)
        
        try:
            import yaml  # https://pyyaml.org/
        except ImportError:
            raise DataSourceError(
                "yaml library not installed. Install with: pip install pyyaml"
            ) from None

        try:
            # Use provided file handle to support S3, HDFS, etc.
            # Read as text (YAML is text format)
            content_bytes = f.readall()
            
            if not content_bytes:
                # Empty file - return empty block following Ray Data pattern
                builder = ArrowBlockBuilder()
                builder.add({"config": {}})
                yield builder.build()
                return
            
            # Check file size
            if len(content_bytes) > _MAX_FILE_SIZE_BYTES:
                raise DataSourceError(
                    f"YAML file {path} is {len(content_bytes)} bytes, "
                    f"exceeds maximum size of {_MAX_FILE_SIZE_BYTES}"
                )
            
            # Decode with error handling
            try:
                content = content_bytes.decode("utf-8")
            except UnicodeDecodeError as e:
                # Try other encodings
                try:
                    content = content_bytes.decode("latin-1")
                    logger.warning(f"YAML file {path} is not UTF-8, using latin-1")
                except UnicodeDecodeError:
                    raise DataSourceError(f"YAML file {path} is not valid UTF-8 or latin-1: {e}") from e
            
            # Parse YAML with safe loader
            try:
                config_data = yaml.safe_load(content)
            except yaml.YAMLError as e:
                raise DataSourceError(f"Invalid YAML in file {path}: {e}") from e
            
            # Handle None (empty YAML)
            if config_data is None:
                config_data = {}

            # Return clean data without metadata wrapping
            builder = ArrowBlockBuilder()
            builder.add({"config": config_data})
            yield builder.build()

        except Exception as e:
            logger.error(f"Error reading YAML file {path}: {e}", exc_info=True)
            # Re-raise exception instead of wrapping in block
            raise DataSourceError(f"Failed to read YAML file {path}: {e}") from e


def test() -> None:
    """Test YAML config datasource with example data."""
    import os
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
    
    # Create test YAML file
    test_file = test_data_dir / "test_config.yaml"
    test_yaml_content = """
robot_name: test_robot
joints:
  - name: joint1
    type: revolute
    limits: [0, 3.14]
  - name: joint2
    type: revolute
    limits: [-1.57, 1.57]
sensors:
  camera:
    resolution: [640, 480]
    fps: 30
"""
    
    with open(test_file, "w") as f:
        f.write(test_yaml_content)
    
    logger.info(f"Testing YAMLConfigDatasource with {test_file}")
    
    try:
        # Test YAML datasource
        datasource = YAMLConfigDatasource(paths=str(test_file))
        dataset = ray.data.read_datasource(datasource)
        
        # Verify dataset
        count = 0
        for batch in dataset.iter_batches(batch_size=10, prefetch_batches=0):
            count += len(batch) if hasattr(batch, '__len__') else 1
        
        logger.info(f"YAMLConfigDatasource test passed: loaded {count} items")
        
        # Cleanup
        if test_file.exists():
            test_file.unlink()
    except Exception as e:
        logger.error(f"YAMLConfigDatasource test failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    test()
