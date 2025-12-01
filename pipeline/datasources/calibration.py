"""Custom datasource for camera calibration files.

Camera calibration data is commonly stored in YAML, JSON, or XML formats
for intrinsic/extrinsic parameters, distortion coefficients, etc.
Follows Ray Data FileBasedDatasource API.
"""

from __future__ import annotations

import json
import logging
import os
import xml.etree.ElementTree as ET  # https://docs.python.org/3/library/xml.etree.elementtree.html
from typing import TYPE_CHECKING, Any, Iterator, Union

import yaml  # https://pyyaml.org/
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


class CalibrationDatasource(FileBasedDatasource):
    """Custom datasource for reading camera calibration files.

    Supports common calibration formats (YAML, JSON, XML) used in robotics
    for camera intrinsic/extrinsic parameters and distortion coefficients.
    """

    def __init__(
        self,
        paths: Union[str, list[str]],
        **kwargs: Any,
    ):
        """Initialize calibration datasource.

        Args:
            paths: Calibration file path(s) or directory path(s)
            **kwargs: Additional FileBasedDatasource options
        """
        super().__init__(paths=paths, **kwargs)

    @log_execution_time
    @handle_errors(error_class=DataSourceError)
    def _read_stream(
        self, f: "pyarrow.NativeFile", path: str
    ) -> Iterator[Block]:
        """Read calibration file and yield calibration blocks.

        Args:
            f: pyarrow.NativeFile handle (calibration requires direct file access)
            path: Path to calibration file

        Yields:
            Block objects (pyarrow.Table) with calibration data

        Raises:
            DataSourceError: If reading fails

        Note:
            Calibration file libraries require direct file access. For cloud storage,
            files must be copied locally first. This is a limitation of the libraries.
        """
        self._validate_file_handle(f, path)
        
        # Validate file exists (for local files)
        if not path.startswith(("s3://", "gs://", "hdfs://", "abfss://")):
            if not os.path.exists(path):
                raise DataSourceError(f"Calibration file does not exist: {path}")
            
            if not os.path.isfile(path):
                raise DataSourceError(f"Calibration path is not a file: {path}")
            
            # Check file size
            file_size = os.path.getsize(path)
            if file_size > _MAX_FILE_SIZE_BYTES:
                raise DataSourceError(
                    f"Calibration file {path} is {file_size} bytes, "
                    f"exceeds maximum size of {_MAX_FILE_SIZE_BYTES}"
                )
        
        path_lower = path.lower()

        if path_lower.endswith((".yaml", ".yml")):
            yield from self._read_yaml_calibration(f, path)
        elif path_lower.endswith(".json"):
            yield from self._read_json_calibration(f, path)
        elif path_lower.endswith(".xml"):
            yield from self._read_xml_calibration(f, path)
        else:
            raise DataSourceError(f"Unsupported calibration format: {path}")

    def _read_yaml_calibration(
        self, f: "pyarrow.NativeFile", path: str
    ) -> Iterator[Block]:
        """Read YAML calibration file.

        Args:
            f: File handle
            path: Path to YAML file

        Yields:
            Block object with calibration data

        Raises:
            DataSourceError: If reading fails
        """
        try:
            # Read from file handle to support cloud storage
            content_bytes = f.readall()
            
            if not content_bytes:
                raise DataSourceError(f"Empty YAML calibration file: {path}")
            
            # Check file size
            if len(content_bytes) > _MAX_FILE_SIZE_BYTES:
                raise DataSourceError(
                    f"YAML calibration file {path} exceeds {_MAX_FILE_SIZE_BYTES} bytes"
                )
            
            # Decode
            try:
                content = content_bytes.decode("utf-8")
            except UnicodeDecodeError as e:
                raise DataSourceError(f"YAML file {path} is not valid UTF-8: {e}") from e
            
            # Parse YAML
            try:
                calib_data = yaml.safe_load(content)
            except yaml.YAMLError as e:
                raise DataSourceError(f"Invalid YAML in file {path}: {e}") from e
            
            if calib_data is None:
                calib_data = {}

            # Return clean calibration data without metadata wrapping
            builder = ArrowBlockBuilder()
            builder.add({"calibration": calib_data})
            yield builder.build()
        except Exception as e:
            raise DataSourceError(f"Failed to read YAML calibration file {path}: {e}") from e

    def _read_json_calibration(
        self, f: "pyarrow.NativeFile", path: str
    ) -> Iterator[Block]:
        """Read JSON calibration file.

        Args:
            f: File handle
            path: Path to JSON file

        Yields:
            Block object with calibration data

        Raises:
            DataSourceError: If reading fails
        """
        try:
            # Read from file handle to support cloud storage
            content_bytes = f.readall()
            
            if not content_bytes:
                raise DataSourceError(f"Empty JSON calibration file: {path}")
            
            # Check file size
            if len(content_bytes) > _MAX_FILE_SIZE_BYTES:
                raise DataSourceError(
                    f"JSON calibration file {path} exceeds {_MAX_FILE_SIZE_BYTES} bytes"
                )
            
            # Decode
            try:
                content = content_bytes.decode("utf-8")
            except UnicodeDecodeError as e:
                raise DataSourceError(f"JSON file {path} is not valid UTF-8: {e}") from e
            
            # Parse JSON
            try:
                calib_data = json.loads(content)
            except json.JSONDecodeError as e:
                raise DataSourceError(f"Invalid JSON in file {path}: {e}") from e

            # Return clean calibration data without metadata wrapping
            builder = ArrowBlockBuilder()
            builder.add({"calibration": calib_data})
            yield builder.build()
        except Exception as e:
            raise DataSourceError(f"Failed to read JSON calibration file {path}: {e}") from e

    def _read_xml_calibration(
        self, f: "pyarrow.NativeFile", path: str
    ) -> Iterator[Block]:
        """Read XML calibration file.

        Args:
            f: File handle
            path: Path to XML file

        Yields:
            Block object with calibration data

        Raises:
            DataSourceError: If reading fails
        """
        try:
            # Read from file handle to support cloud storage
            content_bytes = f.readall()
            
            if not content_bytes:
                raise DataSourceError(f"Empty XML calibration file: {path}")
            
            # Check file size
            if len(content_bytes) > _MAX_FILE_SIZE_BYTES:
                raise DataSourceError(
                    f"XML calibration file {path} exceeds {_MAX_FILE_SIZE_BYTES} bytes"
                )
            
            # Decode
            try:
                content = content_bytes.decode("utf-8")
            except UnicodeDecodeError as e:
                raise DataSourceError(f"XML file {path} is not valid UTF-8: {e}") from e
            
            # Parse XML
            try:
                root = ET.fromstring(content)
            except ET.ParseError as e:
                raise DataSourceError(f"Invalid XML in file {path}: {e}") from e
            
            if root is None:
                raise DataSourceError(f"Failed to parse XML root from {path}")

            # Parse OpenCV-style calibration XML
            calib_data: dict[str, Any] = {}
            for child in root:
                if child is None:
                    continue
                
                if child.tag == "camera_matrix":
                    try:
                        calib_data["camera_matrix"] = self._parse_matrix(child)
                    except Exception as e:
                        logger.warning(f"Failed to parse camera_matrix: {e}")
                        calib_data["camera_matrix_error"] = str(e)
                elif child.tag == "distortion_coefficients":
                    try:
                        calib_data["distortion_coefficients"] = self._parse_matrix(child)
                    except Exception as e:
                        logger.warning(f"Failed to parse distortion_coefficients: {e}")
                        calib_data["distortion_coefficients_error"] = str(e)

            # Return clean calibration data without metadata wrapping
            builder = ArrowBlockBuilder()
            builder.add({"calibration": calib_data})
            yield builder.build()
        except Exception as e:
            raise DataSourceError(f"Failed to read XML calibration file {path}: {e}") from e

    def _parse_matrix(self, elem: Any) -> list[list[float]]:
        """Parse matrix element from XML.

        Args:
            elem: XML element containing matrix data

        Returns:
            Matrix as list of lists

        Raises:
            ValueError: If parsing fails
        """
        if elem is None:
            return []
        
        matrix = []
        for row in elem.findall("row"):
            if row is None or row.text is None:
                continue
            
            try:
                row_data = [float(val) for val in row.text.split()]
                matrix.append(row_data)
            except ValueError as e:
                logger.warning(f"Failed to parse matrix row: {e}")
                continue
        
        return matrix


def test() -> None:
    """Test calibration datasource with example data."""
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
    
    # Create test YAML calibration file
    test_file = test_data_dir / "test_calibration.yaml"
    test_calib_content = """
camera_matrix:
  - [640.0, 0.0, 320.0]
  - [0.0, 640.0, 240.0]
  - [0.0, 0.0, 1.0]
distortion_coefficients:
  - [0.1, -0.05, 0.0, 0.0, 0.0]
"""
    
    with open(test_file, "w") as f:
        f.write(test_calib_content)
    
    logger.info(f"Testing CalibrationDatasource with {test_file}")
    
    try:
        # Test calibration datasource
        datasource = CalibrationDatasource(paths=str(test_file))
        dataset = ray.data.read_datasource(datasource)
        
        # Verify dataset
        count = 0
        for batch in dataset.iter_batches(batch_size=10, prefetch_batches=0):
            count += len(batch) if hasattr(batch, '__len__') else 1
        
        logger.info(f"CalibrationDatasource test passed: loaded {count} items")
        
        # Cleanup
        if test_file.exists():
            test_file.unlink()
    except Exception as e:
        logger.error(f"CalibrationDatasource test failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    test()
