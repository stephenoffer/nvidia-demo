"""GR00T-specific data format support.

Handles GR00T native data formats and metadata schemas.
Follows Ray Data FileBasedDatasource API for consistency.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, TYPE_CHECKING, Union

import ray
from ray.data import Dataset
from ray.data._internal.delegating_block_builder import DelegatingBlockBuilder

from pipeline.exceptions import DataSourceError, ValidationError, ConfigurationError
from pipeline.utils.constants import _DEFAULT_BATCH_SIZE
from pipeline.utils.decorators import handle_errors, log_execution_time
from pipeline.utils.retry import retry_cloud_storage
from pipeline.utils.input_validation import InputValidator

if TYPE_CHECKING:
    import pyarrow

logger = logging.getLogger(__name__)

# Constants
_MAX_FILES_TO_SCAN = 10000
_MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024 * 1024  # 10GB
_SUPPORTED_EXTENSIONS = {".parquet", ".jsonl", ".json"}
_VALID_DATA_TYPES = {"sensor", "video", "text", "multimodal"}


class GR00TDatasource:
    """Datasource for GR00T-specific data formats.

    Supports GR00T native formats, metadata schemas, and validation.
    Note: This is a high-level datasource that doesn't inherit from FileBasedDatasource
    because it operates on directories and multiple files, not single file streams.
    """

    def __init__(
        self,
        groot_path: Union[str, Path],
        include_metadata: bool = True,
        validate_schema: bool = True,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        max_files: Optional[int] = None,
        file_extensions: Optional[List[str]] = None,
    ) -> None:
        """Initialize GR00T datasource.

        Args:
            groot_path: Path to GR00T data directory
            include_metadata: Whether to include GR00T metadata
            validate_schema: Whether to validate GR00T schema
            batch_size: Batch size for processing
            max_files: Maximum number of files to process (None = unlimited)
            file_extensions: List of file extensions to include (None = all supported)

        Raises:
            DataSourceError: If groot_path is invalid
            ValueError: If batch_size is invalid
        """
        # Validate and sanitize input
        if isinstance(groot_path, str):
            # Basic path validation
            if not groot_path or not groot_path.strip():
                raise DataSourceError("GR00T path cannot be empty")
            self.groot_path = Path(groot_path)
        elif isinstance(groot_path, Path):
            self.groot_path = groot_path
        else:
            raise DataSourceError(f"groot_path must be str or Path, got {type(groot_path)}")

        if batch_size <= 0:
            raise ConfigurationError(f"batch_size must be positive, got {batch_size}")
        
        if max_files is not None and max_files <= 0:
            raise ConfigurationError(f"max_files must be positive, got {max_files}")

        self.include_metadata = bool(include_metadata)
        self.validate_schema = bool(validate_schema)
        self.batch_size = int(batch_size)
        self.max_files = max_files
        self.file_extensions = set(file_extensions) if file_extensions else _SUPPORTED_EXTENSIONS
        
        # Validate file extensions
        invalid_extensions = self.file_extensions - _SUPPORTED_EXTENSIONS
        if invalid_extensions:
            logger.warning(
                f"Unsupported file extensions: {invalid_extensions}. "
                f"Supported: {_SUPPORTED_EXTENSIONS}"
            )
            self.file_extensions &= _SUPPORTED_EXTENSIONS

    @log_execution_time
    @handle_errors(error_class=DataSourceError)
    def load(self) -> Dataset:
        """Load GR00T data.

        Returns:
            Ray Dataset with GR00T-formatted data

        Raises:
            DataSourceError: If data loading fails
            ValidationError: If schema validation fails
        """
        # Validate path exists and is directory
        if not self.groot_path.exists():
            raise DataSourceError(f"GR00T data path does not exist: {self.groot_path}")

        if not self.groot_path.is_dir():
            raise DataSourceError(f"GR00T data path is not a directory: {self.groot_path}")

        # Check read permissions
        if not self.groot_path.is_dir() or not self.groot_path.stat().st_mode & 0o444:
            raise DataSourceError(f"GR00T data path is not readable: {self.groot_path}")

        logger.info(f"Loading GR00T data from {self.groot_path}")

        # Scan for GR00T data files with error handling
        try:
            data_files = self._scan_data_files()
        except (OSError, PermissionError) as e:
            raise DataSourceError(f"Failed to scan GR00T data directory: {e}") from e

        if not data_files:
            logger.warning(f"No GR00T data files found in {self.groot_path}")
            return ray.data.from_items([])

        logger.info(f"Found {len(data_files)} GR00T data files")

        # Load files by type
        try:
            combined = self._load_files(data_files)
        except Exception as e:
            raise DataSourceError(f"Failed to load GR00T files: {e}") from e

        # Format and validate GR00T data
        formatted = self._format_and_validate(combined)

        return formatted

    def _scan_data_files(self) -> List[Path]:
        """Scan directory for GR00T data files.

        Returns:
            List of data file paths

        Raises:
            DataSourceError: If scanning fails
        """
        data_files: List[Path] = []
        
        try:
            for ext in self.file_extensions:
                pattern = f"**/*{ext}"
                files = list(self.groot_path.glob(pattern))
                
                # Apply max_files limit
                if self.max_files is not None:
                    files = files[:self.max_files]
                
                data_files.extend(files)
                
                # Check if we've hit the limit
                if self.max_files is not None and len(data_files) >= self.max_files:
                    data_files = data_files[:self.max_files]
                    break
            
            # Remove duplicates and sort for reproducibility
            data_files = sorted(set(data_files))
            
            # Final limit check
            if self.max_files is not None and len(data_files) > self.max_files:
                logger.warning(
                    f"Limiting to {self.max_files} files (found {len(data_files)})"
                )
                data_files = data_files[:self.max_files]
            
        except Exception as e:
            raise DataSourceError(f"Error scanning for data files: {e}") from e

        return data_files

    def _load_files(self, data_files: List[Path]) -> Dataset:
        """Load files and combine into single dataset.

        Args:
            data_files: List of file paths to load

        Returns:
            Combined Ray Dataset

        Raises:
            DataSourceError: If loading fails
        """
        parquet_files = [f for f in data_files if f.suffix == ".parquet"]
        jsonl_files = [f for f in data_files if f.suffix == ".jsonl"]
        json_files = [f for f in data_files if f.suffix == ".json"]

        datasets: List[Dataset] = []

        # Load Parquet files
        if parquet_files:
            try:
                parquet_datasets = [
                    self._load_parquet_file(str(f)) for f in parquet_files
                ]
                datasets.extend(parquet_datasets)
            except Exception as e:
                raise DataSourceError(f"Failed to load Parquet files: {e}") from e

        # Load JSONL files
        if jsonl_files:
            try:
                jsonl_datasets = [
                    self._load_jsonl_file(str(f)) for f in jsonl_files
                ]
                datasets.extend(jsonl_datasets)
            except Exception as e:
                raise DataSourceError(f"Failed to load JSONL files: {e}") from e

        # Load JSON files
        if json_files:
            try:
                json_datasets = [
                    self._load_json_file(str(f)) for f in json_files
                ]
                datasets.extend(json_datasets)
            except Exception as e:
                raise DataSourceError(f"Failed to load JSON files: {e}") from e

        if not datasets:
            return ray.data.from_items([])

        # Combine datasets
        if len(datasets) == 1:
            return datasets[0]
        else:
            try:
                return ray.data.union(*datasets)
            except Exception as e:
                raise DataSourceError(f"Failed to combine datasets: {e}") from e

    @retry_cloud_storage(max_retries=3)
    def _load_parquet_file(self, file_path: str) -> Dataset:
        """Load a single Parquet file with retry logic.

        Args:
            file_path: Path to Parquet file

        Returns:
            Ray Dataset from Parquet file

        Raises:
            DataSourceError: If file cannot be read
        """
        try:
            # Validate file exists and is readable
            path_obj = Path(file_path)
            if not path_obj.exists():
                raise DataSourceError(f"Parquet file does not exist: {file_path}")
            
            if not path_obj.is_file():
                raise DataSourceError(f"Parquet path is not a file: {file_path}")
            
            # Check file size
            file_size = path_obj.stat().st_size
            if file_size > _MAX_FILE_SIZE_BYTES:
                logger.warning(
                    f"Parquet file {file_path} is {file_size} bytes, "
                    f"exceeds recommended size of {_MAX_FILE_SIZE_BYTES}"
                )
            
            return ray.data.read_parquet(file_path)
        except Exception as e:
            raise DataSourceError(f"Failed to load Parquet file {file_path}: {e}") from e

    @retry_cloud_storage(max_retries=3)
    def _load_jsonl_file(self, file_path: str) -> Dataset:
        """Load a single JSONL file with retry logic.

        Args:
            file_path: Path to JSONL file

        Returns:
            Ray Dataset from JSONL file

        Raises:
            DataSourceError: If file cannot be read
        """
        try:
            # Validate file exists and is readable
            path_obj = Path(file_path)
            if not path_obj.exists():
                raise DataSourceError(f"JSONL file does not exist: {file_path}")
            
            if not path_obj.is_file():
                raise DataSourceError(f"JSONL path is not a file: {file_path}")
            
            # Check file size
            file_size = path_obj.stat().st_size
            if file_size > _MAX_FILE_SIZE_BYTES:
                logger.warning(
                    f"JSONL file {file_path} is {file_size} bytes, "
                    f"exceeds recommended size of {_MAX_FILE_SIZE_BYTES}"
                )
            
            return ray.data.read_json(file_path)
        except Exception as e:
            raise DataSourceError(f"Failed to load JSONL file {file_path}: {e}") from e

    def _load_json_file(self, file_path: str) -> Dataset:
        """Load a single JSON file.

        Args:
            file_path: Path to JSON file

        Returns:
            Ray Dataset from JSON file

        Raises:
            DataSourceError: If file cannot be read
        """
        try:
            path_obj = Path(file_path)
            if not path_obj.exists():
                raise DataSourceError(f"JSON file does not exist: {file_path}")
            
            if not path_obj.is_file():
                raise DataSourceError(f"JSON path is not a file: {file_path}")
            
            # Read JSON file
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Convert to list if single object
            if not isinstance(data, list):
                data = [data]
            
            return ray.data.from_items(data)
        except json.JSONDecodeError as e:
            raise DataSourceError(f"Invalid JSON in file {file_path}: {e}") from e
        except Exception as e:
            raise DataSourceError(f"Failed to load JSON file {file_path}: {e}") from e

    def _format_and_validate(self, dataset: Dataset) -> Dataset:
        """Format and validate GR00T data.

        Args:
            dataset: Raw dataset to format

        Returns:
            Formatted and validated dataset
        """
        def format_batch(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            """Format a batch of GR00T data."""
            formatted_items: List[Dict[str, Any]] = []
            
            for item in batch:
                try:
                    if not isinstance(item, dict):
                        logger.warning(f"Skipping non-dict item: {type(item)}")
                        if not self.validate_schema:
                            formatted_items.append({
                                "format_error": f"Item is not a dictionary: {type(item)}",
                                "raw_item": str(item),
                            })
                        continue
                    
                    formatted_item = self._format_groot_item(item)
                    formatted_items.append(formatted_item)
                except ValidationError as e:
                    logger.warning(f"Validation error for GR00T item: {e}")
                    if not self.validate_schema:
                        item["validation_error"] = str(e)
                        formatted_items.append(item)
                except Exception as e:
                    logger.warning(f"Failed to format GR00T item: {e}", exc_info=True)
                    if not self.validate_schema:
                        item["format_error"] = str(e)
                        formatted_items.append(item)
            
            return formatted_items

        return dataset.map_batches(
            format_batch,
            batch_size=self.batch_size,
            batch_format="pandas",
        )

    def _format_groot_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Format GR00T data item.

        Args:
            item: Raw GR00T data item

        Returns:
            Formatted GR00T item

        Raises:
            ValidationError: If item format is invalid
        """
        if not isinstance(item, dict):
            raise ValidationError(f"GR00T item must be a dictionary, got {type(item)}")
        
        formatted: Dict[str, Any] = {
            "data_type": item.get("data_type", "sensor"),
            "format": "groot",
            "source": item.get("source", "groot"),
        }

        # Validate data_type
        if formatted["data_type"] not in _VALID_DATA_TYPES:
            logger.warning(
                f"Invalid data_type '{formatted['data_type']}', "
                f"defaulting to 'sensor'. Valid types: {_VALID_DATA_TYPES}"
            )
            formatted["data_type"] = "sensor"

        # GR00T metadata schema
        if self.include_metadata:
            groot_metadata: Dict[str, Any] = {}
            metadata_keys = [
                "episode_id",
                "robot_id",
                "task_id",
                "instruction",
                "timestamp",
                "simulation_time",
                "episode_time",
            ]
            
            for key in metadata_keys:
                if key in item:
                    value = item[key]
                    # Validate metadata value types
                    if key in ["episode_id", "robot_id", "task_id"]:
                        if not isinstance(value, (str, int)):
                            logger.warning(
                                f"Invalid type for {key}: {type(value)}, "
                                f"expected str or int"
                            )
                            continue
                    groot_metadata[key] = value

            if groot_metadata:
                formatted["groot_metadata"] = groot_metadata

        # GR00T-specific fields with validation
        for field in ["observations", "actions", "rewards", "sensor_data"]:
            if field in item:
                value = item[field]
                # Validate field is not None
                if value is not None:
                    formatted[field] = value
                else:
                    logger.debug(f"Field {field} is None, skipping")

        # Validate GR00T schema if requested
        if self.validate_schema:
            validation_result = self._validate_groot_schema(formatted)
            formatted["groot_schema_valid"] = validation_result["is_valid"]
            if not validation_result["is_valid"]:
                formatted["groot_schema_errors"] = validation_result["errors"]
                if self.validate_schema:
                    raise ValidationError(
                        f"Schema validation failed: {validation_result['errors']}"
                    )

        return formatted

    def _validate_groot_schema(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Validate GR00T data schema.

        Args:
            item: GR00T data item

        Returns:
            Validation result dictionary with 'is_valid' and 'errors' keys
        """
        if not isinstance(item, dict):
            return {
                "is_valid": False,
                "errors": [f"Item must be a dictionary, got {type(item)}"],
            }

        errors: List[str] = []
        required_fields = ["data_type", "format"]

        # Check required fields
        for field in required_fields:
            if field not in item:
                errors.append(f"Missing required field: {field}")
            elif item[field] is None:
                errors.append(f"Required field '{field}' is None")

        # Validate GR00T metadata if present
        if "groot_metadata" in item:
            metadata = item["groot_metadata"]
            if not isinstance(metadata, dict):
                errors.append("groot_metadata must be a dictionary")
            else:
                # Validate metadata fields
                for key in ["episode_id", "robot_id", "task_id"]:
                    if key in metadata:
                        value = metadata[key]
                        if not isinstance(value, (str, int)):
                            errors.append(
                                f"groot_metadata.{key} must be string or int, "
                                f"got {type(value)}"
                            )
                        elif isinstance(value, str) and not value:
                            errors.append(f"groot_metadata.{key} cannot be empty string")

        # Validate data type
        if "data_type" in item:
            data_type = item["data_type"]
            if data_type not in _VALID_DATA_TYPES:
                errors.append(
                    f"Invalid data_type '{data_type}', "
                    f"must be one of {_VALID_DATA_TYPES}"
                )

        # Validate format
        if "format" in item and item["format"] != "groot":
            errors.append(f"Invalid format '{item['format']}', expected 'groot'")

        return {
            "is_valid": len(errors) == 0,
            "errors": errors,
        }


def test() -> None:
    """Test GR00T datasource with example data."""
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
    
    # Create test GR00T JSONL file
    test_file = test_data_dir / "test_groot.jsonl"
    test_data = [
        {
            "episode_id": "episode_001",
            "robot_id": "robot_1",
            "task_id": "task_pick",
            "data_type": "sensor",
            "format": "groot",
            "groot_metadata": {
                "episode_id": "episode_001",
                "robot_id": "robot_1",
                "task_id": "task_pick",
            },
            "sensor_data": {"joint_positions": [0.1, 0.2, 0.3]},
        },
        {
            "episode_id": "episode_002",
            "robot_id": "robot_1",
            "task_id": "task_place",
            "data_type": "sensor",
            "format": "groot",
            "groot_metadata": {
                "episode_id": "episode_002",
                "robot_id": "robot_1",
                "task_id": "task_place",
            },
            "sensor_data": {"joint_positions": [0.4, 0.5, 0.6]},
        },
    ]
    
    # Write test file
    import json
    with open(test_file, "w") as f:
        for item in test_data:
            f.write(json.dumps(item) + "\n")
    
    logger.info(f"Testing GR00TDatasource with {test_file}")
    
    try:
        # Test GR00T datasource
        loader = GR00TDatasource(
            groot_path=str(test_data_dir),
            max_files=10,
        )
        dataset = loader.load()
        
        # Verify dataset
        count = 0
        for batch in dataset.iter_batches(batch_size=10, prefetch_batches=0):
            count += len(batch) if hasattr(batch, '__len__') else 1
        
        logger.info(f"GR00TDatasource test passed: loaded {count} items")
        
        # Cleanup
        if test_file.exists():
            test_file.unlink()
    except Exception as e:
        logger.error(f"GR00TDatasource test failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    test()
