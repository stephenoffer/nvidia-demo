"""Base class for custom file-based datasources.

Implements Ray Data FileBasedDatasource pattern following the official API.
Subclasses must implement _read_stream() which yields Block objects.
See: https://docs.ray.io/en/latest/data/loading-data.html#custom-datasources
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Iterator, List, Optional, Union

from ray.data.block import Block
from ray.data.datasource.file_based_datasource import (
    FileBasedDatasource as RayFileBasedDatasource,
)
from ray.data.datasource.file_meta_provider import DefaultFileMetadataProvider

from pipeline.exceptions import DataSourceError
from pipeline.utils.decorators import handle_errors, log_execution_time

if TYPE_CHECKING:
    import pyarrow

logger = logging.getLogger(__name__)


class FileBasedDatasource(RayFileBasedDatasource):
    """Base class for custom file-based datasources.

    Extends Ray Data's FileBasedDatasource following the official API.
    Subclasses must implement _read_stream() which takes a pyarrow.NativeFile
    and yields Block objects using ArrowBlockBuilder.

    Metadata handling:
    - Path metadata: Ray Data automatically adds "path" column when include_paths=True.
      Subclasses should not manually add "path" fields to records.
    - Format/data_type: Do not add generic "format" or "data_type" fields.
      Return clean data with domain-specific columns.
    - Domain metadata: Only add domain-specific metadata that provides value
      (e.g., "dataset_name" for HDF5, "topic" for ROS bags).
    """

    def __init__(
        self,
        paths: Union[str, List[str]],
        include_paths: bool = True,
        meta_provider: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize file-based datasource.

        Args:
            paths: File path(s) or directory path(s) to read
            include_paths: Whether to include file paths in output.
                When True, Ray Data automatically adds a "path" column to each record.
                Subclasses should NOT manually add "path" fields.
            meta_provider: File metadata provider (default: DefaultFileMetadataProvider)
            **kwargs: Additional FileBasedDatasource options

        Raises:
            DataSourceError: If paths are invalid
        """
        # Validate paths
        if not paths:
            raise DataSourceError("paths cannot be empty")

        if isinstance(paths, str):
            paths_list = [paths]
        elif isinstance(paths, list):
            paths_list = paths
        else:
            raise DataSourceError(f"paths must be str or list[str], got {type(paths)}")

        if not paths_list:
            raise DataSourceError("paths list cannot be empty")

        # Use default metadata provider if not specified
        if meta_provider is None:
            meta_provider = DefaultFileMetadataProvider()

        try:
            super().__init__(
                paths=paths,
                include_paths=include_paths,
                meta_provider=meta_provider,
                **kwargs,
            )
        except Exception as e:
            raise DataSourceError(f"Failed to initialize datasource: {e}") from e

    def _validate_file_handle(self, f: "pyarrow.NativeFile", path: str) -> None:
        """Validate file handle and path before reading.
        
        Args:
            f: pyarrow.NativeFile handle for the file
            path: Path to the file being read
        
        Raises:
            DataSourceError: If file handle or path is invalid
        """
        if f is None:
            raise DataSourceError(f"File handle is None for path: {path}")
        if not path or not path.strip():
            raise DataSourceError("Path cannot be empty")

    @log_execution_time
    @handle_errors(error_class=DataSourceError)
    def _read_stream(
        self, f: "pyarrow.NativeFile", path: str
    ) -> Iterator[Block]:
        """Streaming read a single file.

        Must be implemented by subclasses. This method is called by Ray Data
        for each file, and should yield Block objects.

        IMPORTANT: Return clean data without metadata wrapping:
        - Do NOT add "path" field - Ray Data handles this automatically via include_paths
        - Do NOT add generic "format" or "data_type" fields
        - Only include domain-specific columns and metadata

        Args:
            f: pyarrow.NativeFile handle for the file
            path: Path to the file being read (provided for reference, but don't add to records)

        Yields:
            Block objects containing clean data from the file

        Raises:
            DataSourceError: If file reading fails
            NotImplementedError: If subclass doesn't implement this method
        """
        self._validate_file_handle(f, path)
        raise NotImplementedError(
            "Subclasses must implement _read_stream() following Ray Data API"
        )

