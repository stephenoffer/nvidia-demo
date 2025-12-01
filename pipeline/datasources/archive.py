"""Custom datasource for archive files (ZIP, TAR, etc.).

Archive formats are commonly used in robotics for packaging datasets,
logs, and compressed data collections.
Follows Ray Data FileBasedDatasource API.
"""

from __future__ import annotations

import logging
import os
import tarfile  # https://docs.python.org/3/library/tarfile.html
import zipfile  # https://docs.python.org/3/library/zipfile.html
from typing import TYPE_CHECKING, Any, Iterator, Union

from ray.data._internal.arrow_block import ArrowBlockBuilder
from ray.data.block import Block

from pipeline.datasources.base import FileBasedDatasource
from pipeline.exceptions import DataSourceError, ConfigurationError
from pipeline.utils.decorators import handle_errors, log_execution_time

if TYPE_CHECKING:
    import pyarrow

logger = logging.getLogger(__name__)

# Constants
_MAX_EXTRACT_SIZE_BYTES = 10 * 1024 * 1024  # 10MB
_MAX_ENTRIES_PER_ARCHIVE = 100000
_MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024 * 1024  # 10GB


class ArchiveDatasource(FileBasedDatasource):
    """Custom datasource for reading archive files.

    Supports ZIP, TAR, GZIP, and other archive formats commonly used
    for packaging robotics datasets and logs.
    """

    def __init__(
        self,
        paths: Union[str, list[str]],
        extract: bool = False,
        max_entries: Optional[int] = None,
        **kwargs: Any,
    ):
        """Initialize archive datasource.

        Args:
            paths: Archive file path(s) or directory path(s)
            extract: Whether to extract archive contents (vs metadata only)
            max_entries: Maximum number of entries to process (None = unlimited)
            **kwargs: Additional FileBasedDatasource options

        Raises:
            ValueError: If parameters are invalid
        """
        super().__init__(paths=paths, **kwargs)
        
        # Validate parameters
        if max_entries is not None and max_entries <= 0:
            raise ConfigurationError(f"max_entries must be positive, got {max_entries}")
        
        self.extract = bool(extract)
        self.max_entries = max_entries

    @log_execution_time
    @handle_errors(error_class=DataSourceError)
    def _read_stream(
        self, f: "pyarrow.NativeFile", path: str
    ) -> Iterator[Block]:
        """Read archive file and yield entry blocks.

        Args:
            f: pyarrow.NativeFile handle (archives require direct file access)
            path: Path to archive file

        Yields:
            Block objects (pyarrow.Table) with archive entry data

        Raises:
            DataSourceError: If reading fails

        Note:
            Archive libraries require direct file access. For cloud storage,
            files must be copied locally first.
        """
        self._validate_file_handle(f, path)
        
        # Validate file exists (for local files)
        if not path.startswith(("s3://", "gs://", "hdfs://", "abfss://")):
            if not os.path.exists(path):
                raise DataSourceError(f"Archive file does not exist: {path}")
            
            if not os.path.isfile(path):
                raise DataSourceError(f"Archive path is not a file: {path}")
            
            # Check file size
            file_size = os.path.getsize(path)
            if file_size > _MAX_FILE_SIZE_BYTES:
                raise DataSourceError(
                    f"Archive file {path} is {file_size} bytes, "
                    f"exceeds maximum size of {_MAX_FILE_SIZE_BYTES}"
                )
        
        path_lower = path.lower()

        try:
            if path_lower.endswith((".zip", ".whl")):
                yield from self._read_zip(path)
            elif path_lower.endswith((".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2")):
                yield from self._read_tar(path)
            else:
                raise DataSourceError(f"Unsupported archive format: {path}")

        except Exception as e:
            logger.error(f"Error reading archive {path}: {e}", exc_info=True)
            # Re-raise exception instead of wrapping in block
            raise DataSourceError(f"Failed to read archive {path}: {e}") from e

    def _read_zip(self, path: str) -> Iterator[Block]:
        """Read ZIP archive.

        Args:
            path: Path to ZIP file

        Yields:
            Block objects with ZIP entry data

        Raises:
            DataSourceError: If reading fails
        """
        if not os.path.exists(path):
            raise DataSourceError(f"ZIP file does not exist: {path}")
        
        builder = ArrowBlockBuilder()
        entry_count = 0

        try:
            with zipfile.ZipFile(path, "r") as zip_file:
                # Validate ZIP file
                try:
                    zip_file.testzip()
                except Exception as e:
                    raise DataSourceError(f"ZIP file {path} is corrupted: {e}") from e
                
                file_list = zip_file.infolist()
                
                # Apply max_entries limit
                if self.max_entries is not None:
                    file_list = file_list[:self.max_entries]
                
                for file_info in file_list:
                    # Check max_entries limit
                    if self.max_entries is not None and entry_count >= self.max_entries:
                        logger.info(f"Reached max_entries limit ({self.max_entries})")
                        break
                    
                    # Return clean data without metadata wrapping
                    entry_data = {
                        "entry_name": file_info.filename,
                        "size": file_info.file_size,
                        "compressed_size": file_info.compress_size,
                        "is_dir": file_info.is_dir(),
                        "entry_index": entry_count,
                    }

                    if self.extract and not file_info.is_dir():
                        try:
                            # Limit extraction size to avoid OOM
                            if file_info.file_size > _MAX_EXTRACT_SIZE_BYTES:
                                logger.warning(
                                    f"Skipping extraction of {file_info.filename} "
                                    f"({file_info.file_size} bytes exceeds {_MAX_EXTRACT_SIZE_BYTES} limit)"
                                )
                                entry_data["extract_error"] = "File too large to extract"
                            else:
                                content = zip_file.read(file_info.filename)
                                entry_data["content"] = content
                                entry_data["content_size"] = len(content)
                        except Exception as e:
                            logger.warning(f"Failed to extract {file_info.filename}: {e}")
                            entry_data["extract_error"] = str(e)

                    builder.add(entry_data)
                    entry_count += 1
                    
                    # Yield block periodically
                    if builder.num_rows() >= _MAX_ENTRIES_PER_ARCHIVE:
                        yield builder.build()
                        builder = ArrowBlockBuilder()

            if builder.num_rows() > 0:
                yield builder.build()
            
            logger.info(f"Processed {entry_count} entries from ZIP archive: {path}")

        except zipfile.BadZipFile as e:
            raise DataSourceError(f"Invalid ZIP file {path}: {e}") from e
        except Exception as e:
            raise DataSourceError(f"Failed to read ZIP archive {path}: {e}") from e

    def _read_tar(self, path: str) -> Iterator[Block]:
        """Read TAR archive.

        Args:
            path: Path to TAR file

        Yields:
            Block objects with TAR entry data

        Raises:
            DataSourceError: If reading fails
        """
        if not os.path.exists(path):
            raise DataSourceError(f"TAR file does not exist: {path}")
        
        # Determine compression mode
        mode = "r"
        if path.endswith(".gz") or path.endswith(".tgz"):
            mode = "r:gz"
        elif path.endswith(".bz2") or path.endswith(".tbz2"):
            mode = "r:bz2"

        builder = ArrowBlockBuilder()
        entry_count = 0

        try:
            with tarfile.open(path, mode) as tar_file:
                members = tar_file.getmembers()
                
                # Apply max_entries limit
                if self.max_entries is not None:
                    members = members[:self.max_entries]
                
                for member in members:
                    # Check max_entries limit
                    if self.max_entries is not None and entry_count >= self.max_entries:
                        logger.info(f"Reached max_entries limit ({self.max_entries})")
                        break
                    
                    # Return clean data without metadata wrapping
                    entry_data = {
                        "entry_name": member.name,
                        "size": member.size,
                        "is_dir": member.isdir(),
                        "is_file": member.isfile(),
                        "entry_index": entry_count,
                    }

                    if self.extract and member.isfile():
                        try:
                            # Limit extraction size to avoid OOM
                            if member.size > _MAX_EXTRACT_SIZE_BYTES:
                                logger.warning(
                                    f"Skipping extraction of {member.name} "
                                    f"({member.size} bytes exceeds {_MAX_EXTRACT_SIZE_BYTES} limit)"
                                )
                                entry_data["extract_error"] = "File too large to extract"
                            else:
                                file_obj = tar_file.extractfile(member)
                                if file_obj:
                                    content = file_obj.read()
                                    entry_data["content"] = content
                                    entry_data["content_size"] = len(content)
                        except Exception as e:
                            logger.warning(f"Failed to extract {member.name}: {e}")
                            entry_data["extract_error"] = str(e)

                    builder.add(entry_data)
                    entry_count += 1
                    
                    # Yield block periodically
                    if builder.num_rows() >= _MAX_ENTRIES_PER_ARCHIVE:
                        yield builder.build()
                        builder = ArrowBlockBuilder()

            if builder.num_rows() > 0:
                yield builder.build()
            
            logger.info(f"Processed {entry_count} entries from TAR archive: {path}")

        except tarfile.TarError as e:
            raise DataSourceError(f"Invalid TAR file {path}: {e}") from e
        except Exception as e:
            raise DataSourceError(f"Failed to read TAR archive {path}: {e}") from e


def test() -> None:
    """Test archive datasource with example data."""
    import zipfile
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
    
    # Create test ZIP file
    test_file = test_data_dir / "test_archive.zip"
    
    # Create a simple ZIP file with test data
    with zipfile.ZipFile(test_file, "w") as zf:
        zf.writestr("file1.txt", "Test content 1")
        zf.writestr("file2.txt", "Test content 2")
        zf.writestr("subdir/file3.txt", "Test content 3")
    
    logger.info(f"Testing ArchiveDatasource with {test_file}")
    
    try:
        # Test archive datasource
        datasource = ArchiveDatasource(paths=str(test_file), extract=False)
        dataset = ray.data.read_datasource(datasource)
        
        # Verify dataset
        count = 0
        for batch in dataset.iter_batches(batch_size=10, prefetch_batches=0):
            count += len(batch) if hasattr(batch, '__len__') else 1
        
        logger.info(f"ArchiveDatasource test passed: loaded {count} entries")
        
        # Cleanup
        if test_file.exists():
            test_file.unlink()
    except Exception as e:
        logger.error(f"ArchiveDatasource test failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    test()
