"""Resource management utilities for infrastructure reliability.

Provides connection pooling, cleanup, health checks, and resource monitoring
to ensure production-grade reliability and prevent resource leaks.
"""

from __future__ import annotations

import atexit
import logging
import os
import shutil
import signal
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

logger = logging.getLogger(__name__)


class ResourceManager:
    """Manages resources and ensures proper cleanup on shutdown.

    Tracks file handles, connections, temp files, and other resources
    to prevent leaks and ensure graceful shutdown.
    """

    def __init__(self):
        """Initialize resource manager."""
        self._temp_files: set[str] = set()
        self._cleanup_handlers: list[Callable[[], None]] = []
        self._shutdown_registered = False
        self._register_shutdown_handlers()

    def _register_shutdown_handlers(self) -> None:
        """Register signal handlers for graceful shutdown."""
        if self._shutdown_registered:
            return

        def _signal_handler(signum: int, frame: Any) -> None:
            logger.warning(f"Received signal {signum}, initiating graceful shutdown")
            self.cleanup_all()
            sys.exit(0)

        signal.signal(signal.SIGTERM, _signal_handler)
        signal.signal(signal.SIGINT, _signal_handler)
        atexit.register(self.cleanup_all)
        self._shutdown_registered = True

    def register_temp_file(self, path: str) -> None:
        """Register a temporary file for cleanup.

        Args:
            path: Path to temporary file
        """
        self._temp_files.add(path)

    def register_cleanup(self, handler: Callable[[], None]) -> None:
        """Register a cleanup handler to call on shutdown.

        Args:
            handler: Function to call for cleanup
        """
        self._cleanup_handlers.append(handler)

    def cleanup_all(self) -> None:
        """Clean up all registered resources."""
        logger.info("Cleaning up resources...")

        # Run registered cleanup handlers
        for handler in self._cleanup_handlers:
            try:
                handler()
            except Exception as e:
                logger.error(f"Error in cleanup handler: {e}", exc_info=True)

        # Clean up temp files
        for temp_path in list(self._temp_files):
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                    logger.debug(f"Cleaned up temp file: {temp_path}")
            except Exception as e:
                logger.warning(f"Failed to delete temp file {temp_path}: {e}")

        self._temp_files.clear()
        self._cleanup_handlers.clear()

    @contextmanager
    def temp_file(self, suffix: str = "", prefix: str = "pipeline_") -> Iterator[str]:
        """Create a temporary file that's automatically cleaned up.

        Args:
            suffix: File suffix
            prefix: File prefix

        Yields:
            Path to temporary file
        """
        fd, path = tempfile.mkstemp(suffix=suffix, prefix=prefix)
        try:
            os.close(fd)  # Close file descriptor, we'll reopen if needed
            self.register_temp_file(path)
            yield path
        finally:
            try:
                if os.path.exists(path):
                    os.unlink(path)
                    self._temp_files.discard(path)
            except Exception as e:
                logger.warning(f"Failed to delete temp file {path}: {e}")


# Global resource manager instance
_resource_manager = ResourceManager()


def get_resource_manager() -> ResourceManager:
    """Get the global resource manager instance.

    Returns:
        ResourceManager instance
    """
    return _resource_manager


def check_disk_space(path: str, required_bytes: int) -> tuple[bool, int]:
    """Check if sufficient disk space is available.

    Args:
        path: Path to check (directory or file)
        required_bytes: Required bytes

    Returns:
        Tuple of (has_space, available_bytes)
    """
    try:
        path_obj = Path(path)
        if path_obj.is_file():
            path_obj = path_obj.parent
        elif not path_obj.exists():
            # Create parent directory to check
            path_obj = path_obj.parent

        stat = shutil.disk_usage(str(path_obj))
        available = stat.free
        has_space = available >= required_bytes

        if not has_space:
            logger.warning(
                f"Insufficient disk space at {path}: "
                f"required {required_bytes}, available {available}"
            )

        return has_space, available
    except Exception as e:
        logger.error(f"Failed to check disk space at {path}: {e}")
        # Assume space is available if check fails
        return True, 0


def validate_path(path: str, check_exists: bool = False, check_writable: bool = False) -> None:
    """Validate a file or directory path.

    Args:
        path: Path to validate
        check_exists: Whether to check if path exists
        check_writable: Whether to check if path is writable

    Raises:
        ValueError: If path is invalid
        PermissionError: If path is not writable
        FileNotFoundError: If path doesn't exist and check_exists=True
    """
    if not path:
        raise ValueError("Path cannot be empty")

    path_obj = Path(path)

    # Check for path traversal
    if ".." in str(path_obj):
        raise ValueError(f"Path contains '..' which is not allowed: {path}")

    if check_exists and not path_obj.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")

    if check_writable:
        if path_obj.exists():
            if path_obj.is_file() and not os.access(path_obj, os.W_OK):
                raise PermissionError(f"File is not writable: {path}")
            elif path_obj.is_dir() and not os.access(path_obj, os.W_OK):
                raise PermissionError(f"Directory is not writable: {path}")
        else:
            # Check if parent directory is writable
            parent = path_obj.parent
            if parent.exists() and not os.access(parent, os.W_OK):
                raise PermissionError(f"Parent directory is not writable: {parent}")

