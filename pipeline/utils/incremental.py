"""Incremental processing utilities for large-scale datasets.

Tracks processed data to avoid reprocessing and supports incremental updates.
Critical for GR00T: Internet-scale datasets require incremental processing.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class IncrementalProcessor:
    """Manages incremental processing of datasets.

    Tracks which data has been processed to avoid reprocessing unchanged data.
    """

    def __init__(
        self,
        state_dir: str = ".incremental_state",
        enable_incremental: bool = True,
    ):
        """Initialize incremental processor.

        Args:
            state_dir: Directory to store processing state
            enable_incremental: Whether incremental processing is enabled
        """
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.enable_incremental = enable_incremental
        self.processed_files: Set[str] = set()
        self.processed_hashes: Set[str] = set()
        self._load_state()

    def _load_state(self) -> None:
        """Load processing state from disk."""
        if not self.enable_incremental:
            return

        state_file = self.state_dir / "processed_state.json"
        if state_file.exists():
            try:
                with open(state_file) as f:
                    state = json.load(f)
                    self.processed_files = set(state.get("processed_files", []))
                    self.processed_hashes = set(state.get("processed_hashes", []))
                logger.info(
                    f"Loaded incremental state: {len(self.processed_files)} files, "
                    f"{len(self.processed_hashes)} hashes"
                )
            except (IOError, OSError, json.JSONDecodeError) as e:
                logger.warning(f"Failed to load incremental state: {e}")

    def _save_state(self) -> None:
        """Save processing state to disk."""
        if not self.enable_incremental:
            return

        state_file = self.state_dir / "processed_state.json"
        try:
            state = {
                "processed_files": list(self.processed_files),
                "processed_hashes": list(self.processed_hashes),
                "last_updated": datetime.utcnow().isoformat(),
            }
            with open(state_file, "w") as f:
                json.dump(state, f, indent=2)
        except (IOError, OSError) as e:
            logger.error(f"Failed to save incremental state: {e}")

    def compute_file_hash(self, file_path: str) -> str:
        """Compute hash of a file.

        Args:
            file_path: Path to file

        Returns:
            SHA256 hash of file
        """
        try:
            # For large files, use modification time and size as proxy
            # Full hash computation can be expensive for very large files
            stat = os.stat(file_path)
            hash_input = f"{file_path}:{stat.st_mtime}:{stat.st_size}"
            return hashlib.sha256(hash_input.encode()).hexdigest()
        except (OSError, IOError) as e:
            logger.warning(f"Failed to compute hash for {file_path}: {e}")
            return ""

    def is_processed(self, file_path: str, content_hash: Optional[str] = None) -> bool:
        """Check if a file has been processed.

        Args:
            file_path: Path to file
            content_hash: Optional content hash (computed if None)

        Returns:
            True if file has been processed
        """
        if not self.enable_incremental:
            return False

        # Check by file path
        if file_path in self.processed_files:
            return True

        # Check by content hash if provided
        if content_hash:
            if content_hash in self.processed_hashes:
                return True

        return False

    def mark_processed(
        self, file_path: str, content_hash: Optional[str] = None
    ) -> None:
        """Mark a file as processed.

        Args:
            file_path: Path to file
            content_hash: Optional content hash
        """
        if not self.enable_incremental:
            return

        self.processed_files.add(file_path)
        if content_hash:
            self.processed_hashes.add(content_hash)

    def filter_unprocessed(
        self, file_paths: List[str], compute_hashes: bool = True
    ) -> List[str]:
        """Filter list of files to only unprocessed ones.

        Args:
            file_paths: List of file paths
            compute_hashes: Whether to compute hashes for checking

        Returns:
            List of unprocessed file paths
        """
        if not self.enable_incremental:
            return file_paths

        unprocessed = []
        for file_path in file_paths:
            content_hash = None
            if compute_hashes:
                content_hash = self.compute_file_hash(file_path)

            if not self.is_processed(file_path, content_hash):
                unprocessed.append(file_path)
                # Mark as processed immediately to avoid duplicates
                self.mark_processed(file_path, content_hash)

        logger.info(
            f"Filtered {len(file_paths)} files: {len(unprocessed)} unprocessed, "
            f"{len(file_paths) - len(unprocessed)} already processed"
        )

        return unprocessed

    def get_processed_count(self) -> int:
        """Get count of processed files.

        Returns:
            Number of processed files
        """
        return len(self.processed_files)

    def clear_state(self) -> None:
        """Clear all processing state."""
        self.processed_files.clear()
        self.processed_hashes.clear()
        self._save_state()
        logger.info("Cleared incremental processing state")

    def save(self) -> None:
        """Save current state to disk."""
        self._save_state()


class ChangeDetector:
    """Detects changes in datasets for incremental processing."""

    def __init__(self, state_dir: str = ".incremental_state"):
        """Initialize change detector.

        Args:
            state_dir: Directory to store change detection state
        """
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def detect_changes(
        self,
        input_paths: List[str],
        previous_version_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Detect changes in input paths.

        Args:
            input_paths: List of input paths to check
            previous_version_id: Optional previous version ID to compare against

        Returns:
            Dictionary with change information
        """
        current_hashes = {}
        for path in input_paths:
            if os.path.exists(path):
                current_hashes[path] = self._compute_path_hash(path)

        # Load previous hashes if version ID provided
        previous_hashes = {}
        if previous_version_id:
            previous_hashes = self._load_version_hashes(previous_version_id)

        # Compare hashes
        added = []
        modified = []
        removed = []
        unchanged = []

        for path, current_hash in current_hashes.items():
            if path not in previous_hashes:
                added.append(path)
            elif previous_hashes[path] != current_hash:
                modified.append(path)
            else:
                unchanged.append(path)

        for path in previous_hashes:
            if path not in current_hashes:
                removed.append(path)

        return {
            "added": added,
            "modified": modified,
            "removed": removed,
            "unchanged": unchanged,
            "current_hashes": current_hashes,
        }

    def _compute_path_hash(self, path: str) -> str:
        """Compute hash for a path (file or directory).

        Args:
            path: Path to file or directory

        Returns:
            Hash string
        """
        if os.path.isfile(path):
            return self._compute_file_hash(path)
        elif os.path.isdir(path):
            return self._compute_dir_hash(path)
        else:
            return ""

    def _compute_file_hash(self, file_path: str) -> str:
        """Compute hash of a file.

        Args:
            file_path: Path to file

        Returns:
            SHA256 hash
        """
        try:
            stat = os.stat(file_path)
            hash_input = f"{file_path}:{stat.st_mtime}:{stat.st_size}"
            return hashlib.sha256(hash_input.encode()).hexdigest()
        except (OSError, IOError):
            return ""

    def _compute_dir_hash(self, dir_path: str) -> str:
        """Compute hash of a directory.

        Args:
            dir_path: Path to directory

        Returns:
            SHA256 hash
        """
        try:
            # Hash all files in directory
            file_hashes = []
            for root, dirs, files in os.walk(dir_path):
                for file in sorted(files):
                    file_path = os.path.join(root, file)
                    file_hash = self._compute_file_hash(file_path)
                    if file_hash:
                        file_hashes.append(file_hash)

            hash_input = "\n".join(file_hashes)
            return hashlib.sha256(hash_input.encode()).hexdigest()
        except (OSError, IOError):
            return ""

    def _load_version_hashes(self, version_id: str) -> Dict[str, str]:
        """Load hashes from a previous version.

        Args:
            version_id: Version ID

        Returns:
            Dictionary mapping paths to hashes
        """
        version_file = self.state_dir / f"{version_id}_hashes.json"
        if version_file.exists():
            try:
                with open(version_file) as f:
                    return json.load(f)
            except (IOError, OSError, json.JSONDecodeError):
                pass
        return {}

    def save_version_hashes(
        self, version_id: str, hashes: Dict[str, str]
    ) -> None:
        """Save hashes for a version.

        Args:
            version_id: Version ID
            hashes: Dictionary mapping paths to hashes
        """
        version_file = self.state_dir / f"{version_id}_hashes.json"
        try:
            with open(version_file, "w") as f:
                json.dump(hashes, f, indent=2)
        except (IOError, OSError) as e:
            logger.error(f"Failed to save version hashes: {e}")


def create_incremental_processor(
    state_dir: str = ".incremental_state", enable_incremental: bool = True
) -> IncrementalProcessor:
    """Create an incremental processor instance.

    Args:
        state_dir: Directory for state storage
        enable_incremental: Whether incremental processing is enabled

    Returns:
        IncrementalProcessor instance
    """
    return IncrementalProcessor(state_dir=state_dir, enable_incremental=enable_incremental)


def create_change_detector(state_dir: str = ".incremental_state") -> ChangeDetector:
    """Create a change detector instance.

    Args:
        state_dir: Directory for state storage

    Returns:
        ChangeDetector instance
    """
    return ChangeDetector(state_dir=state_dir)

