"""Data corruption detection utilities.

Detects corrupted data files using checksums and validation.
Critical for GR00T: Corrupted data trains bad models.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Optional


logger = logging.getLogger(__name__)


class CorruptionDetector:
    """Detects data corruption using checksums and validation."""

    def __init__(self, compute_checksums: bool = True, validate_structure: bool = True):
        """Initialize corruption detector.

        Args:
            compute_checksums: Whether to compute and verify checksums
            validate_structure: Whether to validate data structure
        """
        self.compute_checksums = compute_checksums
        self.validate_structure = validate_structure

    def detect_corruption(self, item: dict[str, Any]) -> dict[str, Any]:
        """Detect corruption in a data item.

        Args:
            item: Data item

        Returns:
            Dictionary with corruption detection results
        """
        corruption_flags: list[str] = []
        warnings: list[str] = []

        # Check for error fields
        if "error" in item:
            corruption_flags.append("error_field_present")

        # Validate structure
        if self.validate_structure:
            structure_issues = self._validate_structure(item)
            corruption_flags.extend(structure_issues)

        # Compute checksum if requested
        if self.compute_checksums:
            checksum = self._compute_checksum(item)
            if checksum:
                item["data_checksum"] = checksum

        is_corrupted = len(corruption_flags) > 0

        return {
            "is_corrupted": is_corrupted,
            "corruption_flags": corruption_flags,
            "corruption_warnings": warnings,
            "corruption_detected": is_corrupted,
        }

    def _validate_structure(self, item: dict[str, Any]) -> list[str]:
        """Validate data structure.

        Args:
            item: Data item

        Returns:
            List of structure issues
        """
        issues = []

        # Check for required top-level fields
        if "data_type" not in item and "format" not in item:
            issues.append("missing_data_type_or_format")

        # Validate based on data type
        data_type = item.get("data_type") or item.get("format", "unknown")

        if data_type == "video":
            if "frames" not in item and "video" not in item and "bytes" not in item:
                issues.append("video_missing_data")

        elif data_type == "sensor":
            if "sensor_data" not in item and "observations" not in item:
                issues.append("sensor_missing_data")

        elif data_type == "text":
            if "text" not in item and "content" not in item:
                issues.append("text_missing_data")

        return issues

    def _compute_checksum(self, item: dict[str, Any]) -> Optional[str]:
        """Compute checksum for data item using SHA256.

        Args:
            item: Data item

        Returns:
            Checksum string or None
        """
        try:
            # Create deterministic representation
            checksum_data = {
                "data_type": item.get("data_type"),
                "format": item.get("format"),
                "path": item.get("path"),
            }

            # Include data hash if available
            if "bytes" in item:
                data_bytes = item["bytes"]
                if isinstance(data_bytes, bytes):
                    # Use SHA256 for better security than MD5
                    checksum_data["data_hash"] = hashlib.sha256(data_bytes).hexdigest()
            
            # Include key data fields for validation
            for key in ["episode_id", "robot_id", "timestamp", "step"]:
                if key in item:
                    checksum_data[key] = item[key]

            import json

            checksum_str = json.dumps(checksum_data, sort_keys=True)
            # Use full SHA256 hash for better collision resistance
            return hashlib.sha256(checksum_str.encode()).hexdigest()
        except (TypeError, ValueError, json.JSONEncodeError) as e:
            logger.warning(f"Failed to compute checksum: {e}")
            return None


def detect_corruption_batch(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Detect corruption in a batch of items.

    Args:
        batch: List of data items

    Returns:
        List of items with corruption detection results
    """
    detector = CorruptionDetector()
    detected = []
    for item in batch:
        corruption_result = detector.detect_corruption(item)
        item.update(corruption_result)
        detected.append(item)
    return detected

