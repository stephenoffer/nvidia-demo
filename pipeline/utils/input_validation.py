"""Input validation and sanitization utilities.

Validates and sanitizes all pipeline inputs to prevent security vulnerabilities
and ensure data integrity. Consolidates functionality from input_sanitization.py.

Critical for GR00T: Input validation prevents security issues and data corruption.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class InputValidator:
    """Validates and sanitizes pipeline inputs.
    
    Consolidates InputValidator and InputSanitizer functionality.
    """

    # Allowed path schemes
    ALLOWED_SCHEMES = {"file", "s3", "gs", "hdfs", "abfss", "http", "https"}

    # Dangerous path patterns (includes XSS and injection patterns)
    DANGEROUS_PATTERNS = [
        r"\.\./",  # Path traversal
        r"\.\.\\",  # Windows path traversal
        r"^/etc/",  # System directories
        r"^/usr/",  # System directories
        r"^C:\\Windows",  # Windows system
        r"<script",  # XSS
        r"javascript:",  # JavaScript injection
        r"on\w+\s*=",  # Event handlers
        r"eval\s*\(",  # Code execution
        r"exec\s*\(",  # Code execution
        r"__import__",  # Python import injection
    ]

    def __init__(self, strict: bool = True):
        """Initialize input validator.

        Args:
            strict: Whether to use strict validation (reject invalid inputs)
        """
        self.strict = strict

    def validate_path(self, path: str) -> tuple[bool, str]:
        """Validate a file path.

        Args:
            path: Path to validate

        Returns:
            Tuple (is_valid, error_message)
        """
        if not isinstance(path, str):
            return False, "Path must be a string"

        if not path:
            return False, "Path cannot be empty"

        # Check for dangerous patterns
        for pattern in self.DANGEROUS_PATTERNS:
            if re.search(pattern, path):
                return False, f"Path contains dangerous pattern: {pattern}"

        # Validate URL scheme if present
        if "://" in path:
            parsed = urlparse(path)
            if parsed.scheme not in self.ALLOWED_SCHEMES:
                return False, f"Unsupported URL scheme: {parsed.scheme}"

        # Check path length
        if len(path) > 4096:
            return False, "Path too long (max 4096 characters)"

        return True, ""

    def sanitize_path(self, path: str, allow_absolute: bool = False) -> str:
        """Sanitize a file path.

        Args:
            path: Path to sanitize
            allow_absolute: Whether to allow absolute paths

        Returns:
            Sanitized path

        Raises:
            ValueError: If path is invalid or dangerous (when strict=True)
        """
        if not isinstance(path, str):
            if self.strict:
                raise TypeError("Path must be a string")
            return ""

        # Check for dangerous patterns
        for pattern in self.DANGEROUS_PATTERNS:
            if re.search(pattern, path, re.IGNORECASE):
                if self.strict:
                    raise ValueError(f"Dangerous pattern detected in path: {pattern}")
                # Remove dangerous patterns if not strict
                path = re.sub(pattern, "", path, flags=re.IGNORECASE)

        # Normalize path
        try:
            normalized = os.path.normpath(path)
            
            # Check for path traversal
            if ".." in normalized:
                if self.strict:
                    raise ValueError("Path traversal detected")
                # Remove path traversal if not strict
                normalized = normalized.replace("..", "")
            
            # Check absolute paths
            if os.path.isabs(normalized) and not allow_absolute:
                if self.strict:
                    raise ValueError("Absolute paths not allowed")
                # Make relative if not strict
                normalized = os.path.relpath(normalized)
            
            return normalized
        except (OSError, ValueError) as e:
            if self.strict:
                raise
            return path

    def validate_config(self, config: dict[str, Any]) -> tuple[bool, list[str]]:
        """Validate pipeline configuration.

        Args:
            config: Configuration dictionary

        Returns:
            Tuple (is_valid, list_of_errors)
        """
        errors = []

        # Validate input paths
        if "input_paths" in config:
            input_paths = config["input_paths"]
            if not isinstance(input_paths, list):
                errors.append("input_paths must be a list")
            else:
                for i, path in enumerate(input_paths):
                    is_valid, error_msg = self.validate_path(str(path))
                    if not is_valid:
                        errors.append(f"input_paths[{i}]: {error_msg}")

        # Validate output path
        if "output_path" in config:
            output_path = config.get("output_path")
            if output_path:
                is_valid, error_msg = self.validate_path(str(output_path))
                if not is_valid:
                    errors.append(f"output_path: {error_msg}")

        # Validate numeric fields
        numeric_fields = ["num_gpus", "num_cpus", "batch_size", "checkpoint_interval"]
        for field in numeric_fields:
            if field in config:
                value = config[field]
                if not isinstance(value, (int, float)):
                    errors.append(f"{field} must be numeric")
                elif value < 0:
                    errors.append(f"{field} must be non-negative")

        return len(errors) == 0, errors

    def sanitize_string(self, text: str, max_length: Optional[int] = None) -> str:
        """Sanitize a string input.

        Args:
            text: String to sanitize
            max_length: Maximum allowed length (None = no limit)

        Returns:
            Sanitized string

        Raises:
            ValueError: If string is invalid (when strict=True)
        """
        if not isinstance(text, str):
            if self.strict:
                raise TypeError("Value must be a string")
            return ""

        # Check length
        if max_length and len(text) > max_length:
            if self.strict:
                raise ValueError(f"String too long: {len(text)} > {max_length}")
            text = text[:max_length]

        # Remove dangerous patterns
        sanitized = text
        for pattern in self.DANGEROUS_PATTERNS:
            sanitized = re.sub(pattern, "", sanitized, flags=re.IGNORECASE)

        # Remove null bytes
        sanitized = sanitized.replace("\x00", "")

        # Remove control characters (except newlines and tabs)
        sanitized = re.sub(r"[\x00-\x08\x0B-\x0C\x0E-\x1F]", "", sanitized)

        return sanitized

    def sanitize_url(self, url: str, allowed_schemes: Optional[list[str]] = None) -> str:
        """Sanitize URL.

        Args:
            url: URL to sanitize
            allowed_schemes: List of allowed URL schemes (default: http, https, s3, gs)

        Returns:
            Sanitized URL

        Raises:
            ValueError: If URL is invalid or uses disallowed scheme
        """
        if not isinstance(url, str):
            if self.strict:
                raise TypeError("URL must be a string")
            return ""

        if allowed_schemes is None:
            allowed_schemes = list(self.ALLOWED_SCHEMES)

        parsed = urlparse(url)

        # Check scheme
        if parsed.scheme and parsed.scheme not in allowed_schemes:
            if self.strict:
                raise ValueError(f"URL scheme '{parsed.scheme}' not allowed. Allowed: {allowed_schemes}")
            return ""

        # Sanitize path component
        if parsed.path:
            sanitized_path = self.sanitize_path(parsed.path, allow_absolute=False)
            # Reconstruct URL
            url = url.replace(parsed.path, sanitized_path)

        return url


def validate_inputs(input_paths: list[str], output_path: str) -> tuple[bool, list[str]]:
    """Validate pipeline inputs.

    Args:
        input_paths: List of input paths
        output_path: Output path

    Returns:
        Tuple (is_valid, list_of_errors)
    """
    validator = InputValidator(strict=True)
    errors = []

    # Validate input paths
    for i, path in enumerate(input_paths):
        is_valid, error_msg = validator.validate_path(path)
        if not is_valid:
            errors.append(f"Input path {i}: {error_msg}")

    # Validate output path
    is_valid, error_msg = validator.validate_path(output_path)
    if not is_valid:
        errors.append(f"Output path: {error_msg}")

    return len(errors) == 0, errors

