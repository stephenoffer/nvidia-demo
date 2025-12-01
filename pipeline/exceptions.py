"""Custom exception hierarchy for the pipeline.

Provides structured error handling with proper exception types.
"""

from __future__ import annotations


class PipelineError(Exception):
    """Base exception for all pipeline errors."""

    def __init__(self, message: str, error_code: str | None = None, details: dict | None = None):
        """Initialize pipeline error.

        Args:
            message: Error message
            error_code: Optional error code for programmatic handling
            details: Optional dictionary with additional error details
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}


class ConfigurationError(PipelineError):
    """Raised when configuration is invalid."""

    pass


class DataLoadError(PipelineError):
    """Raised when data loading fails."""

    pass


class DataValidationError(PipelineError):
    """Raised when data validation fails."""

    pass


class ProcessingError(PipelineError):
    """Raised when data processing fails."""

    pass


class GPUError(PipelineError):
    """Raised when GPU operations fail."""

    pass


class RayError(PipelineError):
    """Raised when Ray operations fail."""

    pass


class DeduplicationError(PipelineError):
    """Raised when deduplication fails."""

    pass


class CheckpointError(PipelineError):
    """Raised when checkpoint operations fail."""

    pass


class StorageError(PipelineError):
    """Raised when storage operations fail."""

    pass


class NetworkError(PipelineError):
    """Raised when network operations fail."""

    pass


class ResourceError(PipelineError):
    """Raised when resource allocation fails."""

    pass


class DataSourceError(DataLoadError):
    """Raised when data source operations fail."""

    pass


class ValidationError(DataValidationError):
    """Raised when data validation fails."""

    pass


class TimeoutError(PipelineError):
    """Raised when an operation times out."""

    pass


class RetryError(PipelineError):
    """Raised when retry attempts are exhausted."""

    pass


class MetricsError(PipelineError):
    """Raised when metrics collection or reporting fails."""

    pass

