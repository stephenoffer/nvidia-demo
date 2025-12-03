"""Partial failure handling utilities.

Allows pipeline to continue processing available data sources even if
some sources fail. Critical for GR00T: Partial failures should not kill
the entire pipeline.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional


logger = logging.getLogger(__name__)


class PartialFailureHandler:
    """Handles partial failures gracefully.

    Allows pipeline to continue with available data sources even if
    some sources fail.
    """

    def __init__(
        self,
        continue_on_failure: bool = True,
        log_failures: bool = True,
        max_failures: Optional[int] = None,
    ):
        """Initialize partial failure handler.

        Args:
            continue_on_failure: Whether to continue on failures
            log_failures: Whether to log failures
            max_failures: Maximum number of failures before stopping (None = unlimited)
        """
        self.continue_on_failure = continue_on_failure
        self.log_failures = log_failures
        self.max_failures = max_failures
        self.failure_count = 0

    def execute_with_fallback(
        self, func: Callable[[], Any], fallback_value: Any = None, error_context: str = ""
    ) -> Any:
        """Execute function with fallback on failure.

        Args:
            func: Function to execute
            fallback_value: Value to return on failure
            error_context: Context for error logging

        Returns:
            Function result or fallback_value on failure
        """
        try:
            return func()
        except (ValueError, RuntimeError, IOError, OSError, AttributeError, TypeError, KeyError) as e:
            self.failure_count += 1

            if self.log_failures:
                logger.warning(f"{error_context} failed: {e}")

            if self.max_failures and self.failure_count >= self.max_failures:
                logger.error(
                    f"Maximum failures ({self.max_failures}) reached, stopping execution"
                )
                raise

            if self.continue_on_failure:
                return fallback_value
            else:
                raise

    def reset(self) -> None:
        """Reset failure count."""
        self.failure_count = 0


def handle_partial_failures(
    operations: list[tuple[Callable[[], Any], Any, str]],
    continue_on_failure: bool = True,
) -> list[Any]:
    """Execute multiple operations with partial failure handling.

    Args:
        operations: List of (function, fallback_value, error_context) tuples
        continue_on_failure: Whether to continue on failures

    Returns:
        List of results (fallback values used for failed operations)
    """
    handler = PartialFailureHandler(continue_on_failure=continue_on_failure)
    results = []

    for func, fallback_value, error_context in operations:
        result = handler.execute_with_fallback(func, fallback_value, error_context)
        results.append(result)

    return results

