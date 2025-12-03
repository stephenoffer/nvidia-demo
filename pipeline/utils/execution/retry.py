"""Retry and timeout utilities for resilient infrastructure operations.

Provides retry logic with exponential backoff, timeout handling, and
circuit breaker patterns for network operations and external services.
Consolidates retry functionality from retry_cloud.py.
"""

from __future__ import annotations

import logging
import random
import time
from functools import wraps
from typing import Any, Callable, TypeVar, Optional

# Import random at module level for efficiency
random = random


logger = logging.getLogger(__name__)

T = TypeVar("T")

# Cloud storage exceptions for retry logic
CLOUD_STORAGE_EXCEPTIONS = (
    IOError,
    OSError,
    RuntimeError,
    ConnectionError,
    TimeoutError,
)


class RetryConfig:
    """Configuration for retry behavior."""

    def __init__(
        self,
        max_attempts: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
    ):
        """Initialize retry configuration.

        Args:
            max_attempts: Maximum number of retry attempts
            initial_delay: Initial delay in seconds
            max_delay: Maximum delay in seconds
            exponential_base: Base for exponential backoff
            jitter: Whether to add random jitter to delays
        """
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter


def retry_with_backoff(
    config: Optional[RetryConfig] = None,
    exceptions: tuple[type[Exception], ...] = (Exception,),
    on_retry: Callable[[Exception, int], None] | None = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for retrying functions with exponential backoff.

    Args:
        config: Retry configuration (uses defaults if None)
        exceptions: Tuple of exceptions to catch and retry
        on_retry: Optional callback called on each retry

    Returns:
        Decorated function with retry logic
    """
    if config is None:
        config = RetryConfig()

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception = None

            for attempt in range(1, config.max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if attempt >= config.max_attempts:
                        logger.error(
                            f"Function {func.__name__} failed after {attempt} attempts: {e}"
                        )
                        raise

                    # Calculate delay with exponential backoff
                    delay = min(
                        config.initial_delay * (config.exponential_base ** (attempt - 1)),
                        config.max_delay,
                    )

                    # Add jitter to prevent thundering herd
                    if config.jitter:
                        jitter_amount = delay * 0.1 * random.random()
                        delay += jitter_amount

                    logger.warning(
                        f"Function {func.__name__} failed (attempt {attempt}/{config.max_attempts}): {e}. "
                        f"Retrying in {delay:.2f}s..."
                    )

                    if on_retry:
                        on_retry(e, attempt)

                    time.sleep(delay)

            # Should never reach here, but satisfy type checker
            assert last_exception is not None
            raise last_exception

        return wrapper

    return decorator


def retry_cloud_storage(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    max_delay: float = 60.0,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for retrying cloud storage operations.

    Args:
        max_retries: Maximum number of retries
        initial_delay: Initial delay between retries (seconds)
        backoff_factor: Multiplier for delay between retries
        max_delay: Maximum delay between retries (seconds)

    Returns:
        Decorated function with retry logic
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            delay = initial_delay
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except CLOUD_STORAGE_EXCEPTIONS as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(
                            f"Cloud storage operation failed (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        time.sleep(delay)
                        delay = min(delay * backoff_factor, max_delay)
                    else:
                        logger.error(
                            f"Cloud storage operation failed after {max_retries + 1} attempts: {e}"
                        )

            # All retries exhausted
            raise RetryError(
                f"Cloud storage operation failed after {max_retries + 1} attempts"
            ) from last_exception

        return wrapper

    return decorator


def retry_with_exponential_backoff(
    func: Callable[..., T],
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
) -> T:
    """Retry a function with exponential backoff.

    Args:
        func: Function to retry
        max_retries: Maximum number of retries
        initial_delay: Initial delay (seconds)
        backoff_factor: Backoff multiplier

    Returns:
        Function result

    Raises:
        Exception: If all retries fail
    """
    delay = initial_delay
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return func()
        except CLOUD_STORAGE_EXCEPTIONS as e:
            last_exception = e
            if attempt < max_retries:
                logger.warning(
                    f"Operation failed (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                    f"Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)
                delay *= backoff_factor
            else:
                logger.error(f"Operation failed after {max_retries + 1} attempts: {e}")

    raise RetryError(f"Operation failed after {max_retries + 1} attempts") from last_exception


# Note: with_timeout moved to pipeline.utils.timeout for proper timeout handling
# This function removed to avoid duplication - import from timeout module instead

