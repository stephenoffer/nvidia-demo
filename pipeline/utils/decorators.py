"""Common decorators for production-ready code.

Provides decorators for logging, timing, retry logic, error handling,
and other common patterns used throughout the pipeline.
"""

from __future__ import annotations

import functools
import logging
import time
from typing import Any, Callable, TypeVar

from pipeline.exceptions import PipelineError

logger = logging.getLogger(__name__)

T = TypeVar("T")


def log_execution_time(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to log function execution time.

    Example:
        ```python
        @log_execution_time
        def slow_function():
            time.sleep(1)
        ```

    Args:
        func: Function to time

    Returns:
        Wrapped function with timing
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            logger.debug(
                f"{func.__name__} completed in {elapsed:.3f}s",
                extra={"function": func.__name__, "duration": elapsed}
            )
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(
                f"{func.__name__} failed after {elapsed:.3f}s: {e}",
                extra={"function": func.__name__, "duration": elapsed, "error": str(e)},
                exc_info=True
            )
            raise

    return wrapper


def handle_errors(
    error_class: type[Exception] = PipelineError,
    log_error: bool = True,
    reraise: bool = True,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to handle errors consistently.

    Args:
        error_class: Exception class to wrap errors in
        log_error: Whether to log errors
        reraise: Whether to re-raise wrapped exception

    Returns:
        Decorator function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            try:
                return func(*args, **kwargs)
            except error_class:
                # Don't wrap if already correct type
                raise
            except Exception as e:
                if log_error:
                    logger.error(
                        f"{func.__name__} failed: {e}",
                        extra={"function": func.__name__, "error": str(e)},
                        exc_info=True
                    )
                if reraise:
                    raise error_class(f"{func.__name__} failed: {e}") from e
                raise

        return wrapper

    return decorator


def validate_not_none(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to validate that function result is not None.

    Args:
        func: Function to validate

    Returns:
        Wrapped function with None check
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        result = func(*args, **kwargs)
        if result is None:
            raise ValueError(f"{func.__name__} returned None")
        return result

    return wrapper


def deprecated(
    reason: str = "",
    version: str = "",
    removal_version: str = "",
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Mark a function as deprecated.

    Args:
        reason: Reason for deprecation
        version: Version when deprecated
        removal_version: Version when will be removed

    Returns:
        Decorator function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            import warnings

            message = f"{func.__name__} is deprecated"
            if version:
                message += f" (since {version})"
            if removal_version:
                message += f" and will be removed in {removal_version}"
            if reason:
                message += f": {reason}"

            warnings.warn(message, DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def rate_limit(calls_per_second: float) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Rate limit function calls.

    Args:
        calls_per_second: Maximum calls per second

    Returns:
        Decorator function
    """
    min_interval = 1.0 / calls_per_second
    last_called = [0.0]

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            elapsed = time.time() - last_called[0]
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
            last_called[0] = time.time()
            return func(*args, **kwargs)

        return wrapper

    return decorator

