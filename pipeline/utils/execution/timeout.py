"""Timeout utilities for long-running operations.

Provides timeout decorators and context managers for operations that
may hang or take too long. Critical for production reliability.
"""

from __future__ import annotations

import logging
import signal
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Use built-in TimeoutError (Python 3.3+)
# If we need custom behavior, use pipeline.exceptions.TimeoutError


@contextmanager
def timeout_context(seconds: float):
    """Context manager for timeout handling.

    Uses signal.alarm on Unix systems. Note: This only works on the main thread
    and may interfere with other signal handlers.

    Args:
        seconds: Timeout in seconds

    Yields:
        None

    Raises:
        TimeoutError: If operation exceeds timeout (built-in exception)

    Example:
        ```python
        with timeout_context(5.0):
            long_running_operation()
        ```
    """
    if seconds <= 0:
        yield
        return

    def timeout_handler(signum: int, frame: Any) -> None:
        from pipeline.exceptions import TimeoutError as PipelineTimeoutError
        raise PipelineTimeoutError(f"Operation timed out after {seconds} seconds")

    # Set up signal handler (Unix only)
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(int(seconds))

    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def with_timeout(seconds: float) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to add timeout to function calls.

    Uses threading.Timer for cross-platform support. Note: This doesn't
    actually interrupt the function, it just raises TimeoutError after
    the timeout period.

    Args:
        seconds: Timeout in seconds

    Returns:
        Decorator function

    Example:
        ```python
        @with_timeout(5.0)
        def slow_function():
            time.sleep(10)  # Will raise TimeoutError
        ```
    """
    from functools import wraps

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            result_container: list[Any] = []
            exception_container: list[Exception] = []

            def target() -> None:
                try:
                    result_container.append(func(*args, **kwargs))
                except Exception as e:
                    exception_container.append(e)

            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(timeout=seconds)

            if thread.is_alive():
                logger.error(f"{func.__name__} exceeded timeout of {seconds}s")
                from pipeline.exceptions import TimeoutError as PipelineTimeoutError
                raise PipelineTimeoutError(
                    f"{func.__name__} exceeded timeout of {seconds} seconds"
                )

            if exception_container:
                raise exception_container[0]

            return result_container[0]

        return wrapper

    return decorator



