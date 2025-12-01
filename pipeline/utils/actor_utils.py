"""Utilities for Ray actor management and cleanup.

Provides common patterns for actor creation, usage, and cleanup.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, TypeVar

import ray

logger = logging.getLogger(__name__)

T = TypeVar("T")


def with_actor_cleanup(
    actor_class: type,
    actor_args: tuple[Any, ...] | None = None,
    actor_kwargs: dict[str, Any] | None = None,
) -> Callable[[Callable[[Any, list[dict[str, Any]]], list[dict[str, Any]]]], Callable[[list[dict[str, Any]]], list[dict[str, Any]]]]:
    """Create a batch processor with automatic actor cleanup.

    Args:
        actor_class: Ray actor class to create
        actor_args: Positional arguments for actor constructor
        actor_kwargs: Keyword arguments for actor constructor

    Returns:
        Decorator function that creates batch processor with cleanup
    """
    def decorator(
        process_func: Callable[[Any, list[dict[str, Any]]], list[dict[str, Any]]]
    ) -> Callable[[list[dict[str, Any]]], list[dict[str, Any]]]:
        """Create batch processor with actor cleanup."""
        def process_batch(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
            """Process batch with actor cleanup."""
            actor = actor_class.remote(
                *(actor_args or ()),
                **(actor_kwargs or {})
            )

            try:
                return process_func(actor, batch)
            finally:
                try:
                    ray.kill(actor)
                except (ValueError, ray.exceptions.RayActorError):
                    pass

        return process_batch

    return decorator


def cleanup_actor(actor: Any, wait: bool = True) -> None:
    """Cleanup Ray actor safely.

    Args:
        actor: Ray actor handle
        wait: Whether to wait for actor to terminate
    """
    try:
        ray.kill(actor, no_restart=True)
        if wait:
            # Wait for actor to terminate (with timeout)
            import time
            from pipeline.utils.constants import _DEFAULT_RAY_ACTOR_TIMEOUT
            
            start_time = time.time()
            timeout = _DEFAULT_RAY_ACTOR_TIMEOUT
            while time.time() - start_time < timeout:
                try:
                    # Check if actor is still alive
                    if hasattr(actor, "_actor_id"):
                        ray.get_actor(actor._actor_id.hex())
                    time.sleep(0.1)
                except (ValueError, ray.exceptions.RayActorError):
                    # Actor terminated
                    break
            else:
                logger.warning(f"Actor {actor} did not terminate gracefully within {timeout}s")
    except (ValueError, ray.exceptions.RayActorError, AttributeError):
        pass


def create_actor_pool(
    actor_class: type,
    num_actors: int,
    actor_args: tuple[Any, ...] | None = None,
    actor_kwargs: dict[str, Any] | None = None,
) -> list[Any]:
    """Create a pool of Ray actors for reuse.

    Args:
        actor_class: Ray actor class to create
        num_actors: Number of actors in pool
        actor_args: Positional arguments for actor constructor
        actor_kwargs: Keyword arguments for actor constructor

    Returns:
        List of actor handles
    """
    actors = []
    for _ in range(num_actors):
        actor = actor_class.remote(
            *(actor_args or ()),
            **(actor_kwargs or {})
        )
        actors.append(actor)
    logger.info(f"Created actor pool with {num_actors} actors")
    return actors


def cleanup_actor_pool(actors: list[Any]) -> None:
    """Cleanup a pool of Ray actors.

    Args:
        actors: List of actor handles
    """
    if not actors:
        return
    
    # Kill all actors first (non-blocking)
    for actor in actors:
        try:
            ray.kill(actor, no_restart=True)
        except (ValueError, ray.exceptions.RayActorError, AttributeError):
            pass
    
    # Wait for all actors to terminate with timeout
    import time
    from pipeline.utils.constants import _DEFAULT_RAY_ACTOR_TIMEOUT
    
    start_time = time.time()
    timeout = _DEFAULT_RAY_ACTOR_TIMEOUT
    remaining_actors = list(actors)
    
    while remaining_actors and (time.time() - start_time < timeout):
        still_alive = []
        for actor in remaining_actors:
            try:
                if hasattr(actor, "_actor_id"):
                    ray.get_actor(actor._actor_id.hex())
                    still_alive.append(actor)
                else:
                    # Can't check, assume dead
                    pass
            except (ValueError, ray.exceptions.RayActorError):
                # Actor terminated
                pass
        remaining_actors = still_alive
        if remaining_actors:
            time.sleep(0.1)
    
    if remaining_actors:
        logger.warning(f"{len(remaining_actors)} actors did not terminate gracefully within {timeout}s")
    
    logger.info(f"Cleaned up {len(actors)} actors")

