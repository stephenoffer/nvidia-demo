"""Ray cluster monitoring and health check utilities.

Provides monitoring capabilities for Ray infrastructure including
resource usage, actor health, task status, and cluster metrics.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

import ray

logger = logging.getLogger(__name__)

# Import ray.util.state if available (Ray 2.0+)
try:
    from ray.util import state as ray_state
    RAY_STATE_AVAILABLE = True
except ImportError:
    RAY_STATE_AVAILABLE = False
    ray_state = None


def get_cluster_metrics() -> dict[str, Any]:
    """Get current Ray cluster metrics.

    Returns:
        Dictionary with cluster metrics including resource usage,
        task counts, actor counts, and object store usage.
    """
    if not ray.is_initialized():
        return {"error": "Ray not initialized"}

    metrics: dict[str, Any] = {
        "timestamp": time.time(),
        "cluster_resources": {},
        "available_resources": {},
        "object_store_stats": {},
        "task_counts": {},
        "actor_counts": {},
    }

    try:
        # Get resource information
        metrics["cluster_resources"] = dict(ray.cluster_resources())
        metrics["available_resources"] = dict(ray.available_resources())

        # Get object store stats
        try:
            # Try to get object store stats via internal API
            if hasattr(ray, "internal") and hasattr(ray.internal, "internal_api"):
                object_store_stats = ray.internal.internal_api.memory_summary()
                metrics["object_store_stats"] = {
                    "used": object_store_stats.get("used", 0),
                    "available": object_store_stats.get("available", 0),
                    "total": object_store_stats.get("total", 0),
                }
        except Exception:
            # Object store stats not available, skip
            pass

        # Get task and actor counts if state API available
        if RAY_STATE_AVAILABLE and ray_state:
            try:
                # Try different Ray state API methods depending on version
                # Ray 2.0+ uses list_tasks() and list_actors()
                if hasattr(ray_state, "list_tasks"):
                    # Ray 2.0+ API
                    running_tasks = list(ray_state.list_tasks(
                        filters=[("state", "=", "RUNNING")],
                        limit=10000,
                    ))
                    metrics["task_counts"]["running"] = len(running_tasks)

                    pending_tasks = list(ray_state.list_tasks(
                        filters=[("state", "=", "PENDING_ARGS_AVAILABLE")],
                        limit=10000,
                    ))
                    metrics["task_counts"]["pending"] = len(pending_tasks)

                    actors = list(ray_state.list_actors(limit=10000))
                    metrics["actor_counts"]["total"] = len(actors)
                    metrics["actor_counts"]["alive"] = sum(
                        1 for a in actors if a.get("state") == "ALIVE"
                    )
                elif hasattr(ray_state, "tasks"):
                    # Older Ray API (if it exists)
                    running_tasks = list(ray_state.tasks(
                        filters=[("state", "=", "RUNNING")],
                        limit=10000,
                    ))
                    metrics["task_counts"]["running"] = len(running_tasks)

                    pending_tasks = list(ray_state.tasks(
                        filters=[("state", "=", "PENDING_ARGS_AVAILABLE")],
                        limit=10000,
                    ))
                    metrics["task_counts"]["pending"] = len(pending_tasks)

                    actors = list(ray_state.actors(limit=10000))
                    metrics["actor_counts"]["total"] = len(actors)
                    metrics["actor_counts"]["alive"] = sum(
                        1 for a in actors if a.get("state") == "ALIVE"
                    )
                else:
                    # State API available but methods not found - skip task/actor counts
                    logger.debug("Ray state API available but tasks/actors methods not found")
            except Exception as e:
                logger.warning(f"Error getting task/actor counts: {e}")

    except Exception as e:
        logger.warning(f"Error getting cluster metrics: {e}")
        metrics["error"] = str(e)

    return metrics


def check_cluster_health() -> dict[str, Any]:
    """Check Ray cluster health.

    Returns:
        Dictionary with health status and any issues found.
    """
    if not ray.is_initialized():
        return {
            "healthy": False,
            "issues": ["Ray not initialized"],
        }

    health: dict[str, Any] = {
        "healthy": True,
        "issues": [],
        "warnings": [],
    }

    try:
        # Check if cluster has resources
        cluster_resources = ray.cluster_resources()
        if not cluster_resources:
            health["healthy"] = False
            health["issues"].append("No cluster resources available")

        # Check CPU availability
        available_resources = ray.available_resources()
        cpu_available = available_resources.get("CPU", 0)
        if cpu_available < 0.1:
            health["warnings"].append(f"Low CPU availability: {cpu_available}")

        # Check GPU availability if GPUs requested
        gpu_available = available_resources.get("GPU", 0)
        if gpu_available < 0.1:
            health["warnings"].append(f"Low GPU availability: {gpu_available}")

        # Check object store memory
        try:
            # Try to get object store stats via internal API
            if hasattr(ray, "internal") and hasattr(ray.internal, "internal_api"):
                object_store_stats = ray.internal.internal_api.memory_summary()
                used_memory = object_store_stats.get("used", 0)
                total_memory = object_store_stats.get("total", 0)
                if total_memory > 0:
                    usage_percent = (used_memory / total_memory) * 100
                    if usage_percent > 90:
                        health["warnings"].append(
                            f"High object store usage: {usage_percent:.1f}%"
                        )
        except Exception:
            # Object store stats not available, skip
            pass

        # Check for dead actors if state API available
        if RAY_STATE_AVAILABLE and ray_state:
            try:
                actors = list(ray_state.actors(limit=1000))
                dead_actors = [
                    a for a in actors
                    if a.get("state") not in ("ALIVE", "PENDING_CREATION")
                ]
                if dead_actors:
                    health["warnings"].append(
                        f"Found {len(dead_actors)} dead actors"
                    )
            except Exception:
                pass

    except Exception as e:
        health["healthy"] = False
        health["issues"].append(f"Error checking cluster health: {e}")

    return health


def wait_for_resources(
    resources: dict[str, float],
    timeout: float = 300.0,
    check_interval: float = 1.0,
) -> bool:
    """Wait for specified resources to become available.

    Args:
        resources: Dictionary of resource requirements (e.g., {"CPU": 4.0, "GPU": 1.0})
        timeout: Maximum time to wait in seconds
        check_interval: Interval between checks in seconds

    Returns:
        True if resources available, False if timeout
    """
    if not ray.is_initialized():
        return False

    start_time = time.time()
    while time.time() - start_time < timeout:
        available = ray.available_resources()
        all_available = all(
            available.get(resource, 0) >= amount
            for resource, amount in resources.items()
        )
        if all_available:
            return True
        time.sleep(check_interval)

    return False


def log_cluster_status() -> None:
    """Log current Ray cluster status for debugging."""
    if not ray.is_initialized():
        logger.warning("Ray not initialized")
        return

    try:
        metrics = get_cluster_metrics()
        logger.info(f"Ray cluster metrics: {metrics}")

        health = check_cluster_health()
        if not health["healthy"]:
            logger.error(f"Ray cluster health issues: {health['issues']}")
        if health["warnings"]:
            logger.warning(f"Ray cluster warnings: {health['warnings']}")
    except Exception as e:
        logger.warning(f"Error logging cluster status: {e}")

