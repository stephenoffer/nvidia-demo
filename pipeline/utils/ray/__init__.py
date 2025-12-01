"""Ray utilities module.

Provides Ray initialization, monitoring, and cluster management utilities.
"""

from pipeline.utils.ray.context import (
    RayContext,
    get_ray_context,
    initialize_ray_context,
)
from pipeline.utils.ray.init import (
    get_ray_cluster_info,
    initialize_ray,  # Keep for backward compatibility
    shutdown_ray,  # Keep for backward compatibility
)
from pipeline.utils.ray.monitoring import (
    check_cluster_health,
    get_cluster_metrics,
    log_cluster_status,
    wait_for_resources,
)

__all__ = [
    # Consolidated context (recommended)
    "RayContext",
    "get_ray_context",
    "initialize_ray_context",
    # Legacy initialization (backward compatibility)
    "initialize_ray",
    "shutdown_ray",
    "get_ray_cluster_info",
    # Monitoring
    "get_cluster_metrics",
    "check_cluster_health",
    "log_cluster_status",
    "wait_for_resources",
]

