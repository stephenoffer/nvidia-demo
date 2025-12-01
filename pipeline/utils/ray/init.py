"""Ray initialization utilities for proper infrastructure setup.

Provides centralized Ray initialization with proper configuration for
production deployments including cluster connections, runtime environments,
resource management, and monitoring.
"""

from __future__ import annotations

import logging
import os
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


def initialize_ray(
    address: Optional[str] = None,
    num_cpus: Optional[int] = None,
    num_gpus: Optional[int] = None,
    object_store_memory: Optional[int] = None,
    runtime_env: Optional[dict[str, Any]] = None,
    namespace: Optional[str] = None,
    ignore_reinit_error: bool = True,
    configure_logging: bool = True,
    include_dashboard: Optional[bool] = None,
    dashboard_host: str = "0.0.0.0",
    dashboard_port: int = 8265,
    enable_object_spilling: bool = True,
    object_spilling_dir: str = "/tmp/ray_spill",
    _system_config: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Initialize Ray with proper infrastructure configuration.

    Handles both local and cluster initialization with proper resource
    management, runtime environments, and monitoring setup.

    Args:
        address: Ray cluster address (None = local, "auto" = auto-detect)
        num_cpus: Number of CPUs to request
        num_gpus: Number of GPUs to request
        object_store_memory: Object store memory in bytes
        runtime_env: Runtime environment configuration (pip packages, env vars, etc.)
        namespace: Ray namespace for job isolation
        ignore_reinit_error: Whether to ignore reinit errors
        configure_logging: Whether to configure Ray logging
        include_dashboard: Whether to include dashboard (None = auto-detect)
        dashboard_host: Dashboard host address
        dashboard_port: Dashboard port
        enable_object_spilling: Whether to enable object spilling
        object_spilling_dir: Directory for object spilling
        _system_config: System configuration (deprecated, use runtime_env)

    Returns:
        Dictionary with initialization results and cluster info
    """
    result: dict[str, Any] = {
        "initialized": False,
        "address": None,
        "cluster_resources": {},
        "node_resources": {},
    }

    # Check if already initialized
    if ray.is_initialized():
        logger.info("Ray already initialized, skipping reinitialization")
        try:
            result["initialized"] = True
            result["address"] = ray.get_runtime_context().get_runtime_context().get("address")
            result["cluster_resources"] = ray.cluster_resources()
            result["node_resources"] = ray.available_resources()
        except Exception as e:
            logger.warning(f"Error getting Ray cluster info: {e}")
        return result

    # Determine Ray address
    if address is None:
        # Check environment variable
        address = os.getenv("RAY_ADDRESS")
        if address:
            logger.info(f"Using Ray address from RAY_ADDRESS env: {address}")
        elif os.getenv("RAY_CLUSTER_MODE") == "true":
            # Auto-detect cluster address
            address = "auto"
            logger.info("Auto-detecting Ray cluster address")

    # Build initialization kwargs
    init_kwargs: dict[str, Any] = {
        "ignore_reinit_error": ignore_reinit_error,
    }

    # Add address if specified
    if address:
        init_kwargs["address"] = address

    # Add resource specifications
    if num_cpus is not None:
        init_kwargs["num_cpus"] = num_cpus
    if num_gpus is not None:
        init_kwargs["num_gpus"] = num_gpus
    if object_store_memory is not None:
        init_kwargs["object_store_memory"] = object_store_memory

    # Configure logging
    if configure_logging:
        init_kwargs["configure_logging"] = True
        # Set log level from environment if available
        log_level = os.getenv("RAY_LOG_LEVEL", "INFO")
        init_kwargs["logging_level"] = log_level

    # Configure dashboard
    if include_dashboard is not None:
        init_kwargs["include_dashboard"] = include_dashboard
    else:
        # Auto-detect: enable dashboard for local, disable for cluster
        init_kwargs["include_dashboard"] = address is None or address == "local"

    if init_kwargs.get("include_dashboard"):
        init_kwargs["dashboard_host"] = dashboard_host
        init_kwargs["dashboard_port"] = dashboard_port

    # Configure runtime environment
    if runtime_env is None:
        runtime_env = {}
    
    # Add environment variables to runtime env
    if "env_vars" not in runtime_env:
        runtime_env["env_vars"] = {}
    
    # Set important environment variables
    important_env_vars = [
        "CUDA_VISIBLE_DEVICES",
        "NCCL_DEBUG",
        "NCCL_TIMEOUT",
        "NCCL_ASYNC_ERROR_HANDLING",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
    ]
    for var in important_env_vars:
        if var in os.environ and var not in runtime_env["env_vars"]:
            runtime_env["env_vars"][var] = os.environ[var]

    if runtime_env:
        init_kwargs["runtime_env"] = runtime_env

    # Add namespace for job isolation
    if namespace:
        init_kwargs["namespace"] = namespace
    elif os.getenv("RAY_NAMESPACE"):
        init_kwargs["namespace"] = os.getenv("RAY_NAMESPACE")

    # Configure object spilling using the stable API
    # Use object_spilling_directory parameter instead of _system_config
    if enable_object_spilling:
        # Ensure spilling directory exists
        try:
            os.makedirs(object_spilling_dir, exist_ok=True)
            # Use the stable API: object_spilling_directory parameter
            init_kwargs["object_spilling_directory"] = object_spilling_dir
        except OSError as e:
            logger.warning(f"Could not create object spilling directory {object_spilling_dir}: {e}")
            # Disable spilling if directory cannot be created
            enable_object_spilling = False

    # Fix object store size for Mac (max 2GB) - override any existing setting
    import platform
    if platform.system() == "Darwin":
        # Mac has a known issue with object store > 2GB
        # Always override to 2GB for Mac, regardless of what was set before
        init_kwargs["object_store_memory"] = 2 * 1024 * 1024 * 1024  # 2GB
        logger.info("Mac detected: Setting object_store_memory to 2GB (Mac limit)")
    
    # Initialize Ray
    try:
        ray.init(**init_kwargs)
        result["initialized"] = True
        result["address"] = address or "local"
        
        # Get cluster information
        try:
            result["cluster_resources"] = dict(ray.cluster_resources())
            result["node_resources"] = dict(ray.available_resources())
            logger.info(
                f"Ray initialized successfully. "
                f"Cluster resources: {result['cluster_resources']}"
            )
        except Exception as e:
            logger.warning(f"Error getting cluster resources: {e}")

        # Log dashboard URL if available
        if init_kwargs.get("include_dashboard"):
            try:
                ctx = ray.get_runtime_context()
                dashboard_url = getattr(ctx, "dashboard_url", None)
                if dashboard_url:
                    logger.info(f"Ray dashboard available at: {dashboard_url}")
                else:
                    # Construct dashboard URL manually
                    dashboard_url = f"http://{dashboard_host}:{dashboard_port}"
                    logger.info(f"Ray dashboard should be available at: {dashboard_url}")
            except Exception:
                # Dashboard URL not available, but that's okay
                pass
        
        # Validate cluster health after initialization
        try:
            from pipeline.utils.ray.monitoring import check_cluster_health
            health = check_cluster_health()
            if not health["healthy"]:
                logger.warning(f"Ray cluster health issues detected: {health['issues']}")
            if health["warnings"]:
                for warning in health["warnings"]:
                    logger.warning(f"Ray cluster warning: {warning}")
        except Exception:
            pass  # Don't fail on health check errors

    except Exception as e:
        logger.error(f"Failed to initialize Ray: {e}", exc_info=True)
        result["error"] = str(e)
        raise

    return result


def get_ray_cluster_info() -> dict[str, Any]:
    """Get information about the current Ray cluster.

    Returns:
        Dictionary with cluster information
    """
    if not ray.is_initialized():
        return {"initialized": False}

    info: dict[str, Any] = {
        "initialized": True,
        "cluster_resources": {},
        "available_resources": {},
        "nodes": [],
    }

    try:
        info["cluster_resources"] = dict(ray.cluster_resources())
        info["available_resources"] = dict(ray.available_resources())
        
        # Get node information
        try:
            nodes = ray.nodes()
            info["nodes"] = [
                {
                    "node_id": node.get("NodeID"),
                    "alive": node.get("Alive"),
                    "resources": node.get("Resources", {}),
                }
                for node in nodes
            ]
        except Exception as e:
            logger.warning(f"Error getting node information: {e}")
            info["nodes"] = []
        
        # Get runtime context
        try:
            ctx = ray.get_runtime_context()
            info["runtime_context"] = {
                "namespace": getattr(ctx, "namespace", None),
                "job_id": str(getattr(ctx, "job_id", None)) if hasattr(ctx, "job_id") else None,
                "node_id": str(getattr(ctx, "node_id", None)) if hasattr(ctx, "node_id") else None,
            }
        except Exception as e:
            logger.warning(f"Error getting runtime context: {e}")

    except Exception as e:
        logger.warning(f"Error getting cluster info: {e}")
        info["error"] = str(e)

    return info


def shutdown_ray(graceful: bool = True, timeout: float = 30.0) -> None:
    """Shutdown Ray gracefully.

    Args:
        graceful: Whether to wait for tasks to complete
        timeout: Timeout in seconds for graceful shutdown
    """
    if not ray.is_initialized():
        logger.info("Ray not initialized, skipping shutdown")
        return

    try:
        if graceful:
            # Wait for running tasks to complete (with timeout)
            import time
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                # Check if there are running tasks
                try:
                    # Try to use state API if available
                    if RAY_STATE_AVAILABLE and ray_state:
                        running_tasks = ray_state.tasks(
                            filters=[("state", "=", "RUNNING")],
                            limit=100,
                        )
                        if not list(running_tasks):
                            break
                    else:
                        # State API not available, proceed with shutdown
                        break
                    time.sleep(1)
                except Exception:
                    # If state API not available or error, proceed with shutdown
                    break

        # Cleanup all actors before shutdown
        try:
            # Get all actors and kill them
            if RAY_STATE_AVAILABLE and ray_state:
                actors = ray_state.actors(limit=1000)
                for actor_info in actors:
                    try:
                        actor_id = actor_info.get("actor_id")
                        if actor_id:
                            actor_handle = ray.get_actor(actor_id)
                            ray.kill(actor_handle, no_restart=True)
                    except Exception:
                        pass
        except Exception:
            # If state API not available, proceed with shutdown
            pass

        ray.shutdown()
        logger.info("Ray shutdown complete")
    except Exception as e:
        logger.warning(f"Error shutting down Ray: {e}")

