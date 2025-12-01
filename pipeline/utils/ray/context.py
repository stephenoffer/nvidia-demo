"""Consolidated Ray cluster and Ray Data context configuration.

Provides centralized configuration for both Ray cluster initialization and
Ray Data context settings with best practices including eager_free=True.

This module consolidates all global Ray Data context and Ray cluster context
variables into a single place for easier management and consistency.
"""

from __future__ import annotations

import logging
import os
import platform
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

# Import Ray Data context
try:
    from ray.data import DataContext
    RAY_DATA_AVAILABLE = True
except ImportError:
    RAY_DATA_AVAILABLE = False
    DataContext = None


class RayContext:
    """Consolidated Ray cluster and Ray Data context manager.
    
    Manages both Ray cluster initialization and Ray Data context configuration
    in a single place with best practices applied.
    """
    
    _initialized = False
    _data_context: Optional[Any] = None
    
    @classmethod
    def initialize(
        cls,
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
        # Ray Data context settings
        eager_free: bool = True,
        use_streaming_executor: Optional[bool] = None,
        prefetch_batches: int = 2,
        target_max_block_size: Optional[int] = None,
        target_min_block_size: Optional[int] = None,
        target_shuffle_max_block_size: Optional[int] = None,
        read_op_min_num_blocks: Optional[int] = None,
        preserve_order: bool = False,
        locality_with_output: bool = False,
        execution_cpu: Optional[int] = None,
        execution_gpu: Optional[int] = None,
        execution_object_store_memory: Optional[float] = None,
    ) -> dict[str, Any]:
        """Initialize Ray cluster and configure Ray Data context.
        
        This method consolidates both Ray cluster initialization and Ray Data
        context configuration with best practices applied.
        
        Args:
            address: Ray cluster address (None = local, "auto" = auto-detect)
            num_cpus: Number of CPUs to request
            num_gpus: Number of GPUs to request
            object_store_memory: Object store memory in bytes
            runtime_env: Runtime environment configuration
            namespace: Ray namespace for job isolation
            ignore_reinit_error: Whether to ignore reinit errors
            configure_logging: Whether to configure Ray logging
            include_dashboard: Whether to include dashboard (None = auto-detect)
            dashboard_host: Dashboard host address
            dashboard_port: Dashboard port
            enable_object_spilling: Whether to enable object spilling
            object_spilling_dir: Directory for object spilling
            # Ray Data context settings
            eager_free: Enable eager memory release (best practice, default: True)
            use_streaming_executor: Enable streaming executor (None = auto-detect)
            prefetch_batches: Number of batches to prefetch (default: 2)
            target_max_block_size: Max block size in bytes (default: 128 MiB)
            target_min_block_size: Min block size in bytes (default: 1 MiB)
            target_shuffle_max_block_size: Max block size for shuffle (default: 1 GiB)
            read_op_min_num_blocks: Min number of blocks for read ops (default: 200)
            preserve_order: Enable deterministic execution (slower)
            locality_with_output: Prefer placing tasks on consumer node
            execution_cpu: CPU limit for Ray Data execution
            execution_gpu: GPU limit for Ray Data execution
            execution_object_store_memory: Object store memory limit for Ray Data
        
        Returns:
            Dictionary with initialization results and context info
        """
        result: dict[str, Any] = {
            "initialized": False,
            "address": None,
            "cluster_resources": {},
            "node_resources": {},
            "data_context_configured": False,
        }
        
        # Initialize Ray cluster
        if not ray.is_initialized():
            logger.info("Initializing Ray cluster...")
            
            # Determine Ray address
            if address is None:
                address = os.getenv("RAY_ADDRESS")
                if address:
                    logger.info(f"Using Ray address from RAY_ADDRESS env: {address}")
                elif os.getenv("RAY_CLUSTER_MODE") == "true":
                    address = "auto"
                    logger.info("Auto-detecting Ray cluster address")
            
            # Build initialization kwargs
            init_kwargs: dict[str, Any] = {
                "ignore_reinit_error": ignore_reinit_error,
            }
            
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
                log_level = os.getenv("RAY_LOG_LEVEL", "INFO")
                init_kwargs["logging_level"] = log_level
            
            # Configure dashboard
            if include_dashboard is not None:
                init_kwargs["include_dashboard"] = include_dashboard
            else:
                init_kwargs["include_dashboard"] = address is None or address == "local"
            
            if init_kwargs.get("include_dashboard"):
                init_kwargs["dashboard_host"] = dashboard_host
                init_kwargs["dashboard_port"] = dashboard_port
            
            # Configure runtime environment
            if runtime_env is None:
                runtime_env = {}
            
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
            
            # Configure object spilling
            if enable_object_spilling:
                try:
                    os.makedirs(object_spilling_dir, exist_ok=True)
                    init_kwargs["object_spilling_directory"] = object_spilling_dir
                except OSError as e:
                    logger.warning(f"Could not create object spilling directory {object_spilling_dir}: {e}")
                    enable_object_spilling = False
            
            # Fix object store size for Mac (max 2GB)
            if platform.system() == "Darwin":
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
                        f"Ray cluster initialized successfully. "
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
                            dashboard_url = f"http://{dashboard_host}:{dashboard_port}"
                            logger.info(f"Ray dashboard should be available at: {dashboard_url}")
                    except Exception:
                        pass
            except Exception as e:
                logger.error(f"Failed to initialize Ray cluster: {e}", exc_info=True)
                result["error"] = str(e)
                raise
        else:
            logger.info("Ray cluster already initialized, using existing connection")
            result["initialized"] = True
            try:
                result["address"] = ray.get_runtime_context().get_runtime_context().get("address") or "local"
                result["cluster_resources"] = dict(ray.cluster_resources())
                result["node_resources"] = dict(ray.available_resources())
            except Exception as e:
                logger.warning(f"Error getting Ray cluster info: {e}")
        
        # Configure Ray Data context
        if RAY_DATA_AVAILABLE and DataContext:
            try:
                cls._data_context = DataContext.get_current()
                
                # BEST PRACTICE: Enable eager memory release
                cls._data_context.eager_free = eager_free
                logger.info(f"Ray Data context: eager_free={eager_free} (best practice for memory management)")
                
                # Configure streaming executor
                if use_streaming_executor is not None:
                    cls._data_context.execution_options.use_streaming_executor = use_streaming_executor
                    if use_streaming_executor:
                        cls._data_context.execution_options.prefetch_batches = prefetch_batches
                        logger.info(f"Streaming executor enabled with prefetch_batches={prefetch_batches}")
                
                # Configure block sizes
                if target_max_block_size is not None:
                    cls._data_context.target_max_block_size = target_max_block_size
                    logger.info(f"Set target_max_block_size to {target_max_block_size} bytes")
                
                if target_min_block_size is not None:
                    cls._data_context.target_min_block_size = target_min_block_size
                    logger.info(f"Set target_min_block_size to {target_min_block_size} bytes")
                
                if target_shuffle_max_block_size is not None:
                    cls._data_context.target_shuffle_max_block_size = target_shuffle_max_block_size
                    logger.info(f"Set target_shuffle_max_block_size to {target_shuffle_max_block_size} bytes")
                
                # Configure read options
                if read_op_min_num_blocks is not None:
                    cls._data_context.read_op_min_num_blocks = read_op_min_num_blocks
                    logger.info(f"Set read_op_min_num_blocks to {read_op_min_num_blocks}")
                
                # Configure execution options
                cls._data_context.execution_options.preserve_order = preserve_order
                cls._data_context.execution_options.locality_with_output = locality_with_output
                
                if preserve_order:
                    logger.info("Enabled deterministic execution (preserve_order=True)")
                if locality_with_output:
                    logger.info("Enabled locality with output (for ML ingest)")
                
                # Configure resource limits
                if execution_cpu is not None or execution_gpu is not None or execution_object_store_memory is not None:
                    cls._data_context.execution_options.resource_limits = (
                        cls._data_context.execution_options.resource_limits.copy(
                            cpu=execution_cpu if execution_cpu is not None else cls._data_context.execution_options.resource_limits.cpu,
                            gpu=execution_gpu if execution_gpu is not None else cls._data_context.execution_options.resource_limits.gpu,
                            object_store_memory=execution_object_store_memory
                            if execution_object_store_memory is not None
                            else cls._data_context.execution_options.resource_limits.object_store_memory,
                        )
                    )
                    logger.info(
                        f"Configured Ray Data resource limits: "
                        f"CPU={execution_cpu}, GPU={execution_gpu}, ObjectStore={execution_object_store_memory}"
                    )
                
                result["data_context_configured"] = True
                cls._initialized = True
                
                logger.info("Ray Data context configured successfully with best practices")
            except Exception as e:
                logger.warning(f"Failed to configure Ray Data context: {e}")
                result["data_context_error"] = str(e)
        else:
            logger.warning("Ray Data not available, skipping DataContext configuration")
        
        return result
    
    @classmethod
    def get_data_context(cls) -> Optional[Any]:
        """Get the configured Ray Data context.
        
        Returns:
            DataContext instance or None if not configured
        """
        if cls._data_context is None and RAY_DATA_AVAILABLE and DataContext:
            try:
                cls._data_context = DataContext.get_current()
            except Exception as e:
                logger.warning(f"Failed to get DataContext: {e}")
        return cls._data_context
    
    @classmethod
    def is_initialized(cls) -> bool:
        """Check if Ray context is initialized.
        
        Returns:
            True if both Ray cluster and DataContext are initialized
        """
        return ray.is_initialized() and cls._initialized
    
    @classmethod
    def shutdown(cls, graceful: bool = True, timeout: float = 30.0) -> None:
        """Shutdown Ray cluster gracefully.
        
        Args:
            graceful: Whether to wait for tasks to complete
            timeout: Timeout in seconds for graceful shutdown
        """
        if not ray.is_initialized():
            logger.info("Ray not initialized, skipping shutdown")
            return
        
        try:
            if graceful:
                import time
                start_time = time.time()
                
                while time.time() - start_time < timeout:
                    try:
                        if RAY_STATE_AVAILABLE and ray_state:
                            running_tasks = ray_state.tasks(
                                filters=[("state", "=", "RUNNING")],
                                limit=100,
                            )
                            if not list(running_tasks):
                                break
                        else:
                            break
                        time.sleep(1)
                    except Exception:
                        break
            
            # Cleanup all actors before shutdown
            try:
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
                pass
            
            ray.shutdown()
            cls._initialized = False
            cls._data_context = None
            logger.info("Ray cluster shutdown complete")
        except Exception as e:
            logger.warning(f"Error shutting down Ray: {e}")


def initialize_ray_context(
    config: Optional[dict[str, Any]] = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Convenience function to initialize Ray context from config dict.
    
    Args:
        config: Configuration dictionary (optional)
        **kwargs: Additional configuration parameters
    
    Returns:
        Dictionary with initialization results
    """
    if config:
        kwargs.update(config)
    
    return RayContext.initialize(**kwargs)


def get_ray_context() -> RayContext:
    """Get the global RayContext instance.
    
    Returns:
        RayContext class (singleton pattern)
    """
    return RayContext

