"""Pipeline lifecycle management: initialization and shutdown.

Handles Ray initialization, RAPIDS setup, GPU validation, and resource cleanup.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import ray

from pipeline.config import PipelineConfig
from pipeline.exceptions import RayError
from pipeline.utils.constants import _DEFAULT_CPUS, _DEFAULT_CPUS_PER_GPU, _DEFAULT_OBJECT_STORE_GB

logger = logging.getLogger(__name__)


class PipelineLifecycleManager:
    """Manages pipeline initialization and shutdown lifecycle."""

    def __init__(self, config: PipelineConfig) -> None:
        """Initialize lifecycle manager.

        Args:
            config: Pipeline configuration
        """
        self.config = config

    def initialize(self) -> None:
        """Initialize Ray, RAPIDS, GPU resources, and health server."""
        self._validate_inputs()
        self._initialize_ray()
        self._initialize_rapids()
        self._log_gpu_status()
        self._start_health_server()

    def _validate_inputs(self) -> None:
        """Validate and sanitize input paths."""
        try:
            from pipeline.utils.input_validation import validate_inputs

            is_valid, errors = validate_inputs(self.config.input_paths, self.config.output_path)
            if not is_valid:
                error_msg = "Input validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
                logger.error(error_msg)
                raise ValueError(error_msg)
        except ImportError:
            logger.warning("Input validation module not available, skipping validation")

    def _initialize_ray(self) -> None:
        """Initialize Ray with proper configuration using consolidated context."""
        from pipeline.utils.ray.context import RayContext

        if not ray.is_initialized():
            default_object_store = self._calculate_object_store_memory()

            self._validate_gpu_availability()

            runtime_env = self._build_runtime_env()

            # Get Ray Data config from pipeline config
            ray_data_cfg = self.config.ray_data_config

            init_result = RayContext.initialize(
                address=None,
                num_cpus=self.config.num_cpus
                or (self.config.num_gpus * _DEFAULT_CPUS_PER_GPU if self.config.num_gpus > 0 else _DEFAULT_CPUS),
                num_gpus=self.config.num_gpus,
                object_store_memory=default_object_store,
                runtime_env=runtime_env if runtime_env else None,
                namespace=os.getenv("RAY_NAMESPACE"),
                ignore_reinit_error=True,
                configure_logging=True,
                include_dashboard=None,
                enable_object_spilling=self.config.streaming,
                object_spilling_dir="/tmp/ray_spill",
                # Ray Data context settings with best practices
                eager_free=True,  # Best practice: enable eager memory release
                use_streaming_executor=self.config.streaming,
                prefetch_batches=2,
                target_max_block_size=ray_data_cfg.get("max_block_size"),
                target_min_block_size=ray_data_cfg.get("min_block_size"),
                target_shuffle_max_block_size=ray_data_cfg.get("shuffle_max_block_size"),
                read_op_min_num_blocks=ray_data_cfg.get("read_op_min_num_blocks"),
                preserve_order=ray_data_cfg.get("preserve_order", False),
                locality_with_output=ray_data_cfg.get("locality_with_output", False),
                execution_cpu=ray_data_cfg.get("execution_cpu"),
                execution_gpu=ray_data_cfg.get("execution_gpu"),
                execution_object_store_memory=ray_data_cfg.get("execution_object_store_memory"),
            )

            if not init_result.get("initialized"):
                raise RayError("Failed to initialize Ray")

            logger.info(f"Ray initialized: {init_result.get('address', 'local')}")
            if init_result.get("data_context_configured"):
                logger.info("Ray Data context configured with best practices")

            self._log_cluster_status()
        else:
            logger.info("Ray already initialized, using existing connection")
            # Still configure DataContext if Ray is already initialized
            try:
                from pipeline.utils.ray.context import RayContext
                ray_data_cfg = self.config.ray_data_config
                RayContext.initialize(
                    eager_free=True,
                    use_streaming_executor=self.config.streaming,
                    prefetch_batches=2,
                    target_max_block_size=ray_data_cfg.get("max_block_size"),
                    target_min_block_size=ray_data_cfg.get("min_block_size"),
                    preserve_order=ray_data_cfg.get("preserve_order", False),
                    locality_with_output=ray_data_cfg.get("locality_with_output", False),
                )
            except Exception as e:
                logger.warning(f"Failed to configure Ray Data context: {e}")
            self._log_cluster_status()

    def _calculate_object_store_memory(self) -> int:
        """Calculate object store memory size."""
        try:
            import psutil

            available_memory = psutil.virtual_memory().available
            return min(available_memory // 2, _DEFAULT_OBJECT_STORE_GB)
        except ImportError:
            logger.warning("psutil not available, using default object store size")
            return _DEFAULT_OBJECT_STORE_GB

    def _start_health_server(self) -> None:
        """Start health check HTTP server for Kubernetes probes."""
        try:
            import os
            enable_health_server = os.getenv("ENABLE_HEALTH_SERVER", "true").lower() == "true"
            if enable_health_server:
                from pipeline.server import start_health_server
                
                port = int(os.getenv("HEALTH_CHECK_PORT", "8080"))
                host = os.getenv("HEALTH_CHECK_HOST", "0.0.0.0")
                
                self.health_server = start_health_server(port=port, host=host)
                logger.info(f"Health check server started on {host}:{port}")
            else:
                logger.info("Health check server disabled via ENABLE_HEALTH_SERVER=false")
                self.health_server = None
        except Exception as e:
            logger.warning(f"Failed to start health server: {e}")
            self.health_server = None

    def _validate_gpu_availability(self) -> None:
        """Validate GPU availability and adjust config."""
        if self.config.num_gpus > 0:
            try:
                import torch

                if not torch.cuda.is_available():
                    logger.warning(
                        f"Requested {self.config.num_gpus} GPUs but CUDA not available. "
                        "Falling back to CPU-only mode."
                    )
                    self.config.num_gpus = 0
                else:
                    num_available_gpus = torch.cuda.device_count()
                    if self.config.num_gpus > num_available_gpus:
                        logger.warning(
                            f"Requested {self.config.num_gpus} GPUs but only {num_available_gpus} available. "
                            f"Using {num_available_gpus} GPUs."
                        )
                    self.config.num_gpus = min(self.config.num_gpus, num_available_gpus)
            except ImportError:
                logger.warning("PyTorch not available, cannot validate GPU availability")
                if self.config.num_gpus > 0:
                    logger.warning("Falling back to CPU-only mode")
                    self.config.num_gpus = 0

    def _build_runtime_env(self) -> dict[str, Any]:
        """Build Ray runtime environment for GPU operations."""
        runtime_env = {}
        if self.config.num_gpus > 0:
            runtime_env["env_vars"] = {
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                "NCCL_DEBUG": os.environ.get("NCCL_DEBUG", "WARN"),
                "NCCL_TIMEOUT": os.environ.get("NCCL_TIMEOUT", "1800"),
                "NCCL_ASYNC_ERROR_HANDLING": os.environ.get("NCCL_ASYNC_ERROR_HANDLING", "1"),
            }
        return runtime_env

    def _log_cluster_status(self) -> None:
        """Log Ray cluster status."""
        try:
            from pipeline.utils.ray.monitoring import log_cluster_status

            log_cluster_status()
        except Exception:
            pass

    def _initialize_rapids(self) -> None:
        """Initialize RAPIDS environment for GPU operations."""
        if self.config.num_gpus > 0:
            try:
                from pipeline.utils.gpu.rapids import initialize_rapids_environment

                rapids_results = initialize_rapids_environment()
                if rapids_results.get("warnings"):
                    for warning in rapids_results["warnings"]:
                        logger.warning(f"RAPIDS warning: {warning}")
            except (ImportError, RuntimeError, ValueError) as e:
                logger.warning(f"Failed to initialize RAPIDS environment: {e}")

    def _log_gpu_status(self) -> None:
        """Log GPU status."""
        if self.config.num_gpus > 0:
            try:
                from pipeline.utils.gpu.monitoring import log_gpu_status

                log_gpu_status()
            except (ImportError, RuntimeError) as e:
                logger.warning(f"Failed to log GPU status: {e}")

    def shutdown(self) -> None:
        """Shutdown Ray and cleanup all resources."""
        logger.info("Shutting down pipeline and cleaning up resources")

        self._cleanup_health_server()
        self._cleanup_loaders()
        self._cleanup_resources()
        self._cleanup_gpu()
        self._cleanup_actors()
        self._shutdown_ray()

    def _cleanup_health_server(self) -> None:
        """Stop health check server."""
        if hasattr(self, 'health_server') and self.health_server:
            try:
                self.health_server.stop()
                logger.info("Health check server stopped")
            except Exception as e:
                logger.warning(f"Error stopping health server: {e}")

    def _cleanup_loaders(self) -> None:
        """Cleanup loader instances."""
        # Loaders are managed by orchestrator, but ensure cleanup here
        pass

    def _cleanup_resources(self) -> None:
        """Cleanup system resources."""
        try:
            from pipeline.utils.resource_manager import get_resource_manager

            resource_mgr = get_resource_manager()
            resource_mgr.cleanup_all()
        except (ImportError, AttributeError, RuntimeError) as e:
            logger.warning(f"Error in resource cleanup: {e}")

    def _cleanup_gpu(self) -> None:
        """Cleanup GPU memory."""
        try:
            from pipeline.utils.gpu.memory import clear_gpu_cache

            clear_gpu_cache()
            logger.info("GPU memory caches cleared")
        except (ImportError, RuntimeError) as e:
            logger.warning(f"Error clearing GPU cache: {e}")

    def _cleanup_actors(self) -> None:
        """Cleanup Ray actors."""
        try:
            if ray.is_initialized():
                try:
                    from ray.util import state as ray_state

                    actors = ray_state.actors(limit=1000)
                    for actor_info in actors:
                        try:
                            actor_id = actor_info.get("actor_id")
                            if actor_id:
                                actor_handle = ray.get_actor(actor_id)
                                ray.kill(actor_handle, no_restart=True)
                        except Exception:
                            pass
                except (ImportError, AttributeError, Exception):
                    pass
        except Exception as e:
            logger.warning(f"Error cleaning up actors: {e}")

    def _shutdown_ray(self) -> None:
        """Shutdown Ray cluster."""
        from pipeline.utils.ray.context import RayContext
        from pipeline.utils.constants import _DEFAULT_RAY_SHUTDOWN_TIMEOUT

        RayContext.shutdown(graceful=True, timeout=_DEFAULT_RAY_SHUTDOWN_TIMEOUT)

