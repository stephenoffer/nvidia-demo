"""Ray Data performance configuration and optimization.

Provides utilities for configuring Ray Data execution options,
resource limits, and performance tuning based on best practices.
See: https://docs.ray.io/en/latest/data/performance-tips.html
"""

import logging
from typing import Any, Dict, Optional

from ray.data import (
    DataContext,  # https://docs.ray.io/en/latest/data/api/doc/ray.data.DataContext.html
)

logger = logging.getLogger(__name__)


class RayDataConfig:
    """Configuration helper for Ray Data performance optimization.

    Provides methods to configure Ray Data execution options,
    resource limits, and performance tuning parameters.
    """

    @staticmethod
    def configure_execution(
        cpu: Optional[int] = None,
        gpu: Optional[int] = None,
        object_store_memory: Optional[float] = None,
        preserve_order: bool = False,
        locality_with_output: bool = False,
    ) -> None:
        """Configure Ray Data execution options.

        Args:
            cpu: CPU limit for Ray Data execution (None = use cluster size)
            gpu: GPU limit for Ray Data execution (None = use cluster size)
            object_store_memory: Object store memory limit in bytes (None = 1/4 of total)
            preserve_order: Enable deterministic execution (may decrease performance)
            locality_with_output: Prefer placing tasks on consumer node (for ML ingest)
        """
        ctx = DataContext.get_current()

        # Configure resource limits
        if cpu is not None or gpu is not None or object_store_memory is not None:
            ctx.execution_options.resource_limits = (
                ctx.execution_options.resource_limits.copy(
                    cpu=cpu if cpu is not None else ctx.execution_options.resource_limits.cpu,
                    gpu=gpu if gpu is not None else ctx.execution_options.resource_limits.gpu,
                    object_store_memory=object_store_memory
                    if object_store_memory is not None
                    else ctx.execution_options.resource_limits.object_store_memory,
                )
            )
            logger.info(
                f"Configured Ray Data resource limits: "
                f"CPU={cpu}, GPU={gpu}, ObjectStore={object_store_memory}"
            )

        # Configure execution options
        ctx.execution_options.preserve_order = preserve_order
        ctx.execution_options.locality_with_output = locality_with_output

        if preserve_order:
            logger.info("Enabled deterministic execution (preserve_order=True)")
        if locality_with_output:
            logger.info("Enabled locality with output (for ML ingest)")

    @staticmethod
    def configure_block_sizes(
        min_block_size: Optional[int] = None,
        max_block_size: Optional[int] = None,
        shuffle_max_block_size: Optional[int] = None,
    ) -> None:
        """Configure Ray Data block size parameters.

        Args:
            min_block_size: Minimum block size in bytes (default: 1 MiB)
            max_block_size: Maximum block size in bytes (default: 128 MiB)
            shuffle_max_block_size: Max block size for shuffle ops (default: 1 GiB)
        """
        ctx = DataContext.get_current()

        if min_block_size is not None:
            ctx.target_min_block_size = min_block_size
            logger.info(f"Set target_min_block_size to {min_block_size} bytes")

        if max_block_size is not None:
            ctx.target_max_block_size = max_block_size
            logger.info(f"Set target_max_block_size to {max_block_size} bytes")

        if shuffle_max_block_size is not None:
            ctx.target_shuffle_max_block_size = shuffle_max_block_size
            logger.info(f"Set target_shuffle_max_block_size to {shuffle_max_block_size} bytes")

    @staticmethod
    def configure_read_options(
        min_num_blocks: Optional[int] = None,
    ) -> None:
        """Configure Ray Data read operation defaults.

        Args:
            min_num_blocks: Minimum number of output blocks for read ops (default: 200)
        """
        ctx = DataContext.get_current()

        if min_num_blocks is not None:
            ctx.read_op_min_num_blocks = min_num_blocks
            logger.info(f"Set read_op_min_num_blocks to {min_num_blocks}")

    @staticmethod
    def get_read_kwargs(
        override_num_blocks: Optional[int] = None,
        num_cpus: Optional[float] = None,
        num_gpus: Optional[float] = None,
        memory: Optional[float] = None,
        concurrency: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Get optimized kwargs for Ray Data read operations.

        Provides sensible defaults for resource allocation and parallelism.

        Args:
            override_num_blocks: Override number of output blocks
            num_cpus: CPUs per read task (default: 1, use 0.25 for IO-bound)
            num_gpus: GPUs per read task
            memory: Heap memory per read task in bytes
            concurrency: Max concurrent Ray tasks (default: auto-detect based on cluster)
            **kwargs: Additional read operation kwargs

        Returns:
            Dictionary of kwargs for read operations
        """
        read_kwargs: Dict[str, Any] = {}

        if override_num_blocks is not None:
            read_kwargs["override_num_blocks"] = override_num_blocks

        # Build ray_remote_args for resource allocation
        ray_remote_args: Dict[str, Any] = {}
        if num_cpus is not None:
            ray_remote_args["num_cpus"] = num_cpus
        if num_gpus is not None:
            ray_remote_args["num_gpus"] = num_gpus
        if memory is not None:
            ray_remote_args["memory"] = memory

        if ray_remote_args:
            read_kwargs["ray_remote_args"] = ray_remote_args

        # Set concurrency limit to prevent overwhelming cluster
        # Default to reasonable limit if not specified
        if concurrency is not None:
            read_kwargs["concurrency"] = concurrency
        elif num_gpus is None and num_cpus is None:
            # Auto-detect reasonable concurrency for IO-bound operations
            try:
                import ray
                cluster_resources = ray.cluster_resources()
                available_cpus = cluster_resources.get("CPU", 1.0)
                # Use 2x CPUs for IO-bound reads (can overlap I/O)
                read_kwargs["concurrency"] = max(1, int(available_cpus * 2))
            except (AttributeError, RuntimeError):
                # Fallback if cluster resources not available
                read_kwargs["concurrency"] = 10

        read_kwargs.update(kwargs)
        return read_kwargs

    @staticmethod
    def get_map_batches_kwargs(
        batch_size: Optional[int] = None,
        num_cpus: Optional[float] = None,
        num_gpus: Optional[float] = None,
        batch_format: Optional[str] = "pandas",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Get optimized kwargs for map_batches operations.

        Provides sensible defaults for batch processing.

        Args:
            batch_size: Batch size (defaults to _TRAINING_BATCH_SIZE if None)
            num_cpus: Number of CPUs per task
            num_gpus: Number of GPUs per task
            batch_format: Batch format ("pandas", "arrow", "numpy", None)
            **kwargs: Additional kwargs to pass to map_batches

        Returns:
            Dictionary of kwargs for map_batches
        """
        from pipeline.utils.constants import _TRAINING_BATCH_SIZE

        if batch_size is None:
            batch_size = _TRAINING_BATCH_SIZE

        map_kwargs: Dict[str, Any] = {"batch_size": batch_size}

        if num_cpus is not None:
            map_kwargs["num_cpus"] = num_cpus
        if num_gpus is not None:
            map_kwargs["num_gpus"] = num_gpus
        if batch_format is not None:
            map_kwargs["batch_format"] = batch_format

        map_kwargs.update(kwargs)
        return map_kwargs

