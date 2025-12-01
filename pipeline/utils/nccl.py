"""NVIDIA NCCL utilities for multi-GPU communication.

NCCL (NVIDIA Collective Communications Library) provides optimized
multi-GPU and multi-node communication primitives. This module provides
utilities for initializing NCCL groups and performing collective operations
in Ray-based distributed GPU workloads.

NVIDIA Libraries:
- NCCL: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/
- PyTorch Distributed: https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
"""

from __future__ import annotations

import logging

import torch  # https://pytorch.org/
import torch.distributed as dist  # https://pytorch.org/docs/stable/distributed.html

logger = logging.getLogger(__name__)


class NCCLGroup:
    """NCCL communication group for multi-GPU operations.

    Manages NCCL initialization and provides collective communication
    primitives for distributed GPU operations.
    """

    def __init__(
        self,
        backend: str = "nccl",
        init_method: Optional[str] = None,
        world_size: Optional[int] = None,
        rank: Optional[int] = None,
    ):
        """Initialize NCCL communication group.

        Args:
            backend: Communication backend ('nccl' for multi-GPU)
            init_method: Initialization method (e.g., 'tcp://localhost:23456')
            world_size: Number of processes in group
            rank: Rank of current process
        """
        self.backend = backend
        self.init_method = init_method
        self.world_size = world_size
        self.rank = rank
        self._initialized = False

    def initialize(self) -> None:
        """Initialize NCCL process group.

        Sets up distributed communication for multi-GPU operations.
        Uses optimal NCCL settings for performance.
        """
        if self._initialized:
            logger.warning("NCCL group already initialized")
            return

        if not torch.cuda.is_available():
            logger.warning("CUDA not available, skipping NCCL initialization")
            return

        try:
            # Set NCCL environment variables for optimal performance
            import os

            # Enable NCCL debug if needed (disable in production for performance)
            if os.environ.get("NCCL_DEBUG") is None:
                os.environ["NCCL_DEBUG"] = "WARN"  # Only show warnings

            # Set NCCL timeout (default is 10 minutes, increase for large operations)
            if os.environ.get("NCCL_TIMEOUT") is None:
                os.environ["NCCL_TIMEOUT"] = "1800"  # 30 minutes

            # Enable NCCL async operations for better performance
            if os.environ.get("NCCL_ASYNC_ERROR_HANDLING") is None:
                os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"

            # Initialize process group
            if self.init_method is None:
                # Use default initialization (environment variables)
                dist.init_process_group(
                    backend=self.backend,
                    init_method="env://",
                    timeout=torch.distributed.default_pg_timeout,  # Use default timeout
                )
            else:
                dist.init_process_group(
                    backend=self.backend,
                    init_method=self.init_method,
                    world_size=self.world_size,
                    rank=self.rank,
                    timeout=torch.distributed.default_pg_timeout,
                )

            self._initialized = True
            logger.info(
                f"NCCL group initialized: rank={dist.get_rank()}, "
                f"world_size={dist.get_world_size()}, backend={self.backend}"
            )
        except (RuntimeError, ValueError, AttributeError) as e:
            logger.error(f"Failed to initialize NCCL group: {e}", exc_info=True)
            raise

    def all_reduce(self, tensor: torch.Tensor, op: dist.ReduceOp = dist.ReduceOp.SUM) -> torch.Tensor:
        """Perform all-reduce operation across all GPUs.

        Args:
            tensor: Input tensor (must be on GPU)
            op: Reduction operation (SUM, MEAN, etc.)

        Returns:
            Reduced tensor (same on all processes)
        """
        if not self._initialized:
            self.initialize()

        if not tensor.is_cuda:
            tensor = tensor.cuda(non_blocking=False)  # Blocking for NCCL operations

        # NCCL all_reduce is GPU-accelerated collective operation
        # It performs reduction across all GPUs efficiently
        dist.all_reduce(tensor, op=op, async_op=False)  # Synchronous for correctness

        # Synchronize after NCCL operation to ensure completion
        # NCCL operations are asynchronous by default, must synchronize
        if tensor.is_cuda:
            torch.cuda.synchronize()

        return tensor

    def all_gather(self, tensor: torch.Tensor) -> list[torch.Tensor]:
        """Gather tensors from all processes.

        Args:
            tensor: Input tensor to gather

        Returns:
            List of tensors from all processes
        """
        if not self._initialized:
            self.initialize()

        if not tensor.is_cuda:
            tensor = tensor.cuda(non_blocking=False)  # Blocking for NCCL operations

        world_size = dist.get_world_size()
        # Pre-allocate output tensors on GPU for efficiency
        gathered = [torch.zeros_like(tensor, device=tensor.device) for _ in range(world_size)]
        # NCCL all_gather collects tensors from all GPUs
        dist.all_gather(gathered, tensor, async_op=False)  # Synchronous for correctness
        # Synchronize after NCCL operation to ensure completion
        if tensor.is_cuda:
            torch.cuda.synchronize()
        return gathered

    def broadcast(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
        """Broadcast tensor from source to all processes.

        Args:
            tensor: Tensor to broadcast (on src) or receive (on others)
            src: Source rank

        Returns:
            Broadcasted tensor
        """
        if not self._initialized:
            self.initialize()

        if not tensor.is_cuda:
            tensor = tensor.cuda(non_blocking=False)  # Blocking for NCCL operations

        # NCCL broadcast sends tensor from src to all GPUs
        dist.broadcast(tensor, src=src, async_op=False)  # Synchronous for correctness
        # Synchronize after NCCL operation to ensure completion
        if tensor.is_cuda:
            torch.cuda.synchronize()
        return tensor

    def cleanup(self) -> None:
        """Cleanup NCCL process group and GPU resources."""
        if self._initialized:
            try:
                # Synchronize before cleanup
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                dist.destroy_process_group()
                self._initialized = False
                logger.info("NCCL group cleaned up")
            except Exception as e:
                logger.warning(f"Error during NCCL cleanup: {e}")
                self._initialized = False


def initialize_nccl_for_ray_actors(
    backend: str = "nccl",
    init_method: Optional[str] = None,
    enable_async: bool = True,
) -> Optional[NCCLGroup]:
    """Initialize NCCL for Ray GPU actors.

    Helper function to set up NCCL communication in Ray remote actors.
    Note: Ray handles inter-actor communication efficiently via object store.
    NCCL is most beneficial for:
    - In-process multi-GPU operations within a single actor
    - Distributed training scenarios
    - Large tensor reductions that benefit from GPU-to-GPU direct communication

    Args:
        backend: Communication backend ('nccl' for NVIDIA GPUs)
        init_method: Initialization method (None = use environment variables)
        enable_async: Whether to enable async error handling for better performance

    Returns:
        NCCLGroup instance if initialization successful, None otherwise
    """
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, skipping NCCL initialization")
        return None

    try:
        # Check if already initialized
        if dist.is_initialized():
            logger.info("NCCL already initialized")
            return NCCLGroup()

        # Set NCCL environment variables for optimal performance
        if enable_async:
            import os

            os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")

        # Initialize NCCL group
        # For Ray actors, this enables in-process multi-GPU operations
        group = NCCLGroup(backend=backend, init_method=init_method)
        group.initialize()
        return group
    except Exception as e:
        logger.warning(f"Failed to initialize NCCL for Ray actors: {e}")
        return None


def all_reduce_tensors(
    tensors: list[torch.Tensor],
    op: dist.ReduceOp = dist.ReduceOp.SUM,
    group: Optional[NCCLGroup] = None,
) -> list[torch.Tensor]:
    """Perform all-reduce on multiple tensors.

    Convenience function for reducing multiple tensors across GPUs.

    Args:
        tensors: List of tensors to reduce
        op: Reduction operation
        group: NCCL group (creates new one if None)

    Returns:
        List of reduced tensors
    """
    if group is None:
        group = NCCLGroup()
        group.initialize()

    reduced = []
    for tensor in tensors:
        reduced_tensor = group.all_reduce(tensor.clone(), op=op)
        reduced.append(reduced_tensor)

    return reduced


def gather_tensors_across_gpus(
    tensor: torch.Tensor,
    group: Optional[NCCLGroup] = None,
) -> torch.Tensor:
    """Gather tensors from all GPUs and concatenate.

    Useful for collecting results from distributed GPU workers.

    Args:
        tensor: Tensor to gather from each GPU
        group: NCCL group (creates new one if None)

    Returns:
        Concatenated tensor from all GPUs
    """
    if group is None:
        group = NCCLGroup()
        group.initialize()

    gathered = group.all_gather(tensor)
    return torch.cat(gathered, dim=0)

