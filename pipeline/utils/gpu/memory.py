"""GPU utilities for proper CUDA memory management and device handling.

Provides production-grade GPU memory management, device selection, error handling,
and cleanup utilities following NVIDIA best practices.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Any, Iterator, Optional, Dict, Tuple

logger = logging.getLogger(__name__)

_DEFAULT_DEVICE_ID = 0
_DEFAULT_SAFETY_MARGIN = 0.1


def get_cuda_device(device_id: Optional[int] = None) -> int:
    """Get CUDA device ID, validating availability.

    Args:
        device_id: Specific device ID (None = use CUDA_VISIBLE_DEVICES or 0)

    Returns:
        Valid CUDA device ID

    Raises:
        GPUError: If CUDA not available or device invalid
    """
    try:
        import torch

        if not torch.cuda.is_available():
            raise GPUError("CUDA not available on this system")

        if device_id is None:
            device_id = _DEFAULT_DEVICE_ID

        num_devices = torch.cuda.device_count()
        if device_id < 0 or device_id >= num_devices:
            raise GPUError(
                f"Invalid CUDA device {device_id}. Available devices: 0-{num_devices - 1}"
            )

        torch.cuda.set_device(device_id)
        device_name = torch.cuda.get_device_name(device_id)
        logger.info(f"Using CUDA device {device_id}: {device_name}")

        return device_id
    except ImportError:
        raise GPUError("PyTorch not available, cannot determine CUDA device") from None


def get_gpu_memory_info(device_id: Optional[int] = None) -> Dict[str, int]:
    """Get GPU memory information using pynvml (nvidia-ml-py).

    Uses NVIDIA's pynvml library for accurate GPU memory information.
    Falls back to PyTorch if pynvml not available.

    Args:
        device_id: CUDA device ID (None = current device)

    Returns:
        Dictionary with memory info (total, allocated, reserved, free)
    """
    # Try pynvml first (NVIDIA's official library)
    try:
        import pynvml  # type: ignore[attr-defined]
        
        pynvml.nvmlInit()
        if device_id is None:
            # Try to get current device from PyTorch if available
            try:
                import torch
                device_id = torch.cuda.current_device() if torch.cuda.is_available() else 0
            except ImportError:
                device_id = 0
        
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        
        return {
            "total": mem_info.total,
            "allocated": mem_info.used,  # pynvml reports used memory
            "reserved": mem_info.used,  # pynvml doesn't distinguish reserved
            "free": mem_info.free,
        }
    except ImportError:
        # Fallback to PyTorch if pynvml not available
        try:
            import torch

            if not torch.cuda.is_available():
                return {"total": 0, "allocated": 0, "reserved": 0, "free": 0}

            if device_id is None:
                device_id = torch.cuda.current_device()

            total = torch.cuda.get_device_properties(device_id).total_memory
            allocated = torch.cuda.memory_allocated(device_id)
            reserved = torch.cuda.memory_reserved(device_id)
            free = total - reserved

            return {
                "total": total,
                "allocated": allocated,
                "reserved": reserved,
                "free": free,
            }
        except (ImportError, RuntimeError, AttributeError) as e:
            logger.warning(f"Failed to get GPU memory info: {e}")
            return {"total": 0, "allocated": 0, "reserved": 0, "free": 0}
    except Exception as e:
        logger.warning(f"Failed to get GPU memory info via pynvml: {e}")
        # Fallback to PyTorch
        try:
            import torch
            if device_id is None:
                device_id = torch.cuda.current_device() if torch.cuda.is_available() else 0
            total = torch.cuda.get_device_properties(device_id).total_memory
            allocated = torch.cuda.memory_allocated(device_id)
            reserved = torch.cuda.memory_reserved(device_id)
            return {
                "total": total,
                "allocated": allocated,
                "reserved": reserved,
                "free": total - reserved,
            }
        except Exception:
            return {"total": 0, "allocated": 0, "reserved": 0, "free": 0}


def check_gpu_memory(
    required_bytes: int, device_id: Optional[int] = None, safety_margin: float = _DEFAULT_SAFETY_MARGIN
) -> Tuple[bool, Dict[str, int]]:
    """Check if sufficient GPU memory is available.

    Args:
        required_bytes: Required memory in bytes
        device_id: CUDA device ID (None = current device)
        safety_margin: Safety margin as fraction (default: 10%)

    Returns:
        Tuple of (has_memory, memory_info)
    """
    mem_info = get_gpu_memory_info(device_id)
    available = mem_info["free"]
    required_with_margin = int(required_bytes * (1 + safety_margin))

    has_memory = available >= required_with_margin

    if not has_memory:
        logger.warning(
            f"Insufficient GPU memory: required {required_with_margin}, "
            f"available {available}, total {mem_info['total']}"
        )

    return has_memory, mem_info


@contextmanager
def gpu_memory_cleanup(device_id: Optional[int] = None) -> Iterator[None]:
    """Context manager for GPU memory cleanup.

    Clears PyTorch cache and synchronizes device on exit.

    Args:
        device_id: CUDA device ID (None = current device)

    Yields:
        None
    """
    try:
        yield
    finally:
        try:
            import torch  # https://pytorch.org/

            if torch.cuda.is_available():
                if device_id is not None:
                    torch.cuda.set_device(device_id)
                # Clear cache to free unused memory
                torch.cuda.empty_cache()
                # Synchronize to ensure all operations complete
                torch.cuda.synchronize()
                logger.debug(f"GPU memory cache cleared for device {device_id or 'current'}")
        except (RuntimeError, AttributeError) as e:
            logger.warning(f"Error during GPU memory cleanup: {e}")


def clear_gpu_cache(device_id: Optional[int] = None) -> None:
    """Clear GPU memory cache.

    Args:
        device_id: CUDA device ID (None = all devices)
    """
    try:
        import torch  # https://pytorch.org/

        if not torch.cuda.is_available():
            return

        if device_id is not None:
            torch.cuda.set_device(device_id)
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        else:
            # Clear cache for all devices
            for dev_id in range(torch.cuda.device_count()):
                torch.cuda.set_device(dev_id)
                torch.cuda.empty_cache()
            torch.cuda.synchronize()

        logger.debug("GPU memory cache cleared")
    except (RuntimeError, AttributeError) as e:
        logger.warning(f"Error clearing GPU cache: {e}")


def set_cuda_device(device_id: int) -> None:
    """Set CUDA device and validate.

    Args:
        device_id: CUDA device ID

    Raises:
        RuntimeError: If device invalid or CUDA unavailable
    """
    try:
        import torch  # https://pytorch.org/

        if not torch.cuda.is_available():
            raise GPUError("CUDA not available")

        num_devices = torch.cuda.device_count()
        if device_id < 0 or device_id >= num_devices:
            raise GPUError(f"Invalid device {device_id}, available: 0-{num_devices - 1}")

        torch.cuda.set_device(device_id)
        logger.info(f"Set CUDA device to {device_id}: {torch.cuda.get_device_name(device_id)}")
    except ImportError:
        raise GPUError("PyTorch not available") from None


def synchronize_cuda(device_id: Optional[int] = None) -> None:
    """Synchronize CUDA operations.

    Ensures all GPU operations complete before proceeding.

    Args:
        device_id: CUDA device ID (None = current device)
    """
    try:
        import torch  # https://pytorch.org/

        if torch.cuda.is_available():
            if device_id is not None:
                torch.cuda.set_device(device_id)
            torch.cuda.synchronize()
    except (RuntimeError, AttributeError) as e:
        logger.warning(f"Error synchronizing CUDA: {e}")


def check_cuda_errors() -> None:
    """Check for CUDA errors and raise if any found.

    Raises:
        RuntimeError: If CUDA error detected
    """
    try:
        import torch  # https://pytorch.org/

        if torch.cuda.is_available():
            # PyTorch automatically checks CUDA errors, but we can be explicit
            torch.cuda.synchronize()
    except RuntimeError as e:
        if "CUDA" in str(e):
            logger.error(f"CUDA error detected: {e}")
            raise
        raise


@contextmanager
def cuda_device_context(device_id: int) -> Iterator[None]:
    """Context manager for CUDA device selection.

    Sets device on enter, restores on exit.

    Args:
        device_id: CUDA device ID

    Yields:
        None
    """
    try:
        import torch  # https://pytorch.org/

        if torch.cuda.is_available():
            previous_device = torch.cuda.current_device()
            set_cuda_device(device_id)
            yield
            torch.cuda.set_device(previous_device)
        else:
            yield
    except (RuntimeError, AttributeError) as e:
        logger.error(f"Error in CUDA device context: {e}")
        raise


def get_cupy_memory_pool() -> Optional[Any]:
    """Get cuPy memory pool for efficient memory management.

    Returns:
        cuPy memory pool or None if cuPy not available
    """
    try:
        import cupy as cp  # https://docs.cupy.dev/

        # Get default memory pool
        mempool = cp.get_default_memory_pool()
        return mempool
    except ImportError:
        return None


def clear_cupy_memory_pool() -> None:
    """Clear cuPy memory pool to free unused memory."""
    try:
        import cupy as cp  # https://docs.cupy.dev/

        mempool = cp.get_default_memory_pool()
        mempool.free_all_blocks()
        pinned_mempool = cp.get_default_pinned_memory_pool()
        pinned_mempool.free_all_blocks()
        logger.debug("cuPy memory pools cleared")
    except ImportError:
        pass
    except (RuntimeError, AttributeError) as e:
        logger.warning(f"Error clearing cuPy memory pool: {e}")


def get_cudf_memory_info() -> Dict[str, Any]:
    """Get cuDF memory information.

    Returns:
        Dictionary with cuDF memory info
    """
    # cuDF doesn't expose memory pool info directly
    # But we can check if it's using RMM (RAPIDS Memory Manager)
    try:
        import rmm  # type: ignore[attr-defined]

        return {
            "rmm_enabled": True,
            "rmm_pool_size": getattr(rmm, "pool_size", None),
        }
    except ImportError:
        return {"rmm_enabled": False}


def enable_cudf_rmm_pool(pool_size: Optional[int] = None) -> None:
    """Enable RMM (RAPIDS Memory Manager) pool for cuDF.

    DEPRECATED: Use pipeline.utils.rapids_init.initialize_rmm_pool() instead.

    Args:
        pool_size: Pool size in bytes (None = use default)
    """
    logger.warning(
        "enable_cudf_rmm_pool() is deprecated. "
        "Use pipeline.utils.rapids_init.initialize_rmm_pool() instead."
    )
    try:
        from pipeline.utils.gpu.rapids import initialize_rmm_pool

        initialize_rmm_pool(pool_size=pool_size)
    except (ImportError, RuntimeError, ValueError) as e:
        logger.warning(f"Error enabling RMM pool: {e}")

