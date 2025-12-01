"""GPU monitoring and health check utilities.

Provides GPU utilization monitoring, health checks, and performance metrics
for production GPU infrastructure management.

NVIDIA Libraries:
- PyTorch: https://pytorch.org/docs/stable/cuda.html
- cuPy: https://docs.cupy.dev/
- nvidia-ml-py: https://github.com/NVIDIA/nvidia-ml-py (optional)
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Dict

logger = logging.getLogger(__name__)


def get_gpu_utilization(device_id: Optional[int] = None) -> Dict[str, Any]:
    """Get GPU utilization metrics.

    Args:
        device_id: CUDA device ID (None = all devices)

    Returns:
        Dictionary with GPU utilization metrics
    """
    try:
        import torch  # https://pytorch.org/

        if not torch.cuda.is_available():
            return {"available": False}

        # Try to use nvidia-ml-py for detailed metrics
        try:
            import pynvml  # type: ignore[attr-defined]

            pynvml.nvmlInit()
            if device_id is None:
                device_id = torch.cuda.current_device()

            handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)

            return {
                "available": True,
                "device_id": device_id,
                "gpu_utilization": util.gpu,
                "memory_utilization": util.memory,
                "memory_used": mem_info.used,
                "memory_total": mem_info.total,
                "memory_free": mem_info.free,
                "temperature": temp,
            }
        except ImportError:
            # Fallback to basic PyTorch metrics
            from pipeline.utils.gpu.memory import get_gpu_memory_info

            mem_info = get_gpu_memory_info(device_id)
            return {
                "available": True,
                "device_id": device_id or torch.cuda.current_device(),
                "memory_used": mem_info["allocated"],
                "memory_total": mem_info["total"],
                "memory_free": mem_info["free"],
                "gpu_utilization": None,  # Not available without nvidia-ml-py
                "temperature": None,
            }
    except Exception as e:
        logger.warning(f"Failed to get GPU utilization: {e}")
        return {"available": False, "error": str(e)}


def check_gpu_health(device_id: Optional[int] = None) -> Dict[str, Any]:
    """Perform GPU health check.

    Args:
        device_id: CUDA device ID (None = current device)

    Returns:
        Dictionary with health check results
    """
    try:
        import torch  # https://pytorch.org/

        if not torch.cuda.is_available():
            return {"healthy": False, "reason": "CUDA not available"}

        if device_id is None:
            device_id = torch.cuda.current_device()

        # Check device exists
        num_devices = torch.cuda.device_count()
        if device_id >= num_devices:
            return {"healthy": False, "reason": f"Invalid device {device_id}"}

        # Check memory availability
        from pipeline.utils.gpu.memory import get_gpu_memory_info

        mem_info = get_gpu_memory_info(device_id)
        memory_usage_pct = (mem_info["allocated"] / mem_info["total"]) * 100 if mem_info["total"] > 0 else 0

        # Check temperature if available
        temp = None
        try:
            import pynvml  # type: ignore[attr-defined]

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        except ImportError:
            pass

        # Health criteria
        healthy = True
        warnings = []

        if memory_usage_pct > 95:
            healthy = False
            warnings.append(f"GPU memory usage critical: {memory_usage_pct:.1f}%")
        elif memory_usage_pct > 85:
            warnings.append(f"GPU memory usage high: {memory_usage_pct:.1f}%")

        if temp is not None and temp > 85:
            healthy = False
            warnings.append(f"GPU temperature critical: {temp}°C")
        elif temp is not None and temp > 80:
            warnings.append(f"GPU temperature high: {temp}°C")

        return {
            "healthy": healthy,
            "device_id": device_id,
            "device_name": torch.cuda.get_device_name(device_id),
            "memory_usage_pct": memory_usage_pct,
            "temperature": temp,
            "warnings": warnings,
        }
    except Exception as e:
        logger.error(f"GPU health check failed: {e}", exc_info=True)
        return {"healthy": False, "reason": str(e)}


def log_gpu_status() -> None:
    """Log current GPU status for debugging."""
    try:
        import torch  # https://pytorch.org/

        if not torch.cuda.is_available():
            logger.info("CUDA not available")
            return

        num_devices = torch.cuda.device_count()
        logger.info(f"CUDA available: {num_devices} device(s)")

        for device_id in range(num_devices):
            from pipeline.utils.gpu.memory import get_gpu_memory_info

            mem_info = get_gpu_memory_info(device_id)
            device_name = torch.cuda.get_device_name(device_id)
            logger.info(
                f"  Device {device_id}: {device_name} - "
                f"Memory: {mem_info['allocated'] / 1024**3:.2f}GB / "
                f"{mem_info['total'] / 1024**3:.2f}GB used"
            )
    except Exception as e:
        logger.warning(f"Failed to log GPU status: {e}")

