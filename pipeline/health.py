"""Health check endpoints for production monitoring."""

from __future__ import annotations

import logging
from typing import Any, Optional, Optional

import ray

logger = logging.getLogger(__name__)


class HealthChecker:
    """Health check utilities for pipeline monitoring."""

    @staticmethod
    def check_ray_cluster() -> dict[str, Any]:
        """Check Ray cluster health.

        Returns:
            Dictionary with health status and details
        """
        try:
            if not ray.is_initialized():
                return {
                    "healthy": False,
                    "status": "uninitialized",
                    "message": "Ray cluster not initialized",
                }

            from pipeline.utils.ray.monitoring import check_cluster_health

            health = check_cluster_health()
            return {
                "healthy": health.get("healthy", False),
                "status": "healthy" if health.get("healthy") else "unhealthy",
                "details": health,
            }
        except Exception as e:
            logger.error(f"Error checking Ray cluster health: {e}", exc_info=True)
            return {
                "healthy": False,
                "status": "error",
                "message": str(e),
            }

    @staticmethod
    def check_gpu_resources() -> dict[str, Any]:
        """Check GPU resource availability.

        Returns:
            Dictionary with GPU health status
        """
        try:
            import torch

            if not torch.cuda.is_available():
                return {
                    "healthy": False,
                    "status": "no_gpu",
                    "message": "CUDA not available",
                    "gpu_count": 0,
                }

            gpu_count = torch.cuda.device_count()
            gpu_info = []

            for i in range(gpu_count):
                try:
                    from pipeline.utils.gpu.memory import get_gpu_memory_info

                    mem_info = get_gpu_memory_info(i)
                    gpu_info.append(
                        {
                            "device_id": i,
                            "name": torch.cuda.get_device_name(i),
                            "memory_total": mem_info.get("total", 0),
                            "memory_allocated": mem_info.get("allocated", 0),
                            "memory_free": mem_info.get("free", 0),
                        }
                    )
                except Exception as e:
                    logger.warning(f"Error getting GPU {i} info: {e}")
                    gpu_info.append({"device_id": i, "error": str(e)})

            return {
                "healthy": True,
                "status": "available",
                "gpu_count": gpu_count,
                "gpus": gpu_info,
            }
        except ImportError:
            return {
                "healthy": False,
                "status": "torch_unavailable",
                "message": "PyTorch not available",
            }
        except Exception as e:
            logger.error(f"Error checking GPU resources: {e}", exc_info=True)
            return {
                "healthy": False,
                "status": "error",
                "message": str(e),
            }

    @staticmethod
    def check_disk_space(path: str, required_gb: int = 10) -> dict[str, Any]:
        """Check available disk space.

        Args:
            path: Path to check
            required_gb: Required space in GB

        Returns:
            Dictionary with disk space status
        """
        try:
            from pipeline.utils.resource_manager import check_disk_space

            has_space, available_bytes = check_disk_space(path, required_gb * 1024 * 1024 * 1024)
            available_gb = available_bytes / (1024 * 1024 * 1024)

            return {
                "healthy": has_space,
                "status": "sufficient" if has_space else "insufficient",
                "available_gb": round(available_gb, 2),
                "required_gb": required_gb,
                "path": path,
            }
        except Exception as e:
            logger.error(f"Error checking disk space: {e}", exc_info=True)
            return {
                "healthy": False,
                "status": "error",
                "message": str(e),
            }

    @staticmethod
    def get_overall_health(output_path: Optional[str] = None) -> dict[str, Any]:
        """Get overall pipeline health status.

        Args:
            output_path: Output path for disk space check

        Returns:
            Dictionary with overall health status
        """
        ray_health = HealthChecker.check_ray_cluster()
        gpu_health = HealthChecker.check_gpu_resources()

        health_checks = {
            "ray_cluster": ray_health,
            "gpu_resources": gpu_health,
        }

        if output_path:
            disk_health = HealthChecker.check_disk_space(output_path)
            health_checks["disk_space"] = disk_health

        overall_healthy = all(
            check.get("healthy", False) for check in health_checks.values()
        )

        return {
            "healthy": overall_healthy,
            "status": "healthy" if overall_healthy else "unhealthy",
            "checks": health_checks,
            "timestamp": __import__("time").time(),
        }

