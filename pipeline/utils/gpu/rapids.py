"""RAPIDS (cuDF/cuML) initialization utilities.

Properly initializes RAPIDS libraries with optimal configuration for production use.
Follows NVIDIA RAPIDS best practices for memory management and performance.

NVIDIA Libraries:
- RAPIDS cuDF: https://docs.rapids.ai/api/cudf/stable/
- RAPIDS cuML: https://docs.rapids.ai/api/cuml/stable/
- RMM: https://docs.rapids.ai/api/rmm/stable/
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Dict

logger = logging.getLogger(__name__)


def initialize_rmm_pool(
    pool_size: Optional[int] = None,
    initial_pool_size: Optional[int] = None,
    maximum_pool_size: Optional[int] = None,
    enable_logging: bool = False,
) -> bool:
    """Initialize RMM (RAPIDS Memory Manager) pool for optimal cuDF/cuML performance.

    RMM provides efficient GPU memory management for RAPIDS libraries.
    Should be called once at application startup before using cuDF/cuML.

    Args:
        pool_size: Total pool size in bytes (None = auto-detect from GPU memory)
        initial_pool_size: Initial pool size in bytes (None = 50% of pool_size)
        maximum_pool_size: Maximum pool size in bytes (None = pool_size)
        enable_logging: Whether to enable RMM logging

    Returns:
        True if RMM initialized successfully, False otherwise
    """
    try:
        import rmm  # type: ignore[attr-defined]

        # Check if already initialized
        if rmm.is_initialized():
            logger.info("RMM already initialized")
            return True

        # Auto-detect pool size from GPU memory if not specified
        if pool_size is None:
            try:
                import torch  # https://pytorch.org/

                if torch.cuda.is_available():
                    # Use 50% of GPU memory for RMM pool
                    total_mem = torch.cuda.get_device_properties(0).total_memory
                    pool_size = total_mem // 2
                    logger.info(f"Auto-detected RMM pool size: {pool_size / 1024**3:.2f} GB")
                else:
                    logger.warning("CUDA not available, cannot auto-detect RMM pool size")
                    return False
            except ImportError:
                logger.warning("PyTorch not available, cannot auto-detect RMM pool size")
                return False

        if initial_pool_size is None:
            initial_pool_size = pool_size // 2  # Start with 50% of pool

        if maximum_pool_size is None:
            maximum_pool_size = pool_size

        # Initialize RMM with pool allocator
        # RMM pool allocator provides better performance than default allocator
        # It pre-allocates memory pools to reduce allocation overhead
        rmm.reinitialize(
            pool_allocator=True,
            initial_pool_size=initial_pool_size,
            maximum_pool_size=maximum_pool_size,
            enable_logging=enable_logging,
            # Use managed memory if available (unified memory)
            # This allows oversubscription and better memory management
            managed_memory=False,  # Set to True if unified memory available
        )

        logger.info(
            f"RMM pool initialized: initial={initial_pool_size / 1024**3:.2f}GB, "
            f"max={maximum_pool_size / 1024**3:.2f}GB"
        )
        return True
    except ImportError:
        logger.warning("RMM not available. Install via: conda install -c rapidsai rmm")
        return False
    except Exception as e:
        logger.error(f"Failed to initialize RMM pool: {e}", exc_info=True)
        return False


def check_cudf_compatibility() -> Dict[str, Any]:
    """Check cuDF version and compatibility.

    Returns:
        Dictionary with compatibility information
    """
    try:
        import cudf  # type: ignore[attr-defined]

        version = cudf.__version__
        logger.info(f"cuDF version: {version}")

        # Check for known compatibility issues
        issues = []
        # Parse version string (may be in format "23.12.0" or "23.12.0+cu12")
        version_parts = version.split("+")[0].split(".")
        try:
            major, minor = int(version_parts[0]), int(version_parts[1])
            if major < 23 or (major == 23 and minor < 12):
                issues.append(f"cuDF version {version} < 23.12.0 may have performance issues")
        except (ValueError, IndexError):
            issues.append(f"Could not parse cuDF version: {version}")

        return {
            "available": True,
            "version": version,
            "issues": issues,
        }
    except ImportError:
        return {"available": False, "version": None, "issues": ["cuDF not installed"]}
    except Exception as e:
        logger.warning(f"Error checking cuDF compatibility: {e}")
        return {"available": False, "version": None, "issues": [str(e)]}


def check_cuml_compatibility() -> Dict[str, Any]:
    """Check cuML version and compatibility.

    Returns:
        Dictionary with compatibility information
    """
    try:
        import cuml  # type: ignore[attr-defined]

        version = cuml.__version__
        logger.info(f"cuML version: {version}")

        # Check for known compatibility issues
        issues = []
        # Parse version string (may be in format "23.12.0" or "23.12.0+cu12")
        version_parts = version.split("+")[0].split(".")
        try:
            major, minor = int(version_parts[0]), int(version_parts[1])
            if major < 23 or (major == 23 and minor < 12):
                issues.append(f"cuML version {version} < 23.12.0 may have performance issues")
        except (ValueError, IndexError):
            issues.append(f"Could not parse cuML version: {version}")

        return {
            "available": True,
            "version": version,
            "issues": issues,
        }
    except ImportError:
        return {"available": False, "version": None, "issues": ["cuML not installed"]}
    except Exception as e:
        logger.warning(f"Error checking cuML compatibility: {e}")
        return {"available": False, "version": None, "issues": [str(e)]}


def optimize_cudf_settings() -> None:
    """Apply optimal cuDF settings for performance.

    Configures cuDF options for best performance in production.
    Follows NVIDIA RAPIDS best practices.
    """
    try:
        import cudf  # type: ignore[attr-defined]

        # Set cuDF options for optimal performance
        # These are cuDF-specific optimizations
        # Use 32-bit types when possible (faster, less memory)
        cudf.set_option("default_integer_bitwidth", 32)  # Use 32-bit integers when possible
        cudf.set_option("default_float_bitwidth", 32)  # Use 32-bit floats when possible

        # Enable string operations optimization (if available)
        # cuDF has optimized string operations on GPU
        try:
            # Some cuDF versions support string optimization
            cudf.set_option("string_optimization", True)
        except Exception:
            pass  # Option may not be available in all versions

        logger.info("cuDF settings optimized for performance")
    except ImportError:
        logger.warning("cuDF not available, cannot optimize settings")
    except Exception as e:
        logger.warning(f"Error optimizing cuDF settings: {e}")


def get_rmm_memory_info() -> Dict[str, Any]:
    """Get RMM memory pool information.

    Returns:
        Dictionary with RMM memory statistics
    """
    try:
        import rmm  # type: ignore[attr-defined]

        if not rmm.is_initialized():
            return {"initialized": False}

        # Get memory statistics
        stats = rmm.mr.get_statistics()
        return {
            "initialized": True,
            "statistics": stats,
        }
    except ImportError:
        return {"initialized": False, "error": "RMM not available"}
    except Exception as e:
        logger.warning(f"Error getting RMM memory info: {e}")
        return {"initialized": False, "error": str(e)}


def initialize_rapids_environment() -> Dict[str, Any]:
    """Initialize complete RAPIDS environment with optimal settings.

    Should be called once at application startup.

    Returns:
        Dictionary with initialization results
    """
    results: Dict[str, Any] = {
        "rmm_initialized": False,
        "cudf_available": False,
        "cuml_available": False,
        "warnings": [],
    }

    # Initialize RMM pool first (required for cuDF/cuML)
    results["rmm_initialized"] = initialize_rmm_pool()

    # Check cuDF compatibility
    cudf_info = check_cudf_compatibility()
    results["cudf_available"] = cudf_info.get("available", False)
    if cudf_info.get("available"):
        if cudf_info.get("issues"):
            results["warnings"].extend(cudf_info["issues"])
        optimize_cudf_settings()

    # Check cuML compatibility
    cuml_info = check_cuml_compatibility()
    results["cuml_available"] = cuml_info.get("available", False)
    if cuml_info.get("issues"):
        results["warnings"].extend(cuml_info["issues"])

    return results

