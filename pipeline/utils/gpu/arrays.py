"""GPU-accelerated array operations using cuPy.

Provides GPU-accelerated replacements for NumPy operations:
- Statistical operations (mean, std, min, max, median)
- Array transformations
- Mathematical operations
- Linear algebra operations
"""

from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

# Try to import cuPy for GPU array operations
try:
    import cupy as cp  # type: ignore[attr-defined]

    _CUPY_AVAILABLE = True
except ImportError:
    import numpy as np

    _CUPY_AVAILABLE = False
    _NP = np
    logger.warning("cuPy not available - GPU array operations will fallback to NumPy")


def gpu_array_stats(
    array: Any,
    num_gpus: int = 1,
) -> Dict[str, float]:
    """Compute GPU-accelerated statistics on array.

    Args:
        array: NumPy array or cuPy array
        num_gpus: Number of GPUs to use

    Returns:
        Dictionary with statistics (mean, std, min, max, median)
    """
    # Validate num_gpus parameter
    if num_gpus < 0:
        raise GPUError(f"num_gpus must be >= 0, got {num_gpus}")

    if not _CUPY_AVAILABLE or num_gpus == 0:
        import numpy as np

        arr = np.asarray(array)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "median": float(np.median(arr)),
        }

    arr = None
    try:
        # Convert to cuPy array if needed
        if not isinstance(array, cp.ndarray):
            arr = cp.asarray(array)
        else:
            arr = array

        result = {
            "mean": float(cp.mean(arr)),
            "std": float(cp.std(arr)),
            "min": float(cp.min(arr)),
            "max": float(cp.max(arr)),
            "median": float(cp.median(arr)),
        }

        # Explicitly free GPU memory if we created the array
        if not isinstance(array, cp.ndarray) and arr is not None:
            del arr

        return result
    except (ValueError, TypeError, RuntimeError) as e:
        logger.warning(f"GPU array stats failed, falling back to CPU: {e}")
        # Cleanup on error
        if arr is not None and not isinstance(array, cp.ndarray):
            del arr
        import numpy as np

        arr = np.asarray(array)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "median": float(np.median(arr)),
        }
    finally:
        # Final cleanup
        if arr is not None and not isinstance(array, cp.ndarray):
            del arr


def gpu_normalize(
    array: Any,
    method: str = "zscore",
    num_gpus: int = 1,
) -> Any:
    """GPU-accelerated array normalization.

    Args:
        array: NumPy array or cuPy array
        method: Normalization method ("zscore", "minmax", "l2")
        num_gpus: Number of GPUs to use

    Returns:
        Normalized array (same type as input)
    """
    # Validate num_gpus parameter
    if num_gpus < 0:
        raise GPUError(f"num_gpus must be >= 0, got {num_gpus}")

    if not _CUPY_AVAILABLE or num_gpus == 0:
        import numpy as np

        arr = np.asarray(array)
        if method == "zscore":
            mean = np.mean(arr)
            std = np.std(arr)
            return (arr - mean) / (std + 1e-8)
        elif method == "minmax":
            min_val = np.min(arr)
            max_val = np.max(arr)
            return (arr - min_val) / (max_val - min_val + 1e-8)
        elif method == "l2":
            norm = np.linalg.norm(arr)
            return arr / (norm + 1e-8)
        return arr

    arr = None
    result = None
    try:
        if not isinstance(array, cp.ndarray):
            arr = cp.asarray(array)
        else:
            arr = array

        if method == "zscore":
            mean = cp.mean(arr)
            std = cp.std(arr)
            result = (arr - mean) / (std + 1e-8)
        elif method == "minmax":
            min_val = cp.min(arr)
            max_val = cp.max(arr)
            result = (arr - min_val) / (max_val - min_val + 1e-8)
        elif method == "l2":
            norm = cp.linalg.norm(arr)
            result = arr / (norm + 1e-8)
        else:
            result = arr
        
        # If we created a new array and result is different, cleanup
        if not isinstance(array, cp.ndarray) and arr is not result:
            # Only delete if we created arr and it's not the result
            pass  # Keep arr for return
        
        return result
    except (ValueError, TypeError, RuntimeError) as e:
        logger.warning(f"GPU normalize failed, falling back to CPU: {e}")
        # Cleanup on error
        if arr is not None and not isinstance(array, cp.ndarray) and arr is not result:
            del arr
        import numpy as np

        arr = np.asarray(array)
        if method == "zscore":
            mean = np.mean(arr)
            std = np.std(arr)
            return (arr - mean) / (std + 1e-8)
        elif method == "minmax":
            min_val = np.min(arr)
            max_val = np.max(arr)
            return (arr - min_val) / (max_val - min_val + 1e-8)
        elif method == "l2":
            norm = np.linalg.norm(arr)
            return arr / (norm + 1e-8)
        return arr


def gpu_remove_outliers(
    array: Any,
    threshold: float = 3.0,
    num_gpus: int = 1,
) -> Any:
    """GPU-accelerated outlier removal using z-score.

    Args:
        array: NumPy array or cuPy array
        threshold: Z-score threshold
        num_gpus: Number of GPUs to use

    Returns:
        Array with outliers removed
    """
    # Validate num_gpus parameter
    if num_gpus < 0:
        raise GPUError(f"num_gpus must be >= 0, got {num_gpus}")

    if not _CUPY_AVAILABLE or num_gpus == 0:
        import numpy as np

        arr = np.asarray(array)
        mean = np.mean(arr)
        std = np.std(arr)
        z_scores = np.abs((arr - mean) / (std + 1e-8))
        return arr[z_scores < threshold]

    arr = None
    z_scores = None
    filtered = None
    try:
        if not isinstance(array, cp.ndarray):
            arr = cp.asarray(array)
        else:
            arr = array

        mean = cp.mean(arr)
        std = cp.std(arr)
        z_scores = cp.abs((arr - mean) / (std + 1e-8))
        filtered = arr[z_scores < threshold]
        
        # Explicitly free intermediate arrays
        del z_scores
        # Only delete arr if we created it
        if not isinstance(array, cp.ndarray):
            del arr
        
        return filtered
    except (ValueError, TypeError, RuntimeError) as e:
        logger.warning(f"GPU remove_outliers failed, falling back to CPU: {e}")
        # Cleanup on error
        if z_scores is not None:
            del z_scores
        if arr is not None and not isinstance(array, cp.ndarray):
            del arr
        import numpy as np

        arr = np.asarray(array)
        mean = np.mean(arr)
        std = np.std(arr)
        z_scores = np.abs((arr - mean) / (std + 1e-8))
        result = arr[z_scores < threshold]
        return result
    finally:
        # Final cleanup
        if z_scores is not None:
            del z_scores
        if arr is not None and not isinstance(array, cp.ndarray) and arr is not filtered:
            del arr

