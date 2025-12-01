"""Custom datasource for point cloud files.

Supports PCD, PLY, LAS, and other point cloud formats.
Follows Ray Data FileBasedDatasource API.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, Iterator, Optional, Union

import numpy as np  # https://numpy.org/
from ray.data._internal.arrow_block import ArrowBlockBuilder
from ray.data.block import Block

from pipeline.datasources.base import FileBasedDatasource
from pipeline.exceptions import DataSourceError, ConfigurationError
from pipeline.utils.decorators import handle_errors, log_execution_time

if TYPE_CHECKING:
    import pyarrow

logger = logging.getLogger(__name__)

# Constants
_MAX_POINTS = 1_000_000  # 1M points limit
_MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024 * 1024  # 10GB


class PointCloudDatasource(FileBasedDatasource):
    """Custom datasource for reading point cloud files.

    Supports common point cloud formats used in robotics:
    - PCD (Point Cloud Data)
    - PLY (Polygon File Format)
    - LAS/LAZ (LiDAR formats)
    """

    def __init__(
        self,
        paths: Union[str, list[str]],
        max_points: Optional[int] = None,
        downsample_large: bool = True,
        **kwargs: Any,
    ):
        """Initialize point cloud datasource.

        Args:
            paths: Point cloud file path(s) or directory path(s)
            max_points: Maximum number of points to read (None = use default limit)
            downsample_large: Whether to downsample large point clouds
            **kwargs: Additional FileBasedDatasource options

        Raises:
            ValueError: If parameters are invalid
        """
        super().__init__(paths=paths, **kwargs)
        
        # Validate parameters
        if max_points is not None and max_points <= 0:
            raise ConfigurationError(f"max_points must be positive, got {max_points}")
        
        self.max_points = max_points if max_points is not None else _MAX_POINTS
        self.downsample_large = bool(downsample_large)

    @log_execution_time
    @handle_errors(error_class=DataSourceError)
    def _read_stream(
        self, f: "pyarrow.NativeFile", path: str
    ) -> Iterator[Block]:
        """Read point cloud file and yield point cloud blocks.

        Args:
            f: pyarrow.NativeFile handle (point clouds require direct file access)
            path: Path to point cloud file

        Yields:
            Block objects (pyarrow.Table) with point cloud data

        Raises:
            DataSourceError: If reading fails

        Note:
            Point cloud libraries require direct file access. For cloud storage,
            files must be copied locally first.
        """
        self._validate_file_handle(f, path)
        
        # Validate file exists (for local files)
        if not path.startswith(("s3://", "gs://", "hdfs://", "abfss://")):
            if not os.path.exists(path):
                raise DataSourceError(f"Point cloud file does not exist: {path}")
            
            if not os.path.isfile(path):
                raise DataSourceError(f"Point cloud path is not a file: {path}")
            
            # Check file size
            file_size = os.path.getsize(path)
            if file_size > _MAX_FILE_SIZE_BYTES:
                raise DataSourceError(
                    f"Point cloud file {path} is {file_size} bytes, "
                    f"exceeds maximum size of {_MAX_FILE_SIZE_BYTES}"
                )
        
        path_lower = path.lower()

        try:
            if path_lower.endswith(".pcd"):
                yield from self._read_pcd(path)
            elif path_lower.endswith(".ply"):
                yield from self._read_ply(path)
            elif path_lower.endswith((".las", ".laz")):
                yield from self._read_las(path)
            else:
                raise DataSourceError(f"Unsupported point cloud format: {path}")

        except Exception as e:
            logger.error(f"Error reading point cloud {path}: {e}", exc_info=True)
            # Re-raise exception instead of wrapping in block
            raise DataSourceError(f"Failed to read point cloud {path}: {e}") from e

    def _read_pcd(self, path: str) -> Iterator[Block]:
        """Read PCD file.

        Args:
            path: Path to PCD file

        Yields:
            Block object with point cloud data

        Raises:
            DataSourceError: If reading fails
        """
        try:
            import open3d as o3d  # http://www.open3d.org/
        except ImportError:
            raise DataSourceError(
                "open3d not installed. Install with: pip install open3d"
            ) from None

        try:
            pcd = o3d.io.read_point_cloud(path)
            if pcd is None:
                raise DataSourceError(f"Failed to read PCD file: {path}")
            
            points = np.asarray(pcd.points)
            if len(points) == 0:
                # Empty point cloud - return empty block
                builder = ArrowBlockBuilder()
                builder.add({"num_points": 0, "points": []})
                yield builder.build()
                return

            num_points = len(points)
            downsampled = False

            if num_points > self.max_points and self.downsample_large:
                logger.warning(
                    f"Point cloud {path} has {num_points} points, "
                    f"downsampling to {self.max_points} points"
                )
                # Downsample for large point clouds
                indices = np.random.choice(num_points, self.max_points, replace=False)
                points = points[indices]
                if pcd.has_colors():
                    colors = np.asarray(pcd.colors)[indices]
                else:
                    colors = None
                if pcd.has_normals():
                    normals = np.asarray(pcd.normals)[indices]
                else:
                    normals = None
                downsampled = True
            else:
                colors = np.asarray(pcd.colors) if pcd.has_colors() else None
                normals = np.asarray(pcd.normals) if pcd.has_normals() else None

            # Return clean data without metadata wrapping
            builder = ArrowBlockBuilder()
            builder.add(
                {
                    "num_points": len(points),
                    "points": points.tolist(),  # Convert to list for Arrow
                    "colors": colors.tolist() if colors is not None else None,
                    "normals": normals.tolist() if normals is not None else None,
                    "downsampled": downsampled,
                    "original_num_points": num_points if downsampled else None,
                }
            )
            yield builder.build()

        except Exception as e:
            raise DataSourceError(f"Failed to read PCD file {path}: {e}") from e

    def _read_ply(self, path: str) -> Iterator[Block]:
        """Read PLY file.

        Args:
            path: Path to PLY file

        Yields:
            Block object with point cloud data

        Raises:
            DataSourceError: If reading fails
        """
        try:
            import open3d as o3d  # http://www.open3d.org/
        except ImportError:
            raise DataSourceError(
                "open3d not installed. Install with: pip install open3d"
            ) from None

        try:
            pcd = o3d.io.read_point_cloud(path)
            if pcd is None:
                raise DataSourceError(f"Failed to read PLY file: {path}")
            
            points = np.asarray(pcd.points)
            if len(points) == 0:
                # Empty point cloud - return empty block
                builder = ArrowBlockBuilder()
                builder.add({"num_points": 0, "points": []})
                yield builder.build()
                return

            num_points = len(points)
            downsampled = False

            if num_points > self.max_points and self.downsample_large:
                logger.warning(
                    f"Point cloud {path} has {num_points} points, "
                    f"downsampling to {self.max_points} points"
                )
                # Downsample for large point clouds
                indices = np.random.choice(num_points, self.max_points, replace=False)
                points = points[indices]
                colors = (
                    np.asarray(pcd.colors)[indices] if pcd.has_colors() else None
                )
                downsampled = True
            else:
                colors = np.asarray(pcd.colors) if pcd.has_colors() else None

            # Return clean data without metadata wrapping
            builder = ArrowBlockBuilder()
            builder.add(
                {
                    "num_points": len(points),
                    "points": points.tolist(),
                    "colors": colors.tolist() if colors is not None else None,
                    "downsampled": downsampled,
                    "original_num_points": num_points if downsampled else None,
                }
            )
            yield builder.build()

        except Exception as e:
            raise DataSourceError(f"Failed to read PLY file {path}: {e}") from e

    def _read_las(self, path: str) -> Iterator[Block]:
        """Read LAS/LAZ file.

        Args:
            path: Path to LAS/LAZ file

        Yields:
            Block object with LiDAR data

        Raises:
            DataSourceError: If reading fails
        """
        try:
            import laspy  # https://laspy.readthedocs.io/
        except ImportError:
            raise DataSourceError(
                "laspy not installed. Install with: pip install laspy"
            ) from None

        try:
            las = laspy.read(path)
            if las is None:
                raise DataSourceError(f"Failed to read LAS file: {path}")

            # Check point cloud size to avoid OOM
            num_points = len(las.points)
            if num_points == 0:
                # Empty LAS file - return empty block
                builder = ArrowBlockBuilder()
                builder.add({"num_points": 0, "points": []})
                yield builder.build()
                return

            downsampled = False

            if num_points > self.max_points and self.downsample_large:
                logger.warning(
                    f"LAS file {path} has {num_points} points, "
                    f"downsampling to {self.max_points} points"
                )
                # Downsample for large point clouds
                indices = np.random.choice(num_points, self.max_points, replace=False)
                x = las.x[indices].tolist()
                y = las.y[indices].tolist()
                z = las.z[indices].tolist()
                intensity = (
                    las.intensity[indices].tolist()
                    if hasattr(las, "intensity") and las.intensity is not None
                    else None
                )
                downsampled = True
            else:
                x = las.x.tolist()
                y = las.y.tolist()
                z = las.z.tolist()
                intensity = (
                    las.intensity.tolist() if hasattr(las, "intensity") and las.intensity is not None else None
                )

            # Return clean data without metadata wrapping
            builder = ArrowBlockBuilder()
            builder.add(
                {
                    "num_points": len(x),
                    "x": x,
                    "y": y,
                    "z": z,
                    "intensity": intensity,
                    "downsampled": downsampled,
                    "original_num_points": num_points if downsampled else None,
                }
            )
            yield builder.build()

        except Exception as e:
            raise DataSourceError(f"Failed to read LAS file {path}: {e}") from e


def test() -> None:
    """Test point cloud datasource with example data."""
    from pathlib import Path
    
    # Initialize Ray if not already initialized
    try:
        import ray
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
    except Exception:
        pass
    
    # Test data directory
    test_data_dir = Path(__file__).parent.parent.parent / "examples" / "data"
    test_data_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Testing PointCloudDatasource")
    logger.info("Note: Point cloud test requires actual PCD/PLY/LAS files. Skipping if no test files found.")
    
    # Check for test point cloud files
    test_files = []
    for ext in [".pcd", ".ply", ".las", ".laz"]:
        test_files.extend(list(test_data_dir.glob(f"*{ext}")))
    
    if not test_files:
        logger.warning("Point cloud test skipped: No PCD/PLY/LAS files found in examples/data/")
        logger.info("To test point cloud, place test files (.pcd, .ply, .las, .laz) in examples/data/")
        return
    
    try:
        # Test point cloud datasource
        datasource = PointCloudDatasource(paths=str(test_files[0]))
        dataset = ray.data.read_datasource(datasource)
        
        # Verify dataset
        count = 0
        for batch in dataset.iter_batches(batch_size=10, prefetch_batches=0):
            count += len(batch) if hasattr(batch, '__len__') else 1
        
        logger.info(f"PointCloudDatasource test passed: loaded {count} point clouds")
    except Exception as e:
        logger.error(f"PointCloudDatasource test failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    test()
