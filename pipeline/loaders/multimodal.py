"""Multimodal data loader using Ray Data.

Handles loading video, text, and sensor data from various sources.
Uses Ray Data for parallel data loading and format detection.
See: https://docs.ray.io/en/latest/data/data.html
"""

import logging
from typing import Any, Dict, List

import ray  # https://docs.ray.io/
from ray.data import Dataset  # https://docs.ray.io/en/latest/data/data.html

from pipeline.config import PipelineConfig
from pipeline.config.ray_data import RayDataConfig
from pipeline.datasources.archive import ArchiveDatasource
from pipeline.datasources.binary import BinaryDatasource
from pipeline.datasources.calibration import CalibrationDatasource
from pipeline.datasources.hdf5 import HDF5Datasource
from pipeline.datasources.mcap import read_mcap
from pipeline.datasources.msgpack import MessagePackDatasource
from pipeline.datasources.pointcloud import PointCloudDatasource
from pipeline.datasources.protobuf import ProtobufDatasource
from pipeline.datasources.rosbag import ROSBagDatasource
from pipeline.datasources.ros2bag import ROS2BagDatasource
from pipeline.datasources.urdf import URDFDatasource
from pipeline.datasources.velodyne import VelodyneDatasource
from pipeline.datasources.yaml_config import YAMLConfigDatasource
from pipeline.loaders.formats import DataFormat, detect_format

logger = logging.getLogger(__name__)


# Helper functions to avoid lambda serialization issues
def _add_video_metadata(batch):
    """Add video metadata to batch."""
    from pipeline.utils.batch_processing import add_metadata_batch
    return add_metadata_batch(batch, "video", "video")


def _add_text_metadata(batch):
    """Add text metadata to batch."""
    from pipeline.utils.batch_processing import add_metadata_batch
    return add_metadata_batch(batch, "text", "text")


def _add_sensor_metadata(batch):
    """Add sensor metadata to batch."""
    from pipeline.utils.batch_processing import add_metadata_batch
    return add_metadata_batch(batch, "sensor", "sensor")


def _add_numpy_metadata(batch):
    """Add numpy metadata to batch."""
    from pipeline.utils.batch_processing import add_metadata_batch
    return add_metadata_batch(batch, "numpy", "array")


def _add_image_metadata(batch):
    """Add image metadata to batch."""
    from pipeline.utils.batch_processing import add_metadata_batch
    return add_metadata_batch(batch, "image", "image")


def _add_audio_metadata(batch):
    """Add audio metadata to batch."""
    from pipeline.utils.batch_processing import add_metadata_batch
    return add_metadata_batch(batch, "audio", "audio")


def _add_tfrecord_metadata(batch):
    """Add tfrecord metadata to batch."""
    from pipeline.utils.batch_processing import add_metadata_batch
    return add_metadata_batch(batch, "tfrecord", "ml_training")


def _add_sqlite_metadata(batch):
    """Add sqlite metadata to batch."""
    from pipeline.utils.batch_processing import add_metadata_batch
    return add_metadata_batch(batch, "sqlite", "database")


def _make_add_json_metadata(format_val):
    """Create function to add JSON metadata."""
    def _add_json_metadata(batch):
        from pipeline.utils.batch_processing import add_metadata_batch
        return add_metadata_batch(batch, format_val, "text")
    return _add_json_metadata


def _add_csv_metadata(batch):
    """Add CSV metadata to batch."""
    from pipeline.utils.batch_processing import add_metadata_batch
    return add_metadata_batch(batch, "csv", "sensor")


class MultimodalLoader:
    """Loader for multimodal datasets (video, text, sensor data).

    Handles format detection and parallel data loading using Ray Data.
    """

    def __init__(self, config: PipelineConfig):
        """Initialize the multimodal loader.

        Args:
            config: Pipeline configuration
        """
        self.config = config
        self._format_handlers = self._build_format_handlers()

    def load(self, input_paths: List[str]) -> Dataset:
        """Load data from multiple sources.

        Supports 20+ robotics data formats including videos, MCAP, ROS bags,
        HDF5, point clouds, NumPy arrays, and more.

        Args:
            input_paths: List of input paths (S3, local, etc.)

        Returns:
            Ray Dataset containing loaded data
        """
        datasets = []

        for path in input_paths:
            if not path or not isinstance(path, str):
                logger.warning("Invalid path %s, skipping", path)
                continue

            # Validate path (but don't sanitize S3/cloud paths as they're not local file paths)
            try:
                from pipeline.utils.validation.input_validation import InputValidator

                validator = InputValidator(strict=True)
                is_valid, error_msg = validator.validate_path(path)
                if not is_valid:
                    logger.error(f"Invalid path {path}: {error_msg}")
                    raise ValueError(f"Invalid path: {error_msg}")
                # Only sanitize local paths, not S3/cloud URLs
                if not path.startswith(("s3://", "gs://", "hdfs://", "abfss://", "http://", "https://")):
                    path = validator.sanitize_path(path)
            except ImportError:
                logger.debug("Input validation not available, skipping path validation")

            format_type = self._detect_format_safe(path)
            
            # Special handling for Omniverse USD files
            if format_type == DataFormat.USD:
                from pipeline.integrations.omniverse import OmniverseLoader
                
                loader = OmniverseLoader(omniverse_path=path, include_metadata=True)
                items = loader.load_usd_scene(path)
                if items:
                    dataset = ray.data.from_items(items)  # Use global ray import
                    datasets.append(dataset)
                    continue
            
            handler = self._format_handlers.get(format_type, self._load_generic)
            logger.info(
                "Loading %s data from %s via handler %s",
                format_type.value,
                path,
                handler.__name__,
            )

            try:
                # Retry cloud storage operations
                if path.startswith(("s3://", "gs://", "hdfs://", "abfss://")):
                    from pipeline.utils.execution.retry import retry_with_exponential_backoff

                    dataset = retry_with_exponential_backoff(
                        lambda: handler(path),
                        max_retries=3,
                        initial_delay=1.0,
                    )
                else:
                    dataset = handler(path)
            except Exception as e:
                logger.error("Failed to load %s with handler %s: %s", path, handler.__name__, e, exc_info=True)
                raise

            datasets.append(dataset)

        # Detect corruption in loaded datasets
        # Use streaming-friendly corruption detection
        try:
            from pipeline.utils.validation.corruption_detection import detect_corruption_batch

            # Add corruption detection as a stage with proper batch format
            for i, dataset in enumerate(datasets):
                from pipeline.utils.constants import _CORRUPTION_BATCH_SIZE

                datasets[i] = dataset.map_batches(
                    detect_corruption_batch, 
                    batch_size=_CORRUPTION_BATCH_SIZE,
                    batch_format="pandas",  # Specify batch format
                )
        except Exception as e:
            logger.warning(f"Failed to add corruption detection: {e}")

        # Union all datasets
        if not datasets:
            logger.warning("No datasets loaded, returning empty dataset")
            return ray.data.from_items([])

        if len(datasets) == 1:
            return datasets[0]

        # Union multiple datasets
        # Note: ray.data.union() requires at least 2 datasets
        return ray.data.union(*datasets)

    def _load_video(self, path: str) -> Dataset:
        """Load video data.

        Args:
            path: Path to video files

        Returns:
            Ray Dataset with video metadata
        """
        # Use Ray Data to read file paths
        dataset = ray.data.read_binary_files(
            paths=path,
            include_paths=True,
        )

        batch_size = self.config.batch_size
        map_kwargs = RayDataConfig.get_map_batches_kwargs(batch_size=batch_size)
        map_kwargs["batch_format"] = "pandas"  # Specify batch format

        return dataset.map_batches(
            _add_video_metadata,
            **map_kwargs,
        )

    def _load_text(self, path: str) -> Dataset:
        """Load text data.

        Args:
            path: Path to text files

        Returns:
            Ray Dataset with text data
        """
        # Try JSONL first, then text files
        if path.endswith(".jsonl"):
            dataset = ray.data.read_json(path)
        else:
            dataset = ray.data.read_text(path)

        batch_size = self.config.batch_size
        map_kwargs = RayDataConfig.get_map_batches_kwargs(batch_size=batch_size)
        map_kwargs["batch_format"] = "pandas"  # Specify batch format

        return dataset.map_batches(
            _add_text_metadata,
            **map_kwargs,
        )

    def _load_parquet(self, path: str) -> Dataset:
        """Load Parquet data with optimized settings.

        Supports column pruning (projection pushdown) for efficiency.

        Args:
            path: Path to Parquet files

        Returns:
            Ray Dataset
        """
        read_kwargs = RayDataConfig.get_read_kwargs(
            override_num_blocks=self.config.ray_data_config.get("override_num_blocks"),
            num_cpus=self.config.ray_data_config.get("read_num_cpus", 1.0),
        )

        return ray.data.read_parquet(path, **read_kwargs)

    def _load_sensor(self, path: str) -> Dataset:
        """Load sensor data.

        Args:
            path: Path to sensor data files

        Returns:
            Ray Dataset with sensor data
        """
        # Sensor data typically in Parquet or CSV
        # Use path.lower() for case-insensitive matching
        path_lower = path.lower()
        if path_lower.endswith(".parquet"):
            read_kwargs = RayDataConfig.get_read_kwargs(
                override_num_blocks=self.config.ray_data_config.get("override_num_blocks"),
                num_cpus=self.config.ray_data_config.get("read_num_cpus", 1.0),
            )
            dataset = ray.data.read_parquet(path, **read_kwargs)
        else:
            # Default to CSV for sensor data
            read_kwargs = RayDataConfig.get_read_kwargs(
                override_num_blocks=self.config.ray_data_config.get("override_num_blocks"),
                num_cpus=self.config.ray_data_config.get("read_num_cpus", 1.0),
            )
            dataset = ray.data.read_csv(path, **read_kwargs)

        batch_size = self.config.batch_size
        map_kwargs = RayDataConfig.get_map_batches_kwargs(batch_size=batch_size)
        map_kwargs["batch_format"] = "pandas"  # Specify batch format

        return dataset.map_batches(
            _add_sensor_metadata,
            **map_kwargs,
        )

    def _load_mcap(self, path: str) -> Dataset:
        """Load MCAP files using Ray Data's native MCAP support.

        Uses ray.data.read_mcap() which provides predicate pushdown
        optimization for efficient filtering by topics, time ranges, etc.

        Args:
            path: Path to MCAP files

        Returns:
            Ray Dataset
        """
        # Get optimized read kwargs
        read_kwargs = RayDataConfig.get_read_kwargs(
            override_num_blocks=self.config.ray_data_config.get("override_num_blocks"),
            num_cpus=self.config.ray_data_config.get("read_num_cpus", 1.0),
            concurrency=self.config.ray_data_config.get("read_concurrency"),
        )

        # Use Ray Data's native MCAP reader with optimized settings
        # See: https://docs.ray.io/en/latest/data/api/doc/ray.data.read_mcap.html
        return read_mcap(paths=path, include_paths=True, **read_kwargs)

    def _load_rosbag(self, path: str) -> Dataset:
        """Load ROS bag files using custom datasource.

        Args:
            path: Path to ROS bag files (str or List[str])

        Returns:
            Ray Dataset
        """
        # FileBasedDatasource accepts Union[str, List[str]], no need to wrap
        datasource = ROSBagDatasource(paths=path)
        return ray.data.read_datasource(datasource)

    def _load_ros2bag(self, path: str) -> Dataset:
        """Load ROS2 bag files using custom datasource.

        Args:
            path: Path to ROS2 bag files (str or List[str])

        Returns:
            Ray Dataset
        """
        # FileBasedDatasource accepts Union[str, List[str]], no need to wrap
        datasource = ROS2BagDatasource(paths=path)
        return ray.data.read_datasource(datasource)

    def _load_hdf5(self, path: str) -> Dataset:
        """Load HDF5 files using custom datasource.

        Args:
            path: Path to HDF5 files (str or List[str])

        Returns:
            Ray Dataset
        """
        # FileBasedDatasource accepts Union[str, List[str]], no need to wrap
        datasource = HDF5Datasource(paths=path)
        return ray.data.read_datasource(datasource)

    def _load_pointcloud(self, path: str) -> Dataset:
        """Load point cloud files using custom datasource.

        Args:
            path: Path to point cloud files (str or List[str])

        Returns:
            Ray Dataset
        """
        # FileBasedDatasource accepts Union[str, List[str]], no need to wrap
        datasource = PointCloudDatasource(paths=path)
        return ray.data.read_datasource(datasource)

    def _load_numpy_native(self, path: str) -> Dataset:
        """Load NumPy array files using Ray Data's native read_numpy().

        Uses ray.data.read_numpy() which provides optimized NumPy reading.

        Args:
            path: Path to NumPy files

        Returns:
            Ray Dataset
        """
        read_kwargs = RayDataConfig.get_read_kwargs(
            override_num_blocks=self.config.ray_data_config.get("override_num_blocks"),
        )

        # Use Ray Data's native NumPy reader
        # See: https://docs.ray.io/en/latest/data/api/doc/ray.data.read_numpy.html
        dataset = ray.data.read_numpy(paths=path, **read_kwargs)

        batch_size = self.config.batch_size
        map_kwargs = RayDataConfig.get_map_batches_kwargs(batch_size=batch_size)
        map_kwargs["batch_format"] = "pandas"  # Specify batch format

        return dataset.map_batches(
            _add_numpy_metadata,
            **map_kwargs,
        )

    def _load_image(self, path: str) -> Dataset:
        """Load image files.

        Args:
            path: Path to image files

        Returns:
            Ray Dataset
        """
        # Use Ray Data's native image reading
        try:
            dataset = ray.data.read_images(paths=path)
            batch_size = self.config.batch_size
            map_kwargs = RayDataConfig.get_map_batches_kwargs(batch_size=batch_size)
            map_kwargs["batch_format"] = "pandas"  # Specify batch format
            return dataset.map_batches(
                _add_image_metadata,
                **map_kwargs
            )
        except AttributeError:
            dataset = ray.data.read_binary_files(paths=path, include_paths=True)

            batch_size = self.config.batch_size
            map_kwargs = RayDataConfig.get_map_batches_kwargs(batch_size=batch_size)
            map_kwargs["batch_format"] = "pandas"  # Specify batch format

            return dataset.map_batches(
                _add_image_metadata,
                **map_kwargs,
            )

    def _load_audio_native(self, path: str) -> Dataset:
        """Load audio files using Ray Data's native read_audio().

        Uses ray.data.read_audio() which provides optimized audio reading.

        Args:
            path: Path to audio files

        Returns:
            Ray Dataset
        """
        read_kwargs = RayDataConfig.get_read_kwargs(
            override_num_blocks=self.config.ray_data_config.get("override_num_blocks"),
        )

        dataset = ray.data.read_audio(paths=path, **read_kwargs)

        batch_size = self.config.batch_size
        map_kwargs = RayDataConfig.get_map_batches_kwargs(batch_size=batch_size)
        map_kwargs["batch_format"] = "pandas"  # Specify batch format

        return dataset.map_batches(
            _add_audio_metadata,
            **map_kwargs,
        )

    def _load_json(self, path: str) -> Dataset:
        """Load JSON/JSONL files.

        Args:
            path: Path to JSON files (supports both .json and .jsonl)

        Returns:
            Ray Dataset
        """
        dataset = ray.data.read_json(path)

        batch_size = self.config.batch_size
        map_kwargs = RayDataConfig.get_map_batches_kwargs(batch_size=batch_size)
        map_kwargs["batch_format"] = "pandas"  # Specify batch format

        format_val = "jsonl" if path.endswith(".jsonl") else "json"
        return dataset.map_batches(
            _make_add_json_metadata(format_val),
            **map_kwargs,
        )

    def _load_csv(self, path: str) -> Dataset:
        """Load CSV files.

        Args:
            path: Path to CSV files

        Returns:
            Ray Dataset
        """
        dataset = ray.data.read_csv(path)

        batch_size = self.config.batch_size
        map_kwargs = RayDataConfig.get_map_batches_kwargs(batch_size=batch_size)
        map_kwargs["batch_format"] = "pandas"  # Specify batch format

        return dataset.map_batches(
            _add_csv_metadata,
            **map_kwargs,
        )

    def _load_tfrecord_native(self, path: str) -> Dataset:
        """Load TFRecord files using Ray Data's native read_tfrecords().

        Uses ray.data.read_tfrecords() which provides optimized TFRecord reading.

        Args:
            path: Path to TFRecord files

        Returns:
            Ray Dataset
        """
        read_kwargs = RayDataConfig.get_read_kwargs(
            override_num_blocks=self.config.ray_data_config.get("override_num_blocks"),
        )

        dataset = ray.data.read_tfrecords(paths=path, **read_kwargs)

        batch_size = self.config.batch_size
        map_kwargs = RayDataConfig.get_map_batches_kwargs(batch_size=batch_size)
        map_kwargs["batch_format"] = "pandas"  # Specify batch format

        return dataset.map_batches(
            _add_tfrecord_metadata,
            **map_kwargs,
        )

    def _load_urdf(self, path: str) -> Dataset:
        """Load URDF/SDF files using custom datasource.

        Args:
            path: Path to URDF/SDF files

        Returns:
            Ray Dataset
        """
        datasource = URDFDatasource(paths=path)
        read_kwargs = RayDataConfig.get_read_kwargs(
            override_num_blocks=self.config.ray_data_config.get("override_num_blocks"),
        )
        return ray.data.read_datasource(datasource, **read_kwargs)

    def _load_sqlite_native(self, path: str) -> Dataset:
        """Load SQLite databases using Ray Data's native read_sql().

        Uses ray.data.read_sql() with SQLite connection string.

        Args:
            path: Path to SQLite database file

        Returns:
            Ray Dataset
        """
        # SQLite connection string format: sqlite:///path/to/database.db
        if not path.startswith("sqlite:///"):
            connection_string = f"sqlite:///{path}"
        else:
            connection_string = path

        read_kwargs = RayDataConfig.get_read_kwargs(
            override_num_blocks=self.config.ray_data_config.get("override_num_blocks"),
        )

        dataset = ray.data.read_sql(
            connection_string=connection_string,
            query="SELECT * FROM sqlite_master WHERE type='table'",
            **read_kwargs,
        )

        batch_size = self.config.batch_size
        map_kwargs = RayDataConfig.get_map_batches_kwargs(batch_size=batch_size)
        map_kwargs["batch_format"] = "pandas"  # Specify batch format

        return dataset.map_batches(
            _add_sqlite_metadata,
            **map_kwargs,
        )

    def _load_archive(self, path: str) -> Dataset:
        """Load archive files using custom datasource.

        Args:
            path: Path to archive files

        Returns:
            Ray Dataset
        """
        datasource = ArchiveDatasource(paths=path, extract=False)
        read_kwargs = RayDataConfig.get_read_kwargs(
            override_num_blocks=self.config.ray_data_config.get("override_num_blocks"),
        )
        return ray.data.read_datasource(datasource, **read_kwargs)

    def _load_protobuf(self, path: str) -> Dataset:
        """Load protobuf files using custom datasource.

        Args:
            path: Path to protobuf files

        Returns:
            Ray Dataset
        """
        datasource = ProtobufDatasource(paths=path)
        read_kwargs = RayDataConfig.get_read_kwargs(
            override_num_blocks=self.config.ray_data_config.get("override_num_blocks"),
        )
        return ray.data.read_datasource(datasource, **read_kwargs)

    def _load_msgpack(self, path: str) -> Dataset:
        """Load MessagePack files using custom datasource.

        Args:
            path: Path to MessagePack files

        Returns:
            Ray Dataset
        """
        datasource = MessagePackDatasource(paths=path)
        read_kwargs = RayDataConfig.get_read_kwargs(
            override_num_blocks=self.config.ray_data_config.get("override_num_blocks"),
        )
        return ray.data.read_datasource(datasource, **read_kwargs)

    def _load_yaml_config(self, path: str) -> Dataset:
        """Load YAML config files using custom datasource.

        Args:
            path: Path to YAML config files

        Returns:
            Ray Dataset
        """
        datasource = YAMLConfigDatasource(paths=path)
        read_kwargs = RayDataConfig.get_read_kwargs(
            override_num_blocks=self.config.ray_data_config.get("override_num_blocks"),
        )
        return ray.data.read_datasource(datasource, **read_kwargs)

    def _load_binary(self, path: str) -> Dataset:
        """Load binary files using custom datasource.

        Args:
            path: Path to binary files

        Returns:
            Ray Dataset
        """
        datasource = BinaryDatasource(paths=path)
        read_kwargs = RayDataConfig.get_read_kwargs(
            override_num_blocks=self.config.ray_data_config.get("override_num_blocks"),
        )
        return ray.data.read_datasource(datasource, **read_kwargs)

    def _load_velodyne(self, path: str) -> Dataset:
        """Load Velodyne LIDAR files using custom datasource.

        Args:
            path: Path to Velodyne files

        Returns:
            Ray Dataset
        """
        datasource = VelodyneDatasource(paths=path)
        read_kwargs = RayDataConfig.get_read_kwargs(
            override_num_blocks=self.config.ray_data_config.get("override_num_blocks"),
        )
        return ray.data.read_datasource(datasource, **read_kwargs)

    def _load_calibration(self, path: str) -> Dataset:
        """Load calibration files using custom datasource.

        Args:
            path: Path to calibration files

        Returns:
            Ray Dataset
        """
        datasource = CalibrationDatasource(paths=path)
        read_kwargs = RayDataConfig.get_read_kwargs(
            override_num_blocks=self.config.ray_data_config.get("override_num_blocks"),
        )
        return ray.data.read_datasource(datasource, **read_kwargs)

    def _load_generic(self, path: str) -> Dataset:
        """Generic loader for unknown formats."""
        return ray.data.read_binary_files(paths=path, include_paths=True)

    def _detect_format_safe(self, path: str) -> DataFormat:
        try:
            return detect_format(path)
        except Exception as e:
            logger.error("Error detecting format for %s: %s", path, e, exc_info=True)
            return DataFormat.SENSOR

    def _build_format_handlers(self) -> Dict[DataFormat, Any]:
        """Map each DataFormat to the appropriate loader."""
        return {
            DataFormat.VIDEO: self._load_video,
            DataFormat.TEXT: self._load_text,
            DataFormat.JSON: self._load_json,
            DataFormat.JSONL: self._load_json,
            DataFormat.PARQUET: self._load_parquet,
            DataFormat.CSV: self._load_csv,
            DataFormat.MCAP: self._load_mcap,
            DataFormat.ROSBAG: self._load_rosbag,
            DataFormat.ROS2BAG: self._load_ros2bag,
            DataFormat.HDF5: self._load_hdf5,
            DataFormat.PCD: self._load_pointcloud,
            DataFormat.PLY: self._load_pointcloud,
            DataFormat.LAS: self._load_pointcloud,
            DataFormat.NPY: self._load_numpy_native,
            DataFormat.NPZ: self._load_numpy_native,
            DataFormat.IMAGE: self._load_image,
            DataFormat.AUDIO: self._load_audio_native,
            DataFormat.TFRECORD: self._load_tfrecord_native,
            DataFormat.URDF: self._load_urdf,
            DataFormat.SDF: self._load_urdf,
            DataFormat.SQLITE: self._load_sqlite_native,
            DataFormat.ZIP: self._load_archive,
            DataFormat.TAR: self._load_archive,
            DataFormat.PROTOBUF: self._load_protobuf,
            DataFormat.MSGPACK: self._load_msgpack,
            DataFormat.YAML_CONFIG: self._load_yaml_config,
            DataFormat.BINARY: self._load_binary,
            DataFormat.VELODYNE: self._load_velodyne,
            DataFormat.CALIBRATION: self._load_calibration,
            DataFormat.SENSOR: self._load_sensor,
        }
