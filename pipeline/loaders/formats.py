"""Format detection and handling for multimodal data."""

from enum import Enum
from pathlib import Path


class DataFormat(Enum):
    """Supported data formats for robotics datasets."""

    # Video formats
    VIDEO = "video"
    # Text formats
    TEXT = "text"
    JSON = "json"
    JSONL = "jsonl"
    # Sensor data formats
    SENSOR = "sensor"
    MCAP = "mcap"
    ROSBAG = "rosbag"
    ROS2BAG = "ros2bag"
    HDF5 = "hdf5"
    # Point cloud formats
    POINTCLOUD = "pointcloud"
    PCD = "pcd"
    PLY = "ply"
    LAS = "las"
    # Array formats
    NUMPY = "numpy"
    NPY = "npy"
    NPZ = "npz"
    # Structured data
    PARQUET = "parquet"
    CSV = "csv"
    # Image formats
    IMAGE = "image"
    # Audio formats
    AUDIO = "audio"
    # ML training formats
    TFRECORD = "tfrecord"
    # Robot description formats
    URDF = "urdf"
    SDF = "sdf"
    # Database formats
    SQLITE = "sqlite"
    # Archive formats
    ZIP = "zip"
    TAR = "tar"
    # Serialization formats
    PROTOBUF = "protobuf"
    MSGPACK = "msgpack"
    # Configuration formats
    YAML_CONFIG = "yaml_config"
    # Binary formats
    BINARY = "binary"
    # LIDAR formats
    VELODYNE = "velodyne"
    # Calibration formats
    CALIBRATION = "calibration"
    # NVIDIA Omniverse formats
    USD = "usd"  # Universal Scene Description (USD, USDA, USDC)
    REPLICATOR = "replicator"  # Omniverse Replicator output
    # Unknown
    UNKNOWN = "unknown"


SUPPORTED_FORMATS = {
    # Video formats
    ".mp4": DataFormat.VIDEO,
    ".avi": DataFormat.VIDEO,
    ".mov": DataFormat.VIDEO,
    ".mkv": DataFormat.VIDEO,
    ".webm": DataFormat.VIDEO,
    # Text formats
    ".txt": DataFormat.TEXT,
    ".log": DataFormat.TEXT,
    ".json": DataFormat.JSON,
    ".jsonl": DataFormat.JSONL,
    # Sensor data formats
    ".mcap": DataFormat.MCAP,
    ".bag": DataFormat.ROSBAG,
    ".db3": DataFormat.ROS2BAG,  # ROS2 SQLite bag format
    ".h5": DataFormat.HDF5,
    ".hdf5": DataFormat.HDF5,
    # Point cloud formats
    ".pcd": DataFormat.PCD,
    ".ply": DataFormat.PLY,
    ".las": DataFormat.LAS,
    ".laz": DataFormat.LAS,
    # Array formats
    ".npy": DataFormat.NPY,
    ".npz": DataFormat.NPZ,
    # Structured data
    ".parquet": DataFormat.PARQUET,
    ".csv": DataFormat.CSV,
    # Image formats
    ".jpg": DataFormat.IMAGE,
    ".jpeg": DataFormat.IMAGE,
    ".png": DataFormat.IMAGE,
    ".tiff": DataFormat.IMAGE,
    ".tif": DataFormat.IMAGE,
    ".bmp": DataFormat.IMAGE,
    # Audio formats
    ".wav": DataFormat.AUDIO,
    ".flac": DataFormat.AUDIO,
    ".mp3": DataFormat.AUDIO,
    # ML training formats
    ".tfrecord": DataFormat.TFRECORD,
    # Robot description formats
    ".urdf": DataFormat.URDF,
    ".sdf": DataFormat.SDF,
    ".xacro": DataFormat.URDF,  # URDF macro files
    # Database formats
    ".db": DataFormat.SQLITE,
    ".sqlite": DataFormat.SQLITE,
    ".sqlite3": DataFormat.SQLITE,
    # Archive formats
    ".zip": DataFormat.ZIP,
    ".tar": DataFormat.TAR,
    ".tar.gz": DataFormat.TAR,
    ".tgz": DataFormat.TAR,
    ".tar.bz2": DataFormat.TAR,
    ".tbz2": DataFormat.TAR,
    # Serialization formats
    ".pb": DataFormat.PROTOBUF,
    ".protobuf": DataFormat.PROTOBUF,
    ".msgpack": DataFormat.MSGPACK,
    ".mpk": DataFormat.MSGPACK,
    # Configuration formats
    ".yaml": DataFormat.YAML_CONFIG,
    ".yml": DataFormat.YAML_CONFIG,
    # Binary formats
    ".bin": DataFormat.BINARY,
    ".dat": DataFormat.BINARY,
    # LIDAR formats
    ".pcap": DataFormat.VELODYNE,
    ".velodyne": DataFormat.VELODYNE,
    # NVIDIA Omniverse formats
    ".usd": DataFormat.USD,
    ".usda": DataFormat.USD,
    ".usdc": DataFormat.USD,
    # Calibration formats (detected by filename pattern)
}


def detect_format(file_path: str) -> DataFormat:
    """Detect data format from file path.

    Args:
        file_path: Path to the data file

    Returns:
        Detected data format
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix in SUPPORTED_FORMATS:
        return SUPPORTED_FORMATS[suffix]

    # Check for sensor data patterns
    if "sensor" in path.name.lower() or "imu" in path.name.lower():
        return DataFormat.SENSOR

    # Check for calibration files by name pattern
    if any(
        pattern in path.name.lower()
        for pattern in ["calibration", "calib", "intrinsic", "extrinsic"]
    ):
        return DataFormat.CALIBRATION

    # Check for Omniverse Replicator output directories
    if "replicator_output" in path.name.lower() or "replicator" in path.name.lower():
        return DataFormat.REPLICATOR

    return DataFormat.UNKNOWN


def is_video_format(file_path: str) -> bool:
    """Check if file is a video format.

    Args:
        file_path: Path to check

    Returns:
        True if video format
    """
    return detect_format(file_path) == DataFormat.VIDEO


def is_text_format(file_path: str) -> bool:
    """Check if file is a text format.

    Args:
        file_path: Path to check

    Returns:
        True if text format
    """
    return detect_format(file_path) == DataFormat.TEXT
