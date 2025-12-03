"""Reader registry for automatic detection and selection of data readers.

Supports both native Ray Data readers and custom datasources.
Provides automatic format detection and explicit reader specification.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional, Union

logger = logging.getLogger(__name__)


class ReaderRegistry:
    """Registry for data readers (native Ray Data and custom datasources).
    
    Maps file extensions and format names to appropriate readers.
    Supports automatic detection and explicit specification.
    """
    
    # Native Ray Data readers mapped by extension
    _native_readers: dict[str, Callable] = {}
    
    # Custom datasources mapped by extension
    _custom_readers: dict[str, Callable] = {}
    
    # Format names mapped to readers (for explicit specification)
    _format_readers: dict[str, Callable] = {}
    
    # Extension priority (for ambiguous extensions)
    _extension_priority: dict[str, list[str]] = {}
    
    @classmethod
    def _initialize_native_readers(cls) -> None:
        """Initialize native Ray Data readers."""
        if cls._native_readers:
            return  # Already initialized
        
        try:
            import ray.data
            
            # Parquet formats
            cls._native_readers[".parquet"] = ray.data.read_parquet
            cls._native_readers[".pq"] = ray.data.read_parquet
            
            # CSV formats
            cls._native_readers[".csv"] = ray.data.read_csv
            cls._native_readers[".tsv"] = ray.data.read_csv
            
            # JSON formats
            cls._native_readers[".json"] = ray.data.read_json
            cls._native_readers[".jsonl"] = ray.data.read_json
            
            # Text formats
            cls._native_readers[".txt"] = ray.data.read_text
            cls._native_readers[".log"] = ray.data.read_text
            
            # Image formats
            cls._native_readers[".jpg"] = ray.data.read_images
            cls._native_readers[".jpeg"] = ray.data.read_images
            cls._native_readers[".png"] = ray.data.read_images
            cls._native_readers[".tiff"] = ray.data.read_images
            cls._native_readers[".tif"] = ray.data.read_images
            cls._native_readers[".bmp"] = ray.data.read_images
            cls._native_readers[".gif"] = ray.data.read_images
            cls._native_readers[".webp"] = ray.data.read_images
            
            # Video formats
            cls._native_readers[".mp4"] = ray.data.read_videos
            cls._native_readers[".avi"] = ray.data.read_videos
            cls._native_readers[".mov"] = ray.data.read_videos
            cls._native_readers[".mkv"] = ray.data.read_videos
            cls._native_readers[".webm"] = ray.data.read_videos
            
            # Audio formats
            cls._native_readers[".wav"] = ray.data.read_audio
            cls._native_readers[".flac"] = ray.data.read_audio
            cls._native_readers[".mp3"] = ray.data.read_audio
            cls._native_readers[".ogg"] = ray.data.read_audio
            
            # NumPy formats
            cls._native_readers[".npy"] = ray.data.read_numpy
            cls._native_readers[".npz"] = ray.data.read_numpy
            
            # TFRecord formats
            cls._native_readers[".tfrecord"] = ray.data.read_tfrecords
            cls._native_readers[".tfrecords"] = ray.data.read_tfrecords
            
            # Binary formats (fallback)
            cls._native_readers[".bin"] = ray.data.read_binary_files
            cls._native_readers[".dat"] = ray.data.read_binary_files
            
            # Register format names
            cls._format_readers["parquet"] = ray.data.read_parquet
            cls._format_readers["csv"] = ray.data.read_csv
            cls._format_readers["json"] = ray.data.read_json
            cls._format_readers["jsonl"] = ray.data.read_json
            cls._format_readers["text"] = ray.data.read_text
            cls._format_readers["images"] = ray.data.read_images
            cls._format_readers["videos"] = ray.data.read_videos
            cls._format_readers["audio"] = ray.data.read_audio
            cls._format_readers["numpy"] = ray.data.read_numpy
            cls._format_readers["tfrecords"] = ray.data.read_tfrecords
            cls._format_readers["binary"] = ray.data.read_binary_files
            
            logger.debug("Initialized native Ray Data readers")
        except ImportError:
            logger.warning("Ray Data not available, native readers not registered")
    
    @classmethod
    def _initialize_custom_readers(cls) -> None:
        """Initialize custom datasource readers."""
        if cls._custom_readers:
            return  # Already initialized
        
        try:
            # MCAP format
            from pipeline.datasources.mcap import read_mcap
            cls._custom_readers[".mcap"] = read_mcap
            cls._format_readers["mcap"] = read_mcap
        except ImportError:
            logger.debug("MCAP datasource not available")
        
        try:
            import ray.data
            
            # ROS Bag formats
            from pipeline.datasources.rosbag import ROSBagDatasource
            cls._custom_readers[".bag"] = lambda paths, **kwargs: ray.data.read_datasource(ROSBagDatasource(paths=paths, **kwargs))
            cls._format_readers["rosbag"] = cls._custom_readers[".bag"]
        except ImportError:
            logger.debug("ROSBag datasource not available")
        
        try:
            import ray.data
            
            # ROS2 Bag formats
            from pipeline.datasources.ros2bag import ROS2BagDatasource
            cls._custom_readers[".db3"] = lambda paths, **kwargs: ray.data.read_datasource(ROS2BagDatasource(paths=paths, **kwargs))
            cls._format_readers["ros2bag"] = cls._custom_readers[".db3"]
        except ImportError:
            logger.debug("ROS2Bag datasource not available")
        
        try:
            import ray.data
            
            # HDF5 formats
            from pipeline.datasources.hdf5 import HDF5Datasource
            cls._custom_readers[".h5"] = lambda paths, **kwargs: ray.data.read_datasource(HDF5Datasource(paths=paths, **kwargs))
            cls._custom_readers[".hdf5"] = lambda paths, **kwargs: ray.data.read_datasource(HDF5Datasource(paths=paths, **kwargs))
            cls._format_readers["hdf5"] = cls._custom_readers[".h5"]
        except ImportError:
            logger.debug("HDF5 datasource not available")
        
        try:
            import ray.data
            
            # Point cloud formats
            from pipeline.datasources.pointcloud import PointCloudDatasource
            cls._custom_readers[".pcd"] = lambda paths, **kwargs: ray.data.read_datasource(PointCloudDatasource(paths=paths, **kwargs))
            cls._custom_readers[".ply"] = lambda paths, **kwargs: ray.data.read_datasource(PointCloudDatasource(paths=paths, **kwargs))
            cls._custom_readers[".las"] = lambda paths, **kwargs: ray.data.read_datasource(PointCloudDatasource(paths=paths, **kwargs))
            cls._custom_readers[".laz"] = lambda paths, **kwargs: ray.data.read_datasource(PointCloudDatasource(paths=paths, **kwargs))
            cls._format_readers["pointcloud"] = cls._custom_readers[".pcd"]
            cls._format_readers["pcd"] = cls._custom_readers[".pcd"]
            cls._format_readers["ply"] = cls._custom_readers[".ply"]
            cls._format_readers["las"] = cls._custom_readers[".las"]
        except ImportError:
            logger.debug("PointCloud datasource not available")
        
        try:
            import ray.data
            
            # URDF/SDF formats
            from pipeline.datasources.urdf import URDFDatasource
            cls._custom_readers[".urdf"] = lambda paths, **kwargs: ray.data.read_datasource(URDFDatasource(paths=paths, **kwargs))
            cls._custom_readers[".sdf"] = lambda paths, **kwargs: ray.data.read_datasource(URDFDatasource(paths=paths, **kwargs))
            cls._custom_readers[".xacro"] = lambda paths, **kwargs: ray.data.read_datasource(URDFDatasource(paths=paths, **kwargs))
            cls._format_readers["urdf"] = cls._custom_readers[".urdf"]
            cls._format_readers["sdf"] = cls._custom_readers[".sdf"]
        except ImportError:
            logger.debug("URDF datasource not available")
        
        try:
            import ray.data
            
            # Archive formats
            from pipeline.datasources.archive import ArchiveDatasource
            cls._custom_readers[".zip"] = lambda paths, **kwargs: ray.data.read_datasource(ArchiveDatasource(paths=paths, **kwargs))
            cls._custom_readers[".tar"] = lambda paths, **kwargs: ray.data.read_datasource(ArchiveDatasource(paths=paths, **kwargs))
            cls._custom_readers[".tar.gz"] = lambda paths, **kwargs: ray.data.read_datasource(ArchiveDatasource(paths=paths, **kwargs))
            cls._custom_readers[".tgz"] = lambda paths, **kwargs: ray.data.read_datasource(ArchiveDatasource(paths=paths, **kwargs))
            cls._format_readers["archive"] = cls._custom_readers[".zip"]
            cls._format_readers["zip"] = cls._custom_readers[".zip"]
            cls._format_readers["tar"] = cls._custom_readers[".tar"]
        except ImportError:
            logger.debug("Archive datasource not available")
        
        try:
            import ray.data
            
            # Protobuf formats
            from pipeline.datasources.protobuf import ProtobufDatasource
            cls._custom_readers[".pb"] = lambda paths, **kwargs: ray.data.read_datasource(ProtobufDatasource(paths=paths, **kwargs))
            cls._custom_readers[".protobuf"] = lambda paths, **kwargs: ray.data.read_datasource(ProtobufDatasource(paths=paths, **kwargs))
            cls._format_readers["protobuf"] = cls._custom_readers[".pb"]
        except ImportError:
            logger.debug("Protobuf datasource not available")
        
        try:
            import ray.data
            
            # MessagePack formats
            from pipeline.datasources.msgpack import MessagePackDatasource
            cls._custom_readers[".msgpack"] = lambda paths, **kwargs: ray.data.read_datasource(MessagePackDatasource(paths=paths, **kwargs))
            cls._custom_readers[".mpk"] = lambda paths, **kwargs: ray.data.read_datasource(MessagePackDatasource(paths=paths, **kwargs))
            cls._format_readers["msgpack"] = cls._custom_readers[".msgpack"]
        except ImportError:
            logger.debug("MessagePack datasource not available")
        
        try:
            import ray.data
            
            # YAML Config formats
            from pipeline.datasources.yaml_config import YAMLConfigDatasource
            cls._custom_readers[".yaml"] = lambda paths, **kwargs: ray.data.read_datasource(YAMLConfigDatasource(paths=paths, **kwargs))
            cls._custom_readers[".yml"] = lambda paths, **kwargs: ray.data.read_datasource(YAMLConfigDatasource(paths=paths, **kwargs))
            cls._format_readers["yaml"] = cls._custom_readers[".yaml"]
            cls._format_readers["yaml_config"] = cls._custom_readers[".yaml"]
        except ImportError:
            logger.debug("YAML Config datasource not available")
        
        try:
            import ray.data
            
            # Binary datasource (for custom binary formats)
            from pipeline.datasources.binary import BinaryDatasource
            cls._format_readers["binary"] = lambda paths, **kwargs: ray.data.read_datasource(BinaryDatasource(paths=paths, **kwargs))
        except ImportError:
            logger.debug("Binary datasource not available")
        
        try:
            import ray.data
            
            # Velodyne formats
            from pipeline.datasources.velodyne import VelodyneDatasource
            cls._custom_readers[".pcap"] = lambda paths, **kwargs: ray.data.read_datasource(VelodyneDatasource(paths=paths, **kwargs))
            cls._format_readers["velodyne"] = cls._custom_readers[".pcap"]
        except ImportError:
            logger.debug("Velodyne datasource not available")
        
        try:
            import ray.data
            
            # Calibration formats (detected by filename pattern)
            from pipeline.datasources.calibration import CalibrationDatasource
            cls._format_readers["calibration"] = lambda paths, **kwargs: ray.data.read_datasource(CalibrationDatasource(paths=paths, **kwargs))
        except ImportError:
            logger.debug("Calibration datasource not available")
        
        try:
            import ray.data
            
            # GR00T format
            from pipeline.datasources.groot import GR00TDatasource
            cls._format_readers["groot"] = lambda paths, **kwargs: ray.data.read_datasource(GR00TDatasource(paths=paths, **kwargs))
        except ImportError:
            logger.debug("GR00T datasource not available")
        
        logger.debug("Initialized custom datasource readers")
    
    @classmethod
    def initialize(cls) -> None:
        """Initialize all readers (native and custom)."""
        cls._initialize_native_readers()
        cls._initialize_custom_readers()
    
    @classmethod
    def detect_reader(
        cls,
        path: str,
        format: Optional[str] = None,
    ) -> Optional[Callable]:
        """Detect appropriate reader for a path.
        
        Args:
            path: File or directory path
            format: Optional explicit format name (overrides auto-detection)
        
        Returns:
            Reader function or None if not found
        """
        cls.initialize()
        
        # Explicit format specification takes precedence
        if format:
            format_lower = format.lower()
            if format_lower in cls._format_readers:
                reader = cls._format_readers[format_lower]
                logger.debug(f"Using explicit format reader: {format}")
                return reader
            else:
                logger.warning(f"Unknown format '{format}', falling back to auto-detection")
        
        # Auto-detect from extension
        path_lower = path.lower()
        
        # Check for directory patterns first
        if "/parquet/" in path_lower or path_lower.endswith("/parquet"):
            return cls._native_readers.get(".parquet")
        if "/json/" in path_lower or path_lower.endswith("/json"):
            return cls._native_readers.get(".json")
        if "/csv/" in path_lower or path_lower.endswith("/csv"):
            return cls._native_readers.get(".csv")
        if "/images/" in path_lower or path_lower.endswith("/images"):
            return cls._native_readers.get(".jpg")
        if "/videos/" in path_lower or path_lower.endswith("/videos"):
            return cls._native_readers.get(".mp4")
        
        # Check file extensions (try longest first for multi-part extensions)
        extensions = [
            ".tar.gz", ".tar.bz2", ".tar.bz", ".tar.xz",
            ".tgz", ".tbz2", ".tbz",
        ]
        for ext in extensions:
            if path_lower.endswith(ext):
                reader = cls._custom_readers.get(ext) or cls._native_readers.get(ext)
                if reader:
                    logger.debug(f"Detected reader from extension: {ext}")
                    return reader
        
        # Check single-part extensions
        from pathlib import Path
        path_obj = Path(path)
        suffix = path_obj.suffix.lower()
        
        # Try custom readers first (they override native for same extension)
        if suffix in cls._custom_readers:
            logger.debug(f"Detected custom reader from extension: {suffix}")
            return cls._custom_readers[suffix]
        
        # Try native readers
        if suffix in cls._native_readers:
            logger.debug(f"Detected native reader from extension: {suffix}")
            return cls._native_readers[suffix]
        
        # Fallback: try binary files reader
        logger.debug(f"No specific reader found for {suffix}, using binary files reader")
        try:
            import ray.data
            return ray.data.read_binary_files
        except ImportError:
            return None
    
    @classmethod
    def get_reader(cls, format: str) -> Optional[Callable]:
        """Get reader by format name.
        
        Args:
            format: Format name (e.g., "parquet", "mcap", "rosbag")
        
        Returns:
            Reader function or None if not found
        """
        cls.initialize()
        return cls._format_readers.get(format.lower())
    
    @classmethod
    def list_formats(cls) -> dict[str, list[str]]:
        """List all supported formats.
        
        Returns:
            Dictionary with 'native' and 'custom' format lists
        """
        cls.initialize()
        
        native_formats = list(cls._format_readers.keys())
        custom_formats = [
            fmt for fmt in cls._format_readers.keys()
            if fmt not in ["parquet", "csv", "json", "jsonl", "text", "images", 
                          "videos", "audio", "numpy", "tfrecords", "binary"]
        ]
        
        return {
            "native": sorted(native_formats),
            "custom": sorted(custom_formats),
        }
    
    @classmethod
    def register_custom_reader(
        cls,
        extension: str,
        reader: Callable,
        format_name: Optional[str] = None,
    ) -> None:
        """Register a custom reader.
        
        Args:
            extension: File extension (e.g., ".custom")
            reader: Reader function
            format_name: Optional format name for explicit specification
        """
        cls.initialize()
        cls._custom_readers[extension.lower()] = reader
        if format_name:
            cls._format_readers[format_name.lower()] = reader
        logger.info(f"Registered custom reader for extension {extension}")


# Convenience functions
def detect_reader(path: str, format: Optional[str] = None) -> Optional[Callable]:
    """Detect appropriate reader for a path.
    
    Args:
        path: File or directory path
        format: Optional explicit format name
    
    Returns:
        Reader function or None if not found
    """
    return ReaderRegistry.detect_reader(path, format)


def get_reader(format: str) -> Optional[Callable]:
    """Get reader by format name.
    
    Args:
        format: Format name
    
    Returns:
        Reader function or None if not found
    """
    return ReaderRegistry.get_reader(format)


def list_formats() -> dict[str, list[str]]:
    """List all supported formats.
    
    Returns:
        Dictionary with 'native' and 'custom' format lists
    """
    return ReaderRegistry.list_formats()

