"""Custom file-based datasources for robotics data formats.

Provides Ray Data datasources ONLY for formats without native Ray Data support.

Native Ray Data formats (use ray.data.read_* directly):
- Videos: ray.data.read_videos()
- NumPy: ray.data.read_numpy()
- TFRecords: ray.data.read_tfrecords()
- Audio: ray.data.read_audio()
- SQL: ray.data.read_sql()
- MCAP: ray.data.read_mcap()
- Images: ray.data.read_images()
- Parquet: ray.data.read_parquet()
- CSV: ray.data.read_csv()
- JSON: ray.data.read_json()
- Text: ray.data.read_text()
- Binary: ray.data.read_binary_files()

Custom datasources (formats without native support):
- URDF/SDF: Robot description files
- ROS Bag: ROS1 bag files
- HDF5: Scientific data format
- Point clouds: PCD, PLY, LAS
- Archives: ZIP, TAR
- Protobuf: Custom binary serialization
- YAML Config: Configuration files
- Binary: Custom structured binary
- Velodyne: LIDAR data
- Calibration: Camera calibration files
- MessagePack: Compact serialization
"""

from pipeline.datasources.archive import ArchiveDatasource
from pipeline.datasources.base import FileBasedDatasource
from pipeline.datasources.binary import BinaryDatasource
from pipeline.datasources.calibration import CalibrationDatasource
from pipeline.datasources.groot import GR00TDatasource
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

__all__ = [
    "FileBasedDatasource",
    "read_mcap",
    "ROSBagDatasource",
    "ROS2BagDatasource",
    "HDF5Datasource",
    "PointCloudDatasource",
    "URDFDatasource",
    "ArchiveDatasource",
    "ProtobufDatasource",
    "YAMLConfigDatasource",
    "BinaryDatasource",
    "VelodyneDatasource",
    "CalibrationDatasource",
    "MessagePackDatasource",
    "GR00TDatasource",
]
