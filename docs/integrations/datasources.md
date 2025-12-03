# Datasource Integrations

The pipeline supports a wide range of datasources for robotics and multimodal data:

## Supported Datasources

| Datasource | Format | Description | GPU Support |
|------------|--------|-------------|-------------|
| **Video** | MP4, AVI, MOV | Frame extraction, temporal segmentation | CUDA |
| **Text** | JSONL, TXT | Tokenization, quality filtering | Embeddings |
| **Parquet** | Parquet | Structured tabular data | cuDF |
| **MCAP** | MCAP | ROS2 robotics data format | CPU only |
| **HDF5** | HDF5 | Scientific datasets | CPU only |
| **Point Cloud** | PCD, PLY | 3D point cloud data | cuPy |
| **ROS Bag** | .bag | ROS1 bag files | CPU only |
| **ROS2 Bag** | .db3 | ROS2 bag files | CPU only |
| **Velodyne** | PCAP, VLP | LIDAR sensor data | cuPy |
| **Protobuf** | .pb, .bin | Protocol buffer messages | CPU only |
| **MessagePack** | .msgpack | Compact binary format | CPU only |
| **Binary** | Custom | Raw binary with struct parsing | CPU only |
| **Archive** | ZIP, TAR | Compressed archives | CPU only |
| **YAML** | YAML | Configuration files | CPU only |
| **URDF/SDF** | XML | Robot description files | CPU only |
| **Calibration** | YAML, JSON | Camera/sensor calibration | CPU only |

## Using Datasources

```python
from pipeline.api import Pipeline

# Multiple datasource types
pipeline = Pipeline(
    sources=[
        # Video files
        {
            "type": "video",
            "path": "s3://bucket/videos/*.mp4",
            "extract_frames": True,
            "frame_rate": 30,
            "resolution": (224, 224)
        },
        # MCAP (ROS2) files
        {
            "type": "mcap",
            "path": "s3://bucket/ros2_bags/*.mcap",
            "topics": ["/camera/image", "/lidar/points"],
            "time_range": (0, 3600)  # First hour
        },
        # Point cloud data
        {
            "type": "pointcloud",
            "path": "s3://bucket/lidar/*.pcd",
            "max_points": 100000,
            "downsample_large": True
        },
        # HDF5 datasets
        {
            "type": "hdf5",
            "path": "s3://bucket/sensor_data/*.h5",
            "datasets": ["imu_data", "joint_angles"],
            "max_datasets": 10
        },
        # ROS1 bag files
        {
            "type": "rosbag",
            "path": "s3://bucket/ros_bags/*.bag",
            "topics": ["/camera/image_raw", "/odom"],
            "max_messages": 10000
        }
    ],
    output="s3://bucket/curated/",
    enable_gpu=True
)
```

## Custom Datasources

```python
from pipeline.datasources.base import FileBasedDatasource
from ray.data.block import Block, ArrowBlockBuilder

class CustomDatasource(FileBasedDatasource):
    def _read_stream(self, f, path: str):
        # Read and process file
        data = f.readall()
        builder = ArrowBlockBuilder()
        builder.add({"data": process_data(data)})
        yield builder.build()
```

See [Extensibility Guide](docs/architecture/EXTENSIBILITY.md) for more details on creating custom datasources.

