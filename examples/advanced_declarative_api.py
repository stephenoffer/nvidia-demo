"""Advanced Declarative API Example.

Advanced example demonstrating complex pipeline configurations including:
- Multiple data source types (video, MCAP, HDF5, point clouds, etc.)
- GPU-accelerated deduplication (with CPU fallback)
- Isaac Lab and Cosmos Dreams integration
- Multi-pipeline orchestration
- Advanced configuration options

Works on both CPU-only and GPU clusters - automatically detects available resources.
"""

import logging
from pathlib import Path

from pipeline.api import Pipeline

logger = logging.getLogger(__name__)


def detect_compute_mode() -> str:
    """Detect available compute mode (CPU or GPU).
    
    Returns:
        "cpu" if no GPUs available, "auto" if GPUs available
    """
    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return "auto"  # Let pipeline auto-detect GPU
    except ImportError:
        pass
    return "cpu"  # CPU-only mode


def main():
    """Advanced example of using Python declarative API."""
    logger.info("Starting advanced declarative API example")
    
    # Detect available compute resources
    compute_mode = detect_compute_mode()
    use_gpu = compute_mode != "cpu"
    
    logger.info(f"Detected compute mode: {compute_mode}")
    if use_gpu:
        try:
            import torch
            num_gpus = torch.cuda.device_count()
            logger.info(f"Available GPUs: {num_gpus}")
        except ImportError:
            num_gpus = 0
    else:
        num_gpus = 0
        logger.info("Running in CPU-only mode")
    
    # Use local mock data if available, otherwise use S3 paths
    data_dir = Path(__file__).parent / "data"
    test_output = Path(__file__).parent / "test_output"
    test_output.mkdir(exist_ok=True)
    
    # Check for mock data files
    mock_parquet = data_dir / "parquet" / "mock_data.parquet"
    mock_jsonl = data_dir / "jsonl" / "mock_data.jsonl"
    mock_hdf5 = data_dir / "hdf5" / "mock_sensor_data.h5"
    existing_hdf5 = data_dir / "test_data.h5"
    existing_jsonl = data_dir / "test_groot.jsonl"
    pcd_path = data_dir / "test_pointcloud.pcd"
    ply_path = data_dir / "test_pointcloud.ply"
    numpy_path = data_dir / "numpy" / "mock_array.npy"
    
    use_local_data = (
        mock_parquet.exists()
        or mock_jsonl.exists()
        or existing_hdf5.exists()
        or existing_jsonl.exists()
        or pcd_path.exists()
        or numpy_path.exists()
    )
    
    if use_local_data:
        logger.info(f"Using local mock data from: {data_dir}")
        sources = []
        
        # Add parquet source
        if mock_parquet.exists():
            sources.append({
                "type": "parquet",
                "path": str(mock_parquet),
            })
        
        # Add JSONL/GR00T source
        if existing_jsonl.exists():
            sources.append({
                "type": "groot",
                "path": str(data_dir),
            })
        elif mock_jsonl.exists():
            sources.append({
                "type": "groot",
                "path": str(data_dir / "jsonl"),
            })
        
        # Add HDF5 source
        if mock_hdf5.exists():
            sources.append({
                "type": "hdf5",
                "path": str(mock_hdf5),
                "datasets": ["joint_positions", "joint_velocities", "base_pose"],
            })
        elif existing_hdf5.exists():
            sources.append({
                "type": "hdf5",
                "path": str(existing_hdf5),
            })
        
        # Add point cloud source
        if pcd_path.exists():
            sources.append({
                "type": "pointcloud",
                "path": str(pcd_path),
            })
        elif ply_path.exists():
            sources.append({
                "type": "pointcloud",
                "path": str(ply_path),
            })
        
        # Add NumPy source
        if numpy_path.exists():
            sources.append({
                "type": "numpy",
                "path": str(data_dir / "numpy"),
            })
        
        if not sources:
            logger.warning("No mock data files found. Please run: python examples/create_mock_data.py")
            sources = [{"type": "parquet", "path": str(data_dir)}]  # Fallback
        
        # Add public video source for testing
        sources.append({
            "type": "video",
            "path": "s3://anonymous@ray-example-data/basketball.mp4",
            "extract_frames": True,
            "frame_rate": 30,
            "max_frames": 100,
        })
        
        output_path = str(test_output / "curated")
    else:
        logger.info("Using S3 data paths (set up S3 credentials if needed)")
        sources = []
        output_path = "s3://robotics-data/curated/"
    
    if not use_local_data:
        # Use S3 paths for advanced example
        sources = [
            # Video data sources (using public Ray example data)
            {
                "type": "video",
                "path": "s3://anonymous@ray-example-data/basketball.mp4",
                "extract_frames": True,
                "frame_rate": 30,
                "max_frames": 100,
            },
            # MCAP (ROS2 message capture) files
            {
                "type": "mcap",
                "path": "s3://robotics-data/rosbags/",
                "topics": ["/camera/image_raw", "/imu/data", "/lidar/points"],
                "time_range": [0, 3600000000000],  # nanoseconds
                "include_metadata": True,
            },
            # ROS1 bag files
            {
                "type": "rosbag",
                "path": "s3://robotics-data/ros1_bags/",
                "topics": ["/camera/image_raw", "/odom"],
            },
            # HDF5 sensor data
            {
                "type": "hdf5",
                "path": "s3://robotics-data/sensor_data/",
                "datasets": ["joint_positions", "joint_velocities", "base_pose"],
            },
            # Point cloud data
            {
                "type": "pointcloud",
                "path": "s3://robotics-data/pointclouds/",
            },
            # NumPy arrays
            {
                "type": "numpy",
                "path": "s3://robotics-data/arrays/",
            },
            # Isaac Lab simulation data
            {
                "type": "isaac_lab",
                "path": "s3://robotics-data/isaac_lab/",
                "robot_type": "humanoid",
                "include_observations": True,
                "include_actions": True,
                "include_rewards": True,
            },
            # Cosmos Dreams synthetic videos
            {
                "type": "cosmos_dreams",
                "path": "s3://robotics-data/cosmos_dreams/",
                "include_metadata": True,
            },
        ]
    
    # Create an advanced pipeline with multiple data sources
    # Use compute_mode="auto" to automatically detect and use available resources
    pipeline = Pipeline(
        sources=sources,
        output=output_path,
        compute_mode=compute_mode,  # Auto-detect CPU/GPU
        num_gpus=num_gpus if use_gpu else 0,
        num_cpus=8 if use_gpu else 4,  # More CPUs for GPU mode, fewer for CPU-only
        batch_size=512 if use_gpu else 128,  # Smaller batch for CPU
        streaming=True,
        dedup_method="both" if use_gpu else "fuzzy",  # Semantic dedup requires GPU
        similarity_threshold=0.95,
        checkpoint_interval=1000,
    )
    
    # Export configuration to YAML (optional)
    pipeline.to_yaml("advanced_pipeline_config.yaml")
    logger.info("Exported configuration to advanced_pipeline_config.yaml")
    
    # For advanced use cases, you can also add simulation/synthetic data loaders
    # Note: These would typically be added via the low-level API
    # For declarative API, use the source types above
    
    # Run pipeline
    logger.info("Running advanced pipeline...")
    try:
        results = pipeline.run()
        
        # Log comprehensive results
        logger.info("Pipeline execution completed successfully")
        logger.info(f"Total samples: {results.get('total_samples', 0):,}")
        logger.info(f"Total duration: {results.get('total_duration', 0.0):.2f}s")
        logger.info(f"Deduplication rate: {results.get('dedup_rate', 0.0):.2%}")
        if use_gpu:
            logger.info(f"Average GPU utilization: {results.get('avg_gpu_util', 0.0):.2%}")
        logger.info(f"Output path: {results.get('output_path', 'N/A')}")
        
        if "stages" in results:
            logger.info(f"Stages completed: {len(results['stages'])}")
            for stage in results["stages"]:
                logger.info(
                    f"Stage '{stage.get('name', 'Unknown')}': "
                    f"{stage.get('duration', 0.0):.2f}s "
                    f"({stage.get('throughput', 0.0):.0f} items/s)"
                )
        
        logger.debug("Example: You can also load the same configuration from YAML using load_from_yaml()")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        logger.error("Note: This example requires data sources. Set up S3 credentials or ensure test data is available in the examples/data/ directory.")
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

