"""Create mock test data for examples.

This script creates mock data files in the examples/data directory
for testing the pipeline examples without requiring external data sources.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_parquet_data(output_dir: Path) -> Path:
    """Create a mock parquet file with sample data."""
    logger.info("Creating mock parquet data...")
    
    # Create sample structured data
    data = pd.DataFrame({
        "id": range(1000),
        "text": [f"Sample text data {i}" for i in range(1000)],
        "value": [float(i * 1.5) for i in range(1000)],
        "category": [f"cat_{i % 10}" for i in range(1000)],
        "timestamp": pd.date_range("2024-01-01", periods=1000, freq="1H"),
        "score": np.random.rand(1000) * 100,
    })
    
    output_path = output_dir / "mock_data.parquet"
    data.to_parquet(output_path, index=False)
    logger.info(f"Created {output_path} with {len(data)} rows")
    return output_path


def create_jsonl_data(output_dir: Path) -> Path:
    """Create a mock JSONL file with sample data."""
    logger.info("Creating mock JSONL data...")
    
    output_path = output_dir / "mock_data.jsonl"
    with open(output_path, "w") as f:
        for i in range(100):
            record = {
                "id": i,
                "text": f"Sample text {i}",
                "metadata": {
                    "source": "mock",
                    "timestamp": f"2024-01-01T{i:02d}:00:00Z",
                },
                "values": [float(j) for j in range(10)],
            }
            f.write(json.dumps(record) + "\n")
    
    logger.info(f"Created {output_path} with 100 records")
    return output_path


def create_hdf5_data(output_dir: Path) -> Path:
    """Create a mock HDF5 file with sample sensor data."""
    logger.info("Creating mock HDF5 data...")
    
    output_path = output_dir / "mock_sensor_data.h5"
    
    # Create sample sensor data
    with pd.HDFStore(output_path, mode="w") as store:
        # Joint positions (7-DOF robot arm)
        joint_positions = pd.DataFrame(
            np.random.rand(1000, 7) * np.pi,  # Random angles in radians
            columns=[f"joint_{i}" for i in range(7)],
        )
        store.put("joint_positions", joint_positions)
        
        # Joint velocities
        joint_velocities = pd.DataFrame(
            np.random.rand(1000, 7) * 2.0 - 1.0,  # Random velocities -1 to 1
            columns=[f"joint_{i}_vel" for i in range(7)],
        )
        store.put("joint_velocities", joint_velocities)
        
        # Base pose (x, y, z, qx, qy, qz, qw)
        base_pose = pd.DataFrame(
            np.random.rand(1000, 7),
            columns=["x", "y", "z", "qx", "qy", "qz", "qw"],
        )
        store.put("base_pose", base_pose)
        
        # Timestamps
        timestamps = pd.DataFrame(
            {"timestamp": pd.date_range("2024-01-01", periods=1000, freq="10ms")}
        )
        store.put("timestamps", timestamps)
    
    logger.info(f"Created {output_path} with sensor datasets")
    return output_path


def create_numpy_data(output_dir: Path) -> Path:
    """Create mock NumPy array files."""
    logger.info("Creating mock NumPy data...")
    
    # Create .npy file
    array_data = np.random.rand(100, 50).astype(np.float32)
    npy_path = output_dir / "mock_array.npy"
    np.save(npy_path, array_data)
    logger.info(f"Created {npy_path}")
    
    # Create .npz file with multiple arrays
    npz_path = output_dir / "mock_arrays.npz"
    np.savez(
        npz_path,
        features=np.random.rand(100, 128),
        labels=np.random.randint(0, 10, size=100),
        metadata=np.array([1, 2, 3, 4, 5]),
    )
    logger.info(f"Created {npz_path}")
    
    return npy_path


def create_mock_data_directory() -> Path:
    """Create mock data directory structure."""
    script_dir = Path(__file__).parent
    data_dir = script_dir / "data"
    data_dir.mkdir(exist_ok=True)
    
    # Create subdirectories for different data types
    (data_dir / "parquet").mkdir(exist_ok=True)
    (data_dir / "jsonl").mkdir(exist_ok=True)
    (data_dir / "hdf5").mkdir(exist_ok=True)
    (data_dir / "numpy").mkdir(exist_ok=True)
    (data_dir / "video").mkdir(exist_ok=True)
    (data_dir / "pointcloud").mkdir(exist_ok=True)
    
    return data_dir


def main():
    """Create all mock test data files."""
    logger.info("Creating mock test data files")
    
    data_dir = create_mock_data_directory()
    logger.info(f"Data directory: {data_dir}")
    
    created_files = []
    
    # Create parquet data
    try:
        parquet_path = create_parquet_data(data_dir / "parquet")
        created_files.append(parquet_path)
    except Exception as e:
        logger.error(f"Failed to create parquet data: {e}")
    
    # Create JSONL data
    try:
        jsonl_path = create_jsonl_data(data_dir / "jsonl")
        created_files.append(jsonl_path)
    except Exception as e:
        logger.error(f"Failed to create JSONL data: {e}")
    
    # Create HDF5 data
    try:
        hdf5_path = create_hdf5_data(data_dir / "hdf5")
        created_files.append(hdf5_path)
    except Exception as e:
        logger.error(f"Failed to create HDF5 data: {e}")
    
    # Create NumPy data
    try:
        numpy_path = create_numpy_data(data_dir / "numpy")
        created_files.append(numpy_path)
    except Exception as e:
        logger.error(f"Failed to create NumPy data: {e}")
    
    # Create a simple text file for testing
    try:
        text_path = data_dir / "mock_text.txt"
        with open(text_path, "w") as f:
            for i in range(100):
                f.write(f"This is sample text line {i}.\n")
        created_files.append(text_path)
        logger.info(f"Created {text_path}")
    except Exception as e:
        logger.error(f"Failed to create text file: {e}")
    
    # Summary
    logger.info(f"Created {len(created_files)} mock data files:")
    for file_path in created_files:
        size_mb = file_path.stat().st_size / (1024 * 1024)
        logger.info(f"  - {file_path.name}: {size_mb:.2f} MB")
    logger.info(f"All files created in: {data_dir}")


if __name__ == "__main__":
    main()

