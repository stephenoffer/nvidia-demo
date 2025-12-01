"""Basic Declarative API Example.

Simple example demonstrating the basic declarative Python API for configuring
and running a multimodal data curation pipeline.

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
    """Basic example of using Python declarative API."""
    logger.info("Starting basic declarative API example")
    
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
    
    use_local_data = (
        mock_parquet.exists()
        or mock_jsonl.exists()
        or existing_hdf5.exists()
        or existing_jsonl.exists()
    )
    
    if use_local_data:
        logger.info(f"Using local mock data from: {data_dir}")
        sources = []
        
        # Add parquet source if available
        if mock_parquet.exists():
            sources.append({
                "type": "parquet",
                "path": str(mock_parquet),
            })
        elif (data_dir / "parquet").exists():
            # Use parquet directory if it exists
            sources.append({
                "type": "parquet",
                "path": str(data_dir / "parquet"),
            })
        
        # Add JSONL source if available
        if existing_jsonl.exists():
            sources.append({
                "type": "groot",  # Use GR00T datasource for JSONL
                "path": str(data_dir),
            })
        elif mock_jsonl.exists():
            sources.append({
                "type": "groot",
                "path": str(data_dir / "jsonl"),
            })
        
        # Add HDF5 source if available
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
        
        if not sources:
            logger.warning("No mock data files found. Please run: python examples/create_mock_data.py")
            sources = [{"type": "parquet", "path": str(data_dir)}]  # Fallback
        
        # Add public video source for testing
        sources.append({
            "type": "video",
            "path": "s3://anonymous@ray-example-data/basketball.mp4",
            "extract_frames": True,
            "frame_rate": 30,
        })
        
        output_path = str(test_output / "curated")
    else:
        logger.info("Using S3 data paths (set up S3 credentials if needed)")
        logger.info("To use local mock data, run: python examples/create_mock_data.py")
        sources = [
            {
                "type": "video",
                "path": "s3://anonymous@ray-example-data/basketball.mp4",
                "extract_frames": True,
                "frame_rate": 30,
            },
            {
                "type": "parquet",
                "path": "s3://robotics-data/structured/",
            },
        ]
        output_path = "s3://robotics-data/curated/"
    
    # Create a simple pipeline with basic data sources
    # Use compute_mode="auto" to automatically detect and use available resources
    pipeline = Pipeline(
        sources=sources,
        output=output_path,
        compute_mode=compute_mode,  # Auto-detect CPU/GPU
        num_gpus=num_gpus if use_gpu else 0,
        num_cpus=4,  # Use 4 CPUs for CPU-only mode
        batch_size=128,  # Smaller batch for CPU
        streaming=True,
        dedup_method="fuzzy",  # CPU-friendly deduplication
    )
    
    # Run pipeline
    logger.info("Running pipeline...")
    try:
        results = pipeline.run()
        
        # Debug: Check result type
        if not isinstance(results, dict):
            raise TypeError(f"Expected dict, got {type(results)}: {results}")
        
        # Log results
        logger.info("Pipeline execution completed successfully")
        logger.info(f"Total samples: {results.get('total_samples', 0):,}")
        logger.info(f"Deduplication rate: {results.get('dedup_rate', 0.0):.2%}")
        logger.info(f"Output path: {results.get('output_path', 'N/A')}")
        if "total_duration" in results:
            logger.info(f"Total duration: {results.get('total_duration', 0.0):.2f}s")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        logger.error("Note: This example requires data sources. Set up S3 credentials or ensure test data is available in the examples/data/ directory.")
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

