"""Example: Using declarative YAML configuration.

Demonstrates how to configure and run a pipeline using YAML files.

Works on both CPU-only and GPU clusters - automatically detects available resources.
"""

import logging
from pathlib import Path

from pipeline.api import load_from_yaml

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
    """Example of using YAML-based pipeline configuration."""
    logger.info("Starting YAML configuration example")
    
    # Detect available compute resources
    compute_mode = detect_compute_mode()
    logger.info(f"Detected compute mode: {compute_mode}")
    if compute_mode == "cpu":
        logger.info("Running in CPU-only mode")
    else:
        try:
            import torch
            num_gpus = torch.cuda.device_count()
            logger.info(f"Available GPUs: {num_gpus}")
        except ImportError:
            pass
    
    # Load pipeline from YAML
    yaml_config = Path(__file__).parent / "pipeline_config.yaml"
    if not yaml_config.exists():
        raise FileNotFoundError(f"YAML config not found: {yaml_config}")
    
    logger.info(f"Loading pipeline configuration from: {yaml_config}")
    
    # Check for local mock data and modify YAML config if needed
    data_dir = Path(__file__).parent / "data"
    mock_parquet = data_dir / "parquet" / "mock_data.parquet"
    existing_hdf5 = data_dir / "test_data.h5"
    existing_jsonl = data_dir / "test_groot.jsonl"
    
    # If mock data exists, we'll need to modify sources
    # For now, load the YAML as-is and let it fail gracefully if S3 paths don't exist
    pipeline = load_from_yaml(str(yaml_config))
    
    # Override compute mode if running CPU-only
    # The YAML config may specify GPU, but we can override for CPU-only testing
    if compute_mode == "cpu":
        # Update pipeline config to use CPU mode
        if hasattr(pipeline, 'config'):
            pipeline.config.compute_mode = "cpu"
            pipeline.config.num_gpus = 0
            pipeline.config.enable_gpu_dedup = False
        elif hasattr(pipeline, 'enable_gpu'):
            pipeline.enable_gpu = False
            pipeline.num_gpus = 0
        logger.info("Overriding YAML config to use CPU-only mode")
    
    # Note: The YAML config uses S3 paths by default
    # To use local mock data, either:
    # 1. Modify pipeline_config.yaml to use local paths (see comments in YAML)
    # 2. Or use the Python API examples which auto-detect local mock data
    if not (mock_parquet.exists() or existing_hdf5.exists() or existing_jsonl.exists()):
        logger.info("YAML config uses S3 paths. For local testing:")
        logger.info("  1. Run: python examples/create_mock_data.py")
        logger.info("  2. Edit pipeline_config.yaml to use local paths (see comments)")
        logger.info("  3. Or use basic_declarative_api.py / advanced_declarative_api.py")

    # Run pipeline
    logger.info("Running pipeline...")
    try:
        results = pipeline.run()

        logger.info("Pipeline execution completed successfully")
        logger.info(f"Total samples: {results.get('total_samples', 0):,}")
        logger.info(f"Deduplication rate: {results.get('dedup_rate', 0.0):.2%}")
        logger.info(f"Output: {results.get('output_path', 'N/A')}")
        if "total_duration" in results:
            logger.info(f"Total duration: {results.get('total_duration', 0.0):.2f}s")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        logger.error("Note: This example requires data sources. Set up S3 credentials or ensure test data is available. The YAML config uses S3 paths by default.")
        raise


if __name__ == "__main__":
    main()

