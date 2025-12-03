"""Full Pipeline Example - Advanced Level.

Demonstrates:
- Complex multi-source pipeline
- Real data sources (public S3 video + local test data)
- GPU-accelerated operations (if available)
- MLOps integration (profiling, validation)
- Experiment tracking
- Advanced configuration
"""

import logging
from pathlib import Path

from pipeline.api import Pipeline, PipelineBuilder, experiment
from pipeline.stages import DataProfiler, SchemaValidator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # Check for local test data
    data_dir = Path(__file__).parent.parent / "data"
    
    # Build pipeline with PipelineBuilder
    builder = PipelineBuilder()
    
    # Add public S3 video (always available)
    builder.add_source(
        "video",
        "s3://anonymous@ray-example-data/basketball.mp4",
        extract_frames=True,
        frame_rate=30,
        max_frames=100,
    )
    
    # Add local test data if available
    test_parquet = data_dir / "parquet" / "mock_data.parquet"
    if test_parquet.exists():
        logger.info(f"Adding local test data: {test_parquet}")
        builder.add_source("parquet", str(test_parquet))
    
    test_jsonl = data_dir / "test_groot.jsonl"
    if test_jsonl.exists():
        logger.info(f"Adding local JSONL data: {test_jsonl}")
        builder.add_source("groot", str(data_dir))
    
    # Add data quality stages
    builder.add_profiler(
        profile_columns=["image", "frame_id"],
        use_gpu=False,  # Set to True if GPU available
    )
    
    builder.add_validator(
        expected_schema={
            "frame_id": int,
            "image": list,
        },
        strict=False,  # Allow extra columns
    )
    
    # Configure compute
    builder.enable_gpu(num_gpus=0)  # Set to > 0 if GPU available
    builder.set_batch_size(128)
    
    # Configure execution
    builder.set_streaming(True)
    builder.set_output("./output/curated")
    
    # Build pipeline
    pipeline = builder.build()
    
    # Run with experiment tracking
    logger.info("Running pipeline with experiment tracking...")
    
    # Run pipeline (experiment tracking is optional)
    try:
        results = pipeline.run(
            experiment_name="full_pipeline_example",
            tracking_backend="mlflow",  # Falls back gracefully if not installed
            tags={"run_type": "example", "level": "advanced"},
        )
    except Exception as e:
        logger.warning(f"Experiment tracking not available: {e}")
        logger.info("Running without experiment tracking...")
        results = pipeline.run()
    
    # Inspect comprehensive results
    print("\n" + "=" * 60)
    print("Full Pipeline Results")
    print("=" * 60)
    print(f"Total samples: {results.get('total_samples', 0):,}")
    print(f"Total duration: {results.get('total_duration', 0.0):.2f}s")
    
    if "stages" in results:
        print(f"\nStage Performance:")
        for stage in results["stages"]:
            print(f"  {stage.get('name', 'Unknown'):30s} "
                  f"{stage.get('duration', 0.0):8.2f}s "
                  f"({stage.get('throughput', 0.0):8.0f} samples/s)")
    
    print(f"\nOutput: {results.get('output_path', 'N/A')}")

