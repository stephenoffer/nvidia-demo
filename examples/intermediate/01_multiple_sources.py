"""Multiple Data Sources - Intermediate Level.

Demonstrates:
- Multiple data source types
- Real data sources (public S3 video + local test data)
- GPU acceleration (if available)
- Result inspection
"""

import logging
from pathlib import Path

from pipeline.api import Pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # Check for local test data
    data_dir = Path(__file__).parent.parent / "data"
    sources = []
    
    # Add public S3 video (always available, no credentials needed)
    sources.append({
        "type": "video",
        "path": "s3://anonymous@ray-example-data/basketball.mp4",
        "extract_frames": True,
        "frame_rate": 30,
        "max_frames": 100,  # Limit for faster execution
    })
    
    # Add local test data if available
    test_parquet = data_dir / "parquet" / "mock_data.parquet"
    if test_parquet.exists():
        logger.info(f"Found local test data: {test_parquet}")
        sources.append({
            "type": "parquet",
            "path": str(test_parquet),
        })
    
    test_jsonl = data_dir / "test_groot.jsonl"
    if test_jsonl.exists():
        logger.info(f"Found local JSONL data: {test_jsonl}")
        sources.append({
            "type": "groot",
            "path": str(data_dir),
        })
    
    if len(sources) == 1:
        logger.info("Using only public S3 video (no local test data found)")
        logger.info("To add local test data, run: python examples/create_mock_data.py")
    
    # Create pipeline with multiple sources
    pipeline = Pipeline(
        sources=sources,
        output="./output/curated",
        enable_gpu=False,  # Set to True if GPU available
        num_cpus=4,
        batch_size=128,
        streaming=True,
    )
    
    # Run pipeline
    logger.info("Running pipeline...")
    results = pipeline.run()
    
    # Inspect results
    print("\n" + "=" * 60)
    print("Pipeline Execution Results")
    print("=" * 60)
    print(f"Total samples processed: {results.get('total_samples', 0):,}")
    print(f"Total duration: {results.get('total_duration', 0.0):.2f} seconds")
    print(f"Output path: {results.get('output_path', 'N/A')}")
    
    if "stages" in results:
        print(f"\nStages completed: {len(results['stages'])}")
        for stage in results["stages"]:
            print(f"  - {stage.get('name', 'Unknown')}: {stage.get('duration', 0.0):.2f}s")

