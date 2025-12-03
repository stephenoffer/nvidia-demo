"""Simple Python API Example - Beginner Level.

The simplest possible example to get started with the pipeline.
Uses real data: public S3 video or local test data.
"""

from pipeline.api import Pipeline

if __name__ == "__main__":
    # SIMPLEST EXAMPLE: One-liner pipeline creation
    # Uses public S3 video (no credentials needed)
    pipeline = Pipeline.quick_start(
        input_paths="s3://anonymous@ray-example-data/basketball.mp4",
        output_path="./output/curated",
    )
    
    # Run the pipeline
    results = pipeline.run()
    
    print(f"✓ Pipeline completed!")
    print(f"  Processed: {results.get('total_samples', 0)} samples")
    print(f"  Output: {results.get('output_path', 'N/A')}")

