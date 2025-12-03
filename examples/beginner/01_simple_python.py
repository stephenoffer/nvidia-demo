"""Simple Python API Example - Beginner Level.

The simplest possible example to get started with the pipeline.
Uses real data: public S3 video or local test data.
"""

from pipeline.api import pipeline

if __name__ == "__main__":
    # SIMPLEST EXAMPLE: Simple function API
    # Uses public S3 video (no credentials needed)
    p = pipeline(
        sources="s3://anonymous@ray-example-data/basketball.mp4",
        output="./output/curated",
    )
    
    # Run the pipeline
    results = p.run()
    
    print(f"✓ Pipeline completed!")
    print(f"  Processed: {results.get('total_samples', 0)} samples")
    print(f"  Output: {results.get('output_path', 'N/A')}")

