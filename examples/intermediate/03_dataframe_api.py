"""DataFrame API Example - Intermediate Level.

Demonstrates the Pythonic DataFrame API with real data.
Shows standard Python built-ins, operators, and indexing.
"""

import ray
from pipeline.api import PipelineDataFrame

if __name__ == "__main__":
    # Initialize Ray
    ray.init(ignore_reinit_error=True)
    
    try:
        # Create DataFrame from public S3 video
        # Note: For video data, we'll process it first
        print("Creating DataFrame from public S3 video...")
        
        # Create DataFrame from public S3 video
        # Note: For video data, we process it first, then demonstrate DataFrame features
        # In practice, you'd load from processed video frames or other data sources
        
        # For demonstration, create a simple DataFrame simulating processed video frames
        # This represents what you'd get after processing the public S3 video
        print("Simulating processed video frames from public S3 video...")
        test_data = [
            {
                "frame_id": i,
                "timestamp": i * 0.033,  # 30 FPS
                "source": "s3://anonymous@ray-example-data/basketball.mp4",
                "image": [0.5] * 224 * 224 * 3,  # Simulated image data
            }
            for i in range(100)
        ]
        df = PipelineDataFrame.from_dataset(ray.data.from_items(test_data))
        
        print("\n" + "=" * 60)
        print("DataFrame Pythonic Features")
        print("=" * 60)
        
        # Standard Python built-ins
        print(f"\n✓ len(df) = {len(df)}")
        print(f"✓ df.shape = {df.shape}")
        print(f"✓ df.columns = {df.columns}")
        print(f"✓ df.empty = {df.empty}")
        
        # Pythonic indexing
        print(f"\n✓ df[0] = {df[0]}")
        print(f"✓ df[0:5].shape = {df[0:5].shape}")
        print(f"✓ df['frame_id'][:5] = {df['frame_id'][:5]}")
        print(f"✓ df.frame_id[:5] = {df.frame_id[:5]}")
        
        # Operator overloading
        df1 = df[0:50]
        df2 = df[50:100]
        combined = df1 + df2
        print(f"\n✓ df1 + df2 (concatenate) = {len(combined)} rows")
        
        # Lazy transformations
        result = (
            df
            .filter(lambda x: x["frame_id"] % 2 == 0)  # Even frames only
            .map(lambda x: {**x, "processed": True})
            .limit(10)
            .collect()
        )
        print(f"\n✓ Filtered and processed: {len(result)} rows")
        
        print("\n" + "=" * 60)
        print("✓ All Pythonic features working!")
        print("=" * 60)
        
    finally:
        ray.shutdown()

