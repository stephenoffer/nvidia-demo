"""DataFrame API Example - Intermediate Level.

Demonstrates the Pythonic DataFrame API with real data.
Shows standard Python built-ins, operators, and indexing.
"""

import ray
from pathlib import Path
from pipeline.api import read, PipelineDataFrame

if __name__ == "__main__":
    # Initialize Ray
    ray.init(ignore_reinit_error=True)
    
    try:
        # Try to load from local parquet file first, otherwise create test data
        data_dir = Path(__file__).parent.parent / "data"
        parquet_file = data_dir / "parquet" / "mock_data.parquet"
        
        if parquet_file.exists():
            print(f"Loading DataFrame from local test data: {parquet_file}")
            df = read(str(parquet_file))
        else:
            # Fallback: Create test data from scratch
            print("Creating DataFrame from test data...")
            test_data = [
                {
                    "id": i,
                    "text": f"Sample text {i}",
                    "value": float(i * 1.5),
                    "category": f"cat_{i % 10}",
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

