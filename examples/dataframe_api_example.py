"""Example demonstrating the DataFrame-like API.

Shows how to use the PipelineDataFrame API inspired by Ray Data, Spark,
Polars, and Pandas for building data pipelines.
"""

from pipeline.api import PipelineDataFrame


def example_basic_operations():
    """Basic DataFrame operations with Pythonic features."""
    # Create DataFrame from paths
    df = PipelineDataFrame.from_paths(["s3://bucket/data/parquet/"])
    
    # Use standard Python built-ins
    print(f"Number of rows: {len(df)}")  # len() support
    print(f"Shape: {df.shape}")  # (rows, columns) tuple
    print(f"Columns: {df.columns}")  # List of column names
    print(f"Empty: {df.empty}")  # Boolean check
    
    # Pythonic indexing and slicing
    first_row = df[0]  # Row indexing
    first_10 = df[0:10]  # Slicing (like Pandas)
    column = df["episode_id"]  # Column access
    value = df.episode_id  # Attribute-style access
    
    # Filter rows
    filtered = df.filter(lambda x: x.get("quality", 0) > 0.8)
    
    # Transform rows
    transformed = filtered.map(lambda x: {
        **x,
        "processed": True,
        "quality_score": x.get("quality", 0) * 100,
    })
    
    # Select columns (using Pythonic indexing)
    selected = transformed[["image", "sensor_data", "quality_score"]]
    
    # Collect results
    results = selected.collect()


def example_groupby_aggregations():
    """Groupby and aggregation operations."""
    df = PipelineDataFrame.from_paths(["s3://bucket/data/"])
    
    # Group by episode_id and aggregate
    aggregated = (
        df
        .groupby("episode_id")
        .agg({
            "sensor_data": "mean",
            "timestamp": "max",
            "quality": "min",
        })
    )
    
    # Or use convenience methods
    episode_stats = (
        df
        .groupby("episode_id")
        .mean("sensor_data", "quality")
        .max("timestamp")
    )
    
    results = episode_stats.collect()


def example_joins():
    """Join operations."""
    df1 = PipelineDataFrame.from_paths(["s3://bucket/episodes/"])
    df2 = PipelineDataFrame.from_paths(["s3://bucket/metadata/"])
    
    # Inner join
    joined = df1.join(df2, on="episode_id", how="inner")
    
    # Left join
    left_joined = df1.join(df2, on="episode_id", how="left")
    
    results = joined.collect()


def example_gpu_acceleration():
    """GPU-accelerated operations."""
    df = PipelineDataFrame.from_paths(["s3://bucket/data/"])
    
    # GPU-accelerated batch processing
    processed = df.map_batches(
        lambda batch: {
            **batch,
            "gpu_processed": True,
        },
        batch_size=1000,
        use_gpu=True,
    )
    
    results = processed.collect()


def example_caching():
    """Caching intermediate results."""
    df = PipelineDataFrame.from_paths(["s3://bucket/data/"])
    
    # Cache expensive computation
    cached = (
        df
        .filter(lambda x: x["quality"] > 0.8)
        .map(lambda x: {**x, "processed": True})
        .cache()  # Cache in memory
    )
    
    # Reuse cached result multiple times
    result1 = cached.select("image").collect()
    result2 = cached.select("sensor_data").collect()


def example_fluent_chaining():
    """Fluent method chaining (inspired by Spark/Polars)."""
    df = PipelineDataFrame.from_paths(["s3://bucket/data/"])
    
    # Chain multiple operations
    result = (
        df
        .filter(lambda x: x["quality"] > 0.8)
        .select("episode_id", "image", "sensor_data", "timestamp")
        .groupby("episode_id")
        .agg({
            "sensor_data": "mean",
            "timestamp": "max",
        })
        .sort("timestamp", ascending=False)
        .limit(100)
        .cache()
        .collect()
    )


def example_distinct_and_sort():
    """Distinct and sort operations."""
    df = PipelineDataFrame.from_paths(["s3://bucket/data/"])
    
    # Remove duplicates
    distinct = df.distinct("episode_id")
    
    # Sort by timestamp
    sorted_df = distinct.sort("timestamp", ascending=True)
    
    results = sorted_df.take(10)


def example_union():
    """Union multiple DataFrames with operator overloading."""
    df1 = PipelineDataFrame.from_paths(["s3://bucket/data1/"])
    df2 = PipelineDataFrame.from_paths(["s3://bucket/data2/"])
    df3 = PipelineDataFrame.from_paths(["s3://bucket/data3/"])
    
    # Union using operator overloading (like pd.concat)
    combined = df1 + df2 + df3  # Using + operator
    # Alternative syntax
    union = df1 | df2 | df3  # Using | operator
    
    # Or use the method
    combined_method = df1.union(df2, df3)
    
    print(f"Combined rows: {len(combined)}")
    results = combined.collect()


def example_write_output():
    """Write results to output."""
    df = PipelineDataFrame.from_paths(["s3://bucket/data/"])
    
    processed = (
        df
        .filter(lambda x: x["quality"] > 0.8)
        .map(lambda x: {**x, "processed": True})
    )
    
    # Write to Parquet
    processed.write_parquet("s3://bucket/output/parquet/")
    
    # Write to JSON
    processed.write_json("s3://bucket/output/json/")


def example_repartitioning():
    """Repartitioning for optimization."""
    df = PipelineDataFrame.from_paths(["s3://bucket/data/"])
    
    # Repartition for better parallelism
    repartitioned = df.repartition(100)
    
    # Coalesce to reduce partitions
    coalesced = repartitioned.coalesce(10)
    
    results = coalesced.collect()


def example_flat_map():
    """Flat map for one-to-many transformations."""
    df = PipelineDataFrame.from_paths(["s3://bucket/data/"])
    
    # Expand each row into multiple rows
    expanded = df.flat_map(lambda x: [
        x,
        {**x, "augmented": True, "variant": "v1"},
        {**x, "augmented": True, "variant": "v2"},
    ])
    
    results = expanded.collect()


def example_pythonic_features():
    """Demonstrate Pythonic features (built-ins, operators, indexing)."""
    # Create test data
    import ray
    ray.init(ignore_reinit_error=True)
    
    test_data = [
        {"id": i, "value": i * 2, "name": f"item_{i}"}
        for i in range(100)
    ]
    df = PipelineDataFrame.from_dataset(ray.data.from_items(test_data))
    
    # Standard Python built-ins
    print(f"Number of rows: {len(df)}")  # len() support
    print(f"Shape: {df.shape}")  # (rows, columns) tuple
    print(f"Columns: {df.columns}")  # List of column names
    print(f"Empty: {df.empty}")  # Boolean check
    
    # Iteration
    print("\nFirst 5 rows:")
    for i, row in enumerate(df):
        if i >= 5:
            break
        print(f"  Row {i}: {row}")
    
    # Membership check
    if {"id": 0, "value": 0} in df:
        print("\n✓ Row found using 'in' operator")
    
    # Pythonic indexing
    print(f"\nFirst row: {df[0]}")
    print(f"First 10 rows shape: {df[0:10].shape}")
    print(f"Column 'id' (first 5): {df.id[:5]}")
    print(f"Column 'value' (first 5): {df['value'][:5]}")
    
    # Multiple column selection
    selected = df[["id", "value"]]
    print(f"\nSelected columns: {selected.columns}")
    
    # Operator overloading
    df1 = df[0:50]
    df2 = df[50:100]
    combined = df1 + df2  # Concatenate (like pd.concat)
    print(f"\nCombined DataFrame rows: {len(combined)}")
    
    # Copy
    df_copy = df.copy()
    print(f"Copy shape: {df_copy.shape}")
    print(f"Same object: {df == df_copy}")  # False (different objects)
    print(f"Same data: {len(df) == len(df_copy)}")  # True
    
    ray.shutdown()


if __name__ == "__main__":
    print("DataFrame API Examples")
    print("=" * 50)
    
    print("\n1. Basic Operations")
    example_basic_operations()
    
    print("\n2. Groupby Aggregations")
    example_groupby_aggregations()
    
    print("\n3. Joins")
    example_joins()
    
    print("\n4. GPU Acceleration")
    example_gpu_acceleration()
    
    print("\n5. Caching")
    example_caching()
    
    print("\n6. Fluent Chaining")
    example_fluent_chaining()
    
    print("\n7. Distinct and Sort")
    example_distinct_and_sort()
    
    print("\n8. Union")
    example_union()
    
    print("\n9. Write Output")
    example_write_output()
    
    print("\n10. Repartitioning")
    example_repartitioning()
    
    print("\n11. Flat Map")
    example_flat_map()
    
    print("\n12. Pythonic Features (Built-ins, Operators, Indexing)")
    example_pythonic_features()

