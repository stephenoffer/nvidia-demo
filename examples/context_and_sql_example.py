"""Examples for PipelineContext and map_sql functionality.

Demonstrates:
1. Using PipelineContext to store global application state
2. Using map_sql to run SQL queries on batches using DuckDB
"""

from pipeline.api import PipelineDataFrame
from pipeline.utils.context import PipelineContext, get_context


def example_context_usage():
    """Example: Using PipelineContext for global state management."""
    print("=== PipelineContext Example ===\n")
    
    # Get the global context
    ctx = get_context()
    
    # Initialize context with Ray settings
    ctx.initialize(
        num_cpus=4,
        num_gpus=1,
        eager_free=True,
    )
    
    # Set application-level variables
    ctx.set("experiment_name", "my_experiment")
    ctx.set("model_version", "v1.0")
    ctx.set("batch_size", 1024)
    
    # Access variables
    experiment = ctx.get("experiment_name")
    print(f"Experiment: {experiment}")
    
    # Use dictionary syntax
    ctx["user_id"] = "user123"
    print(f"User ID: {ctx['user_id']}")
    
    # Check if variable exists
    if "model_version" in ctx:
        print(f"Model version: {ctx['model_version']}")
    
    # Get Ray Data context
    ray_data_ctx = ctx.ray_data_context
    if ray_data_ctx:
        print(f"Ray Data eager_free: {ray_data_ctx.eager_free}")
    
    # Convert to dictionary
    ctx_dict = ctx.to_dict()
    print(f"\nContext state: {ctx_dict}")
    
    # Clear context variables (doesn't reset Ray)
    ctx.clear()
    print(f"After clear: {len(ctx.to_dict()['context_vars'])} variables")


def example_map_sql():
    """Example: Using map_sql to run SQL queries on batches."""
    print("\n=== map_sql Example ===\n")
    
    # Create sample data
    import ray.data
    
    data = [
        {"episode_id": 1, "quality": 0.9, "timestamp": 1000, "name": "episode_1"},
        {"episode_id": 1, "quality": 0.8, "timestamp": 2000, "name": "episode_1"},
        {"episode_id": 2, "quality": 0.7, "timestamp": 3000, "name": "episode_2"},
        {"episode_id": 2, "quality": 0.95, "timestamp": 4000, "name": "episode_2"},
        {"episode_id": 3, "quality": 0.6, "timestamp": 5000, "name": "episode_3"},
    ]
    
    ds = ray.data.from_items(data)
    df = PipelineDataFrame.from_dataset(ds)
    
    print("Original data:")
    df.show()
    
    # Filter rows using SQL
    print("\n1. Filter rows (quality > 0.8):")
    filtered = df.map_sql("SELECT * FROM batch WHERE quality > 0.8")
    filtered.show()
    
    # Aggregate by episode_id
    print("\n2. Aggregate by episode_id:")
    aggregated = df.map_sql(
        """
        SELECT 
            episode_id,
            AVG(quality) as avg_quality,
            MAX(timestamp) as max_timestamp,
            COUNT(*) as count
        FROM batch
        GROUP BY episode_id
        """
    )
    aggregated.show()
    
    # Transform columns
    print("\n3. Transform columns (uppercase name):")
    transformed = df.map_sql(
        "SELECT *, UPPER(name) as upper_name FROM batch"
    )
    transformed.show()
    
    # Complex query with window functions
    print("\n4. Window function (rank by quality per episode):")
    ranked = df.map_sql(
        """
        SELECT 
            *,
            ROW_NUMBER() OVER (PARTITION BY episode_id ORDER BY quality DESC) as quality_rank
        FROM batch
        """
    )
    ranked.show()
    
    # Join with computed values
    print("\n5. Join with computed statistics:")
    stats = df.map_sql(
        """
        SELECT 
            b.*,
            s.avg_quality,
            s.max_quality
        FROM batch b
        JOIN (
            SELECT 
                episode_id,
                AVG(quality) as avg_quality,
                MAX(quality) as max_quality
            FROM batch
            GROUP BY episode_id
        ) s ON b.episode_id = s.episode_id
        """
    )
    stats.show()


def example_context_with_pipeline():
    """Example: Using context in a pipeline workflow."""
    print("\n=== Context with Pipeline Example ===\n")
    
    ctx = get_context()
    
    # Initialize context
    ctx.initialize(num_cpus=2)
    
    # Set pipeline metadata
    ctx.set("pipeline_id", "pipeline_001")
    ctx.set("input_path", "s3://bucket/data/")
    ctx.set("output_path", "s3://bucket/output/")
    
    # Use context in processing
    def process_batch(batch):
        """Process batch using context variables."""
        pipeline_id = ctx.get("pipeline_id")
        # Add pipeline_id to each row
        if isinstance(batch, dict):
            for key in batch.keys():
                batch[key] = [{"pipeline_id": pipeline_id, **row} 
                             if isinstance(row, dict) else row 
                             for row in batch[key]]
        return batch
    
    # Create sample pipeline
    import ray.data
    data = [{"value": i} for i in range(10)]
    ds = ray.data.from_items(data)
    df = PipelineDataFrame.from_dataset(ds)
    
    # Process with context
    processed = df.map_batches(process_batch)
    
    print("Processed data with pipeline_id:")
    processed.show()
    
    # Access context from anywhere in the pipeline
    print(f"\nPipeline ID from context: {ctx.get('pipeline_id')}")


if __name__ == "__main__":
    # Run examples
    try:
        example_context_usage()
        example_map_sql()
        example_context_with_pipeline()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

