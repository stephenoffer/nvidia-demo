"""Quick start examples for the pipeline API.

Demonstrates the simplest ways to use the pipeline API for common tasks.
"""

from pipeline.api import (
    PipelineBuilder,
    PipelineDataFrame,
    create_inference_pipeline,
    create_profiling_pipeline,
    create_simple_pipeline,
    create_validation_pipeline,
    dataframe_from_paths,
    quick_start,
)


def example_quick_start():
    """Simplest way to create and run a pipeline."""
    # One-liner pipeline creation
    pipeline = quick_start(
        input_paths="s3://bucket/data/",
        output_path="s3://bucket/output/",
    )
    results = pipeline.run()


def example_quick_start_multiple_paths():
    """Quick start with multiple input paths."""
    pipeline = quick_start(
        input_paths=[
            "s3://bucket/videos/",
            "s3://bucket/rosbags/",
            "s3://bucket/sensor_data/",
        ],
        output_path="s3://bucket/output/",
        enable_gpu=True,
    )
    results = pipeline.run()


def example_dataframe_quick_start():
    """Quick start using DataFrame API."""
    # Load data
    df = dataframe_from_paths("s3://bucket/data/")
    
    # Transform
    processed = (
        df
        .filter(lambda x: x.get("quality", 0) > 0.8)
        .map(lambda x: {**x, "processed": True})
    )
    
    # Write results
    processed.write_parquet("s3://bucket/output/")


def example_inference_pipeline():
    """Quick start for batch inference."""
    pipeline = create_inference_pipeline(
        input_path="s3://bucket/inference_data/",
        output_path="s3://bucket/predictions/",
        model_uri="models:/groot-model/Production",
        input_column="image",
        use_gpu=True,
    )
    results = pipeline.run()


def example_profiling_pipeline():
    """Quick start for data profiling."""
    pipeline = create_profiling_pipeline(
        input_path="s3://bucket/data/",
        output_path="s3://bucket/profiled/",
        profile_columns=["image", "sensor_data"],
        use_gpu=True,
    )
    results = pipeline.run()


def example_validation_pipeline():
    """Quick start for schema validation."""
    pipeline = create_validation_pipeline(
        input_path="s3://bucket/data/",
        output_path="s3://bucket/validated/",
        expected_schema={
            "image": list,
            "text": str,
            "sensor_data": list,
        },
        strict=True,
    )
    results = pipeline.run()


def example_simple_pipeline_with_custom_stage():
    """Simple pipeline with custom processing."""
    def custom_filter(batch: dict) -> dict:
        """Filter out None values."""
        return {k: v for k, v in batch.items() if v is not None}
    
    pipeline = create_simple_pipeline(
        input_path="s3://bucket/data/",
        output_path="s3://bucket/output/",
        stages=[custom_filter],
    )
    results = pipeline.run()


def example_fluent_builder_quick():
    """Quick fluent builder example."""
    pipeline = (
        PipelineBuilder()
        .add_source("auto", "s3://bucket/data/")
        .add_profiler(profile_columns=["image"])
        .set_output("s3://bucket/output/")
        .build()
    )
    results = pipeline.run()


def example_dataframe_to_pipeline():
    """Convert DataFrame to Pipeline."""
    # Start with DataFrame API
    df = PipelineDataFrame.from_paths("s3://bucket/data/")
    filtered_df = df.filter(lambda x: x["quality"] > 0.8)
    
    # Convert to pipeline for further processing
    pipeline = filtered_df.to_pipeline("s3://bucket/output/")
    results = pipeline.run()


if __name__ == "__main__":
    print("Pipeline API Quick Start Examples")
    print("=" * 50)
    
    print("\n1. Quick Start")
    example_quick_start()
    
    print("\n2. Quick Start with Multiple Paths")
    example_quick_start_multiple_paths()
    
    print("\n3. DataFrame Quick Start")
    example_dataframe_quick_start()
    
    print("\n4. Inference Pipeline")
    example_inference_pipeline()
    
    print("\n5. Profiling Pipeline")
    example_profiling_pipeline()
    
    print("\n6. Validation Pipeline")
    example_validation_pipeline()
    
    print("\n7. Simple Pipeline with Custom Stage")
    example_simple_pipeline_with_custom_stage()
    
    print("\n8. Fluent Builder Quick")
    example_fluent_builder_quick()
    
    print("\n9. DataFrame to Pipeline")
    example_dataframe_to_pipeline()

