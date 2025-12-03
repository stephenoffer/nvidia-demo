"""High-level declarative API for multimodal data pipeline.

Provides both Python and YAML-based configuration interfaces
for easy pipeline setup and execution. Includes fluent builder API,
decorator-based task definitions, and DataFrame-like API inspired by
Prefect, Metaflow, MLflow, Ray Data, Spark, and Polars.

Quick Start:
    ```python
    from pipeline.api import quick_start, PipelineBuilder, PipelineDataFrame
    
    # Simple pipeline
    pipeline = quick_start(
        input_paths="s3://bucket/data/",
        output_path="s3://bucket/output/",
    )
    results = pipeline.run()
    
    # Fluent builder
    pipeline = (
        PipelineBuilder()
        .add_source("video", "s3://bucket/videos/")
        .add_profiler(profile_columns=["image"])
        .set_output("s3://bucket/output/")
        .build()
    )
    results = pipeline.run()
    
    # DataFrame API
    df = PipelineDataFrame.from_paths("s3://bucket/data/")
    results = df.filter(lambda x: x["quality"] > 0.8).collect()
    ```
"""

from pipeline.api.dataframe import GroupedDataFrame, PipelineDataFrame
from pipeline.api.declarative import Pipeline, load_from_yaml
from pipeline.api.fluent import (
    PipelineBuilder,
    experiment,
    run_pipeline,
    stage,
    task,
)
from pipeline.api.helpers import (
    convert_dataframe_to_pipeline,
    create_cosmos_dreams_pipeline,
    create_inference_pipeline,
    create_isaac_lab_pipeline,
    create_omniverse_pipeline,
    create_profiling_pipeline,
    create_simple_pipeline,
    create_validation_pipeline,
    dataframe_from_dataset,
    dataframe_from_paths,
    quick_start,
)
from pipeline.api.multipipeline import MultiPipelineRunner

__all__ = [
    # Core APIs
    "Pipeline",
    "PipelineBuilder",
    "PipelineDataFrame",
    "GroupedDataFrame",
    "MultiPipelineRunner",
    # Configuration
    "load_from_yaml",
    # Decorators and context managers
    "task",
    "stage",
    "experiment",
    "run_pipeline",
    # Convenience functions
    "quick_start",
    "dataframe_from_paths",
    "dataframe_from_dataset",
    "create_simple_pipeline",
    "create_inference_pipeline",
    "create_profiling_pipeline",
    "create_validation_pipeline",
    "create_isaac_lab_pipeline",
    "create_omniverse_pipeline",
    "create_cosmos_dreams_pipeline",
    "convert_dataframe_to_pipeline",
]

