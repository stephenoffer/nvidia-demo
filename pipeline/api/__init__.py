"""High-level declarative API for multimodal data pipeline.

Provides both Python and YAML-based configuration interfaces
for easy pipeline setup and execution. Includes fluent builder API,
decorator-based task definitions, and DataFrame-like API inspired by
Prefect, Metaflow, MLflow, Ray Data, Spark, and Polars.

Quick Start:
    ```python
    from pipeline.api import pipeline, PipelineBuilder, read
    
    # Simple function API
    p = pipeline(
        sources="s3://bucket/data/",
        output="s3://bucket/output/",
    )
    results = p.run()
    
    # Fluent builder (short names)
    p = (
        PipelineBuilder()
        .source("video", "s3://bucket/videos/")
        .profile(profile_columns=["image"])
        .output("s3://bucket/output/")
        .build()
    )
    results = p.run()
    
    # DataFrame API
    df = read("s3://bucket/data/")
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
    cosmos,
    infer,
    isaac,
    omniverse,
    pipeline,
    profile,
    read,
    simple,
    to_pipeline,
    validate,
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
    "pipeline",
    "read",
    "simple",
    "infer",
    "profile",
    "validate",
    "isaac",
    "omniverse",
    "cosmos",
    "to_pipeline",
]

