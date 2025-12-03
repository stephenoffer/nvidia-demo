"""Helper functions and convenience utilities for the pipeline API.

Provides common patterns and shortcuts for building pipelines quickly.
Includes convenience functions for NVIDIA integrations and common workflows.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional, Union

from ray.data import Dataset

from pipeline.api.dataframe import PipelineDataFrame
from pipeline.api.declarative import Pipeline
from pipeline.api.fluent import PipelineBuilder

logger = logging.getLogger(__name__)


def pipeline(
    sources: Union[str, list[str], list[dict[str, Any]]],
    output: str,
    enable_gpu: bool = False,
    num_gpus: int = 0,
    **kwargs: Any,
) -> Pipeline:
    """Create a pipeline from data sources.

    Short, intuitive function for creating pipelines.

    Args:
        sources: Single path (str), list of paths (List[str]), or list of source configs
        output: Output path for processed data
        enable_gpu: Enable GPU acceleration
        num_gpus: Number of GPUs to use
        **kwargs: Additional pipeline configuration

    Returns:
        Configured Pipeline instance

    Example:
        ```python
        from pipeline.api import pipeline

        # Simple pipeline
        p = pipeline(
            sources="s3://bucket/data/",
            output="s3://bucket/output/",
        )
        results = p.run()

        # With GPU
        p = pipeline(
            sources=["s3://bucket/videos/", "s3://bucket/rosbags/"],
            output="s3://bucket/output/",
            num_gpus=4,
        )
        results = p.run()
        ```
    """
    return Pipeline.create(
        sources=sources,
        output=output,
        enable_gpu=enable_gpu,
        num_gpus=num_gpus,
        **kwargs,
    )


def read(
    paths: Union[str, list[str]],
    **read_kwargs: Any,
) -> PipelineDataFrame:
    """Read data into a DataFrame.

    Short alias for PipelineDataFrame.from_paths().

    Args:
        paths: Single path or list of paths
        **read_kwargs: Additional read arguments

    Returns:
        PipelineDataFrame instance

    Example:
        ```python
        from pipeline.api import read

        df = read("s3://bucket/data/")
        results = df.filter(lambda x: x["quality"] > 0.8).collect()
        ```
    """
    return PipelineDataFrame.from_paths(paths, **read_kwargs)


def dataframe_from_dataset(dataset: Dataset) -> PipelineDataFrame:
    """Create DataFrame from Ray Dataset.

    Args:
        dataset: Ray Data Dataset

    Returns:
        PipelineDataFrame instance
    """
    return PipelineDataFrame.from_dataset(dataset)


def simple(
    input_path: str,
    output_path: str,
    stages: Optional[list[Callable]] = None,
    **kwargs: Any,
) -> Pipeline:
    """Create a simple pipeline with optional custom stages.

    Args:
        input_path: Input data path
        output_path: Output data path
        stages: Optional list of stage functions or PipelineStage instances
        **kwargs: Additional pipeline configuration

    Returns:
        Configured Pipeline instance

    Example:
        ```python
        from pipeline.api import simple

        def custom_filter(batch):
            return {k: v for k, v in batch.items() if v is not None}

        pipeline = simple(
            input_path="s3://bucket/data/",
            output_path="s3://bucket/output/",
            stages=[custom_filter],
        )
        results = pipeline.run()
        ```
    """
    builder = (
        PipelineBuilder()
        .source("auto", input_path)
        .output(output_path)
    )
    
    if stages:
        for stage in stages:
            builder.stage(stage)
    
        return builder.build()


def infer(
    input_path: str,
    output_path: str,
    model_uri: str,
    input_column: str = "data",
    use_gpu: bool = True,
    **kwargs: Any,
) -> Pipeline:
    """Create a pipeline for batch inference.

    Args:
        input_path: Input data path
        output_path: Output data path
        model_uri: Model URI or path
        input_column: Input column name
        use_gpu: Use GPU for inference
        **kwargs: Additional inference configuration

    Returns:
        Configured Pipeline instance

    Example:
        ```python
        from pipeline.api import infer

        pipeline = infer(
            input_path="s3://bucket/inference_data/",
            output_path="s3://bucket/predictions/",
            model_uri="models:/groot-model/Production",
            input_column="image",
            use_gpu=True,
        )
        results = pipeline.run()
        ```
    """
    return (
        PipelineBuilder()
        .source("auto", input_path)
        .infer(
            model_uri=model_uri,
            input_column=input_column,
            use_tensorrt=kwargs.pop("use_tensorrt", False),
            **kwargs,
        )
        .output(output_path)
        .gpu(num_gpus=kwargs.pop("num_gpus", 1) if use_gpu else 0)
        .build()
    )


def profile(
    input_path: str,
    output_path: str,
    profile_columns: Optional[list[str]] = None,
    use_gpu: bool = False,
    **kwargs: Any,
) -> Pipeline:
    """Create a pipeline for data profiling.

    Args:
        input_path: Input data path
        output_path: Output data path
        profile_columns: Columns to profile (None = all columns)
        use_gpu: Use GPU acceleration
        **kwargs: Additional profiling configuration

    Returns:
        Configured Pipeline instance

    Example:
        ```python
        from pipeline.api import profile

        pipeline = profile(
            input_path="s3://bucket/data/",
            output_path="s3://bucket/profiled/",
            profile_columns=["image", "sensor_data"],
            use_gpu=True,
        )
        results = pipeline.run()
        ```
    """
    return (
        PipelineBuilder()
        .source("auto", input_path)
        .profile(
            profile_columns=profile_columns,
            use_gpu=use_gpu,
            **kwargs,
        )
        .output(output_path)
        .gpu(num_gpus=kwargs.pop("num_gpus", 1) if use_gpu else 0)
        .build()
    )


def to_pipeline(
    df: PipelineDataFrame,
    output_path: str,
    **kwargs: Any,
) -> Pipeline:
    """Convert a DataFrame to a Pipeline.

    Args:
        df: PipelineDataFrame instance
        output_path: Output path for pipeline results
        **kwargs: Additional pipeline configuration

    Returns:
        Pipeline instance configured with DataFrame as source

    Example:
        ```python
        from pipeline.api import read, to_pipeline

        df = read("s3://bucket/data/")
        filtered_df = df.filter(lambda x: x["quality"] > 0.8)

        # Convert to pipeline
        pipeline = to_pipeline(filtered_df, output_path="s3://bucket/output/")
        results = pipeline.run()
        ```
    """
    import tempfile
    import os
    
    # Checkpoint DataFrame to temporary location
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, "dataframe_checkpoint")
    
    try:
        # Write DataFrame to temporary location
        df.write_parquet(temp_path)
        
        # Create pipeline that reads from checkpoint
        return Pipeline.create(
            sources=temp_path,
            output=output_path,
            **kwargs,
        )
    except Exception as e:
        logger.error(f"Failed to convert DataFrame to pipeline: {e}")
        raise


def validate(
    input_path: str,
    output_path: str,
    expected_schema: dict[str, Any],
    strict: bool = True,
    **kwargs: Any,
) -> Pipeline:
    """Create a pipeline for schema validation.

    Args:
        input_path: Input data path
        output_path: Output data path
        expected_schema: Expected data schema
        strict: Whether to reject invalid items
        **kwargs: Additional validation configuration

    Returns:
        Configured Pipeline instance

    Example:
        ```python
        from pipeline.api import validate

        pipeline = validate(
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
        ```
    """
    return (
        PipelineBuilder()
        .source("auto", input_path)
        .validate(
            expected_schema=expected_schema,
            strict=strict,
            **kwargs,
        )
        .output(output_path)
        .build()
    )


def isaac(
    simulation_path: str,
    output_path: str,
    robot_type: str = "humanoid",
    use_gpu: bool = True,
    **kwargs: Any,
) -> Pipeline:
    """Create a pipeline for Isaac Lab simulation data.

    Args:
        simulation_path: Path to Isaac Lab simulation data
        output_path: Output path for processed data
        robot_type: Type of robot (humanoid, quadruped, etc.)
        use_gpu: Use GPU acceleration
        **kwargs: Additional Isaac Lab configuration

    Returns:
        Configured Pipeline instance

    Example:
        ```python
        from pipeline.api import isaac

        pipeline = isaac(
            simulation_path="s3://bucket/isaac_lab/",
            output_path="s3://bucket/curated/",
            robot_type="humanoid",
            use_gpu=True,
        )
        results = pipeline.run()
        ```
    """
    return (
        PipelineBuilder()
        .isaac(
            simulation_path=simulation_path,
            robot_type=robot_type,
            use_gpu=use_gpu,
            **kwargs,
        )
        .output(output_path)
        .gpu(num_gpus=kwargs.pop("num_gpus", 1) if use_gpu else 0)
        .build()
    )


def omniverse(
    omniverse_path: str,
    output_path: str,
    use_replicator: bool = False,
    use_gpu: bool = True,
    **kwargs: Any,
) -> Pipeline:
    """Create a pipeline for Omniverse USD/Replicator data.

    Args:
        omniverse_path: Path to Omniverse USD files or Replicator output
        output_path: Output path for processed data
        use_replicator: Whether to use Replicator data
        use_gpu: Use GPU acceleration
        **kwargs: Additional Omniverse configuration

    Returns:
        Configured Pipeline instance

    Example:
        ```python
        from pipeline.api import omniverse

        pipeline = omniverse(
            omniverse_path="s3://bucket/omniverse/",
            output_path="s3://bucket/curated/",
            use_replicator=True,
            use_gpu=True,
        )
        results = pipeline.run()
        ```
    """
    return (
        PipelineBuilder()
        .omniverse(
            omniverse_path=omniverse_path,
            use_replicator=use_replicator,
            use_gpu=use_gpu,
            **kwargs,
        )
        .output(output_path)
        .gpu(num_gpus=kwargs.pop("num_gpus", 1) if use_gpu else 0)
        .build()
    )


def cosmos(
    dreams_path: str,
    output_path: str,
    model_name: str = "groot-dreams-v1",
    **kwargs: Any,
) -> Pipeline:
    """Create a pipeline for Cosmos Dreams synthetic video data.

    Args:
        dreams_path: Path to Cosmos Dreams output
        output_path: Output path for processed data
        model_name: Name of the video world model
        **kwargs: Additional Cosmos configuration

    Returns:
        Configured Pipeline instance

    Example:
        ```python
        from pipeline.api import cosmos

        pipeline = cosmos(
            dreams_path="s3://bucket/cosmos_dreams/",
            output_path="s3://bucket/curated/",
            model_name="groot-dreams-v1",
        )
        results = pipeline.run()
        ```
    """
    return (
        PipelineBuilder()
        .cosmos(
            dreams_path=dreams_path,
            model_name=model_name,
            **kwargs,
        )
        .output(output_path)
        .build()
    )

