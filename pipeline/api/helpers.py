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


def quick_start(
    input_paths: Union[str, list[str]],
    output_path: str,
    enable_gpu: bool = False,
    **kwargs: Any,
) -> Pipeline:
    """Quick start function for simple pipeline creation.

    Creates a pipeline with minimal configuration for common use cases.

    Args:
        input_paths: Single path or list of input paths
        output_path: Output path for processed data
        enable_gpu: Enable GPU acceleration
        **kwargs: Additional pipeline configuration

    Returns:
        Configured Pipeline instance

    Example:
        ```python
        from pipeline.api import quick_start

        # Simple pipeline
        pipeline = quick_start(
            input_paths="s3://bucket/data/",
            output_path="s3://bucket/output/",
        )
        results = pipeline.run()

        # With GPU
        pipeline = quick_start(
            input_paths=["s3://bucket/videos/", "s3://bucket/rosbags/"],
            output_path="s3://bucket/output/",
            enable_gpu=True,
        )
        results = pipeline.run()
        ```
    """
    if isinstance(input_paths, str):
        input_paths = [input_paths]
    
    sources = [{"type": "auto", "path": path} for path in input_paths]
    
    return Pipeline(
        sources=sources,
        output=output_path,
        enable_gpu=enable_gpu,
        **kwargs,
    )


def dataframe_from_paths(
    paths: Union[str, list[str]],
    **read_kwargs: Any,
) -> PipelineDataFrame:
    """Convenience function to create DataFrame from paths.

    Alias for PipelineDataFrame.from_paths() for easier imports.

    Args:
        paths: Single path or list of paths
        **read_kwargs: Additional read arguments

    Returns:
        PipelineDataFrame instance

    Example:
        ```python
        from pipeline.api import dataframe_from_paths

        df = dataframe_from_paths("s3://bucket/data/")
        results = df.filter(lambda x: x["quality"] > 0.8).collect()
        ```
    """
    return PipelineDataFrame.from_paths(paths, **read_kwargs)


def dataframe_from_dataset(dataset: Dataset) -> PipelineDataFrame:
    """Convenience function to create DataFrame from Ray Dataset.

    Alias for PipelineDataFrame.from_dataset() for easier imports.

    Args:
        dataset: Ray Data Dataset

    Returns:
        PipelineDataFrame instance

    Example:
        ```python
        import ray.data
        from pipeline.api import dataframe_from_dataset

        ds = ray.data.read_parquet("s3://bucket/data/")
        df = dataframe_from_dataset(ds)
        ```
    """
    return PipelineDataFrame.from_dataset(dataset)


def create_simple_pipeline(
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
        from pipeline.api import create_simple_pipeline

        def custom_filter(batch):
            return {k: v for k, v in batch.items() if v is not None}

        pipeline = create_simple_pipeline(
            input_path="s3://bucket/data/",
            output_path="s3://bucket/output/",
            stages=[custom_filter],
        )
        results = pipeline.run()
        ```
    """
    builder = (
        PipelineBuilder()
        .add_source("auto", input_path)
        .set_output(output_path)
    )
    
    if stages:
        for stage in stages:
            builder.add_stage(stage)
    
    return builder.build()


def create_inference_pipeline(
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
        from pipeline.api import create_inference_pipeline

        pipeline = create_inference_pipeline(
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
        .add_source("auto", input_path)
        .add_inference(
            model_uri=model_uri,
            input_column=input_column,
            use_tensorrt=kwargs.pop("use_tensorrt", False),
            **kwargs,
        )
        .set_output(output_path)
        .enable_gpu(num_gpus=kwargs.pop("num_gpus", 1) if use_gpu else 0)
        .build()
    )


def create_profiling_pipeline(
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
        from pipeline.api import create_profiling_pipeline

        pipeline = create_profiling_pipeline(
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
        .add_source("auto", input_path)
        .add_profiler(
            profile_columns=profile_columns,
            use_gpu=use_gpu,
            **kwargs,
        )
        .set_output(output_path)
        .enable_gpu(num_gpus=kwargs.pop("num_gpus", 1) if use_gpu else 0)
        .build()
    )


def convert_dataframe_to_pipeline(
    df: PipelineDataFrame,
    output_path: str,
    **kwargs: Any,
) -> Pipeline:
    """Convert a DataFrame to a Pipeline for further processing.

    Args:
        df: PipelineDataFrame instance
        output_path: Output path for pipeline results
        **kwargs: Additional pipeline configuration

    Returns:
        Pipeline instance configured with DataFrame as source

    Note:
        This function writes the DataFrame to a temporary location and creates
        a pipeline that reads from it. For better performance, consider using
        the DataFrame API directly or checkpointing the DataFrame first.

    Example:
        ```python
        from pipeline.api import dataframe_from_paths, convert_dataframe_to_pipeline

        df = dataframe_from_paths("s3://bucket/data/")
        filtered_df = df.filter(lambda x: x["quality"] > 0.8)

        # Convert to pipeline for further processing
        pipeline = convert_dataframe_to_pipeline(
            filtered_df,
            output_path="s3://bucket/output/",
        )
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
        return Pipeline.quick_start(
            input_paths=temp_path,
            output_path=output_path,
            **kwargs,
        )
    except Exception as e:
        logger.error(f"Failed to convert DataFrame to pipeline: {e}")
        raise


def create_validation_pipeline(
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
        from pipeline.api import create_validation_pipeline

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
        ```
    """
    return (
        PipelineBuilder()
        .add_source("auto", input_path)
        .add_validator(
            expected_schema=expected_schema,
            strict=strict,
            **kwargs,
        )
        .set_output(output_path)
        .build()
    )


def create_isaac_lab_pipeline(
    simulation_path: str,
    output_path: str,
    robot_type: str = "humanoid",
    use_gpu: bool = True,
    **kwargs: Any,
) -> Pipeline:
    """Create a pipeline for Isaac Lab simulation data.

    Convenience function for processing Isaac Lab simulation trajectories.

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
        from pipeline.api import create_isaac_lab_pipeline

        pipeline = create_isaac_lab_pipeline(
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
        .add_isaac_lab(
            simulation_path=simulation_path,
            robot_type=robot_type,
            use_gpu=use_gpu,
            **kwargs,
        )
        .set_output(output_path)
        .enable_gpu(num_gpus=kwargs.pop("num_gpus", 1) if use_gpu else 0)
        .build()
    )


def create_omniverse_pipeline(
    omniverse_path: str,
    output_path: str,
    use_replicator: bool = False,
    use_gpu: bool = True,
    **kwargs: Any,
) -> Pipeline:
    """Create a pipeline for Omniverse USD/Replicator data.

    Convenience function for processing Omniverse scenes and Replicator outputs.

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
        from pipeline.api import create_omniverse_pipeline

        pipeline = create_omniverse_pipeline(
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
        .add_omniverse(
            omniverse_path=omniverse_path,
            use_replicator=use_replicator,
            use_gpu=use_gpu,
            **kwargs,
        )
        .set_output(output_path)
        .enable_gpu(num_gpus=kwargs.pop("num_gpus", 1) if use_gpu else 0)
        .build()
    )


def create_cosmos_dreams_pipeline(
    dreams_path: str,
    output_path: str,
    model_name: str = "groot-dreams-v1",
    **kwargs: Any,
) -> Pipeline:
    """Create a pipeline for Cosmos Dreams synthetic video data.

    Convenience function for processing GR00T Dreams synthetic videos.

    Args:
        dreams_path: Path to Cosmos Dreams output
        output_path: Output path for processed data
        model_name: Name of the video world model
        **kwargs: Additional Cosmos configuration

    Returns:
        Configured Pipeline instance

    Example:
        ```python
        from pipeline.api import create_cosmos_dreams_pipeline

        pipeline = create_cosmos_dreams_pipeline(
            dreams_path="s3://bucket/cosmos_dreams/",
            output_path="s3://bucket/curated/",
            model_name="groot-dreams-v1",
        )
        results = pipeline.run()
        ```
    """
    return (
        PipelineBuilder()
        .add_cosmos_dreams(
            dreams_path=dreams_path,
            model_name=model_name,
            **kwargs,
        )
        .set_output(output_path)
        .build()
    )

