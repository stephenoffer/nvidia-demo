"""Example demonstrating the fluent API and decorator-based task definitions.

Shows how to use the improved API inspired by Prefect, Metaflow, MLflow, and W&B.
"""

from pipeline.api import PipelineBuilder, experiment, task
from pipeline.stages import BatchInferenceStage, DataProfiler, SchemaValidator


# Example 1: Fluent Builder API (inspired by MLflow)
def example_fluent_builder():
    """Example using the fluent builder pattern."""
    pipeline = (
        PipelineBuilder()
        .source("video", "s3://bucket/videos/")
        .source("mcap", "s3://bucket/rosbags/")
        .profile(profile_columns=["image", "sensor_data"], use_gpu=True)
        .validate(expected_schema={"image": list, "text": str})
        .infer(
            model_uri="models:/groot-model/Production",
            input_column="image",
            use_tensorrt=True,
        )
        .output("s3://bucket/output/")
        .gpu(num_gpus=4)
        .build()
    )
    
    results = pipeline.run(experiment_name="groot_inference_v1")


# Example 2: Decorator-based Tasks (inspired by Prefect/Metaflow)
@task(name="custom_video_processor", retries=2, tags=["video", "gpu"])
def process_video_batch(batch: dict) -> dict:
    """Custom video processing task."""
    # Your custom processing logic
    processed = batch.copy()
    processed["processed"] = True
    return processed


@task(name="custom_feature_extractor", tags=["features"])
def extract_custom_features(batch: dict) -> dict:
    """Custom feature extraction task."""
    # Your custom feature extraction
    batch["custom_features"] = []
    return batch


def example_decorator_tasks():
    """Example using decorator-based task definitions."""
    pipeline = (
        PipelineBuilder()
        .source("video", "s3://bucket/videos/")
        .stage(process_video_batch)
        .stage(extract_custom_features)
        .output("s3://bucket/output/")
        .build()
    )
    
    results = pipeline.run()


# Example 3: Experiment Tracking with MLflow (inspired by MLflow)
def example_experiment_tracking_mlflow():
    """Example using MLflow experiment tracking."""
    with experiment(
        "groot_training_v2",
        tracking_backend="mlflow",
        tags={"model": "groot", "version": "2.0"},
    ):
        pipeline = (
            PipelineBuilder()
            .source("video", "s3://bucket/videos/")
            .profile(profile_columns=["image"])
            .infer(model_uri="models:/groot-model/Production")
            .output("s3://bucket/output/")
            .build()
        )
        
        results = pipeline.run()


# Example 3b: Experiment Tracking with Weights & Biases
def example_experiment_tracking_wandb():
    """Example using Weights & Biases experiment tracking."""
    with experiment(
        "groot_training_v2",
        tracking_backend="wandb",
        project="groot-pipeline",
        entity="my-team",
        tags={"model": "groot", "version": "2.0"},
    ):
        pipeline = (
            PipelineBuilder()
            .source("video", "s3://bucket/videos/")
            .profile(profile_columns=["image"])
            .infer(model_uri="models:/groot-model/Production")
            .output("s3://bucket/output/")
            .build()
        )
        
        results = pipeline.run()


# Example 3c: Experiment Tracking with Both MLflow and W&B
def example_experiment_tracking_both():
    """Example using both MLflow and W&B for experiment tracking."""
    with experiment(
        "groot_training_v2",
        tracking_backend="both",
        project="groot-pipeline",
        tags={"model": "groot", "version": "2.0"},
    ):
        pipeline = (
            PipelineBuilder()
            .source("video", "s3://bucket/videos/")
            .profile(profile_columns=["image"])
            .infer(model_uri="models:/groot-model/Production")
            .output("s3://bucket/output/")
            .build()
        )
        
        results = pipeline.run()


# Example 4: Method Chaining (inspired by MLflow fluent API)
def example_method_chaining():
    """Example using method chaining on Pipeline instance."""
    from pipeline.api import Pipeline
    
    pipeline = Pipeline(
        sources=[
            {"type": "video", "path": "s3://bucket/videos/"},
            {"type": "mcap", "path": "s3://bucket/rosbags/"},
        ],
        output="s3://bucket/output/",
        enable_gpu=True,
        num_gpus=4,
    )
    
    # Chain method calls
    pipeline.profile(profile_columns=["image"], use_gpu=True)
    pipeline.validate(expected_schema={"image": list})
    pipeline.infer(
        model_uri="models:/groot-model/Production",
        use_tensorrt=True,
    )
    
    # Run with MLflow experiment tracking
    results = pipeline.run(
        experiment_name="groot_pipeline_v1",
        tracking_backend="mlflow",
        tags={"environment": "production"},
    )
    
    # Or with W&B tracking
    results = pipeline.run(
        experiment_name="groot_pipeline_v1",
        tracking_backend="wandb",
        project="groot-pipeline",
        tags={"environment": "production"},
    )


# Example 5: Simple Function API
def example_quick_start():
    """Example using simple function API."""
    from pipeline.api import pipeline
    
    p = pipeline(
        sources="s3://bucket/data/",
        output="s3://bucket/curated/",
    )
    
    results = p.run()


if __name__ == "__main__":
    print("Fluent API Examples")
    print("=" * 50)
    
    print("\n1. Fluent Builder API")
    example_fluent_builder()
    
    print("\n2. Decorator-based Tasks")
    example_decorator_tasks()
    
    print("\n3a. Experiment Tracking with MLflow")
    example_experiment_tracking_mlflow()
    
    print("\n3b. Experiment Tracking with Weights & Biases")
    example_experiment_tracking_wandb()
    
    print("\n3c. Experiment Tracking with Both")
    example_experiment_tracking_both()
    
    print("\n4. Method Chaining")
    example_method_chaining()
    
    print("\n5. Quick Start")
    example_quick_start()

