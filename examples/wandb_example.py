"""Example demonstrating Weights & Biases integration.

Shows how to use W&B for experiment tracking alongside or instead of MLflow.
"""

from pipeline.api import PipelineBuilder, experiment, Pipeline


def example_wandb_basic():
    """Basic W&B integration example."""
    pipeline = (
        PipelineBuilder()
        .add_source("video", "s3://bucket/videos/")
        .add_profiler(profile_columns=["image"], use_gpu=True)
        .set_output("s3://bucket/output/")
        .build()
    )
    
    # Run with W&B tracking
    results = pipeline.run(
        experiment_name="groot_pipeline_v1",
        tracking_backend="wandb",
        project="groot-pipeline",
        tags={"model": "groot", "version": "1.0"},
    )


def example_wandb_context_manager():
    """W&B integration using context manager."""
    with experiment(
        "groot_training_v2",
        tracking_backend="wandb",
        project="groot-pipeline",
        entity="my-team",
        tags={"model": "groot", "version": "2.0"},
    ):
        pipeline = (
            PipelineBuilder()
            .add_source("video", "s3://bucket/videos/")
            .add_profiler(profile_columns=["image"], use_gpu=True)
            .add_inference(model_uri="models:/groot-model/Production")
            .set_output("s3://bucket/output/")
            .build()
        )
        
        results = pipeline.run()


def example_both_mlflow_and_wandb():
    """Using both MLflow and W&B simultaneously."""
    with experiment(
        "groot_training_v3",
        tracking_backend="both",
        project="groot-pipeline",
        tags={"model": "groot", "version": "3.0"},
    ):
        pipeline = (
            PipelineBuilder()
            .add_source("video", "s3://bucket/videos/")
            .add_profiler(profile_columns=["image"], use_gpu=True)
            .add_inference(model_uri="models:/groot-model/Production")
            .set_output("s3://bucket/output/")
            .build()
        )
        
        results = pipeline.run()


def example_wandb_with_pipeline_instance():
    """W&B integration with Pipeline instance."""
    pipeline = Pipeline(
        sources=[{"type": "video", "path": "s3://bucket/videos/"}],
        output="s3://bucket/output/",
    )
    
    pipeline.add_profiler(profile_columns=["image"], use_gpu=True)
    pipeline.add_inference(model_uri="models:/groot-model/Production")
    
    # Run with W&B tracking
    results = pipeline.run(
        experiment_name="groot_pipeline_v1",
        tracking_backend="wandb",
        project="groot-pipeline",
        tags={"environment": "production"},
    )


def example_auto_detect():
    """Auto-detect tracking backend (prefers W&B if available)."""
    # Will use W&B if available, otherwise MLflow
    with experiment("groot_training_auto", tags={"model": "groot"}):
        pipeline = (
            PipelineBuilder()
            .add_source("video", "s3://bucket/videos/")
            .set_output("s3://bucket/output/")
            .build()
        )
        
        results = pipeline.run()


if __name__ == "__main__":
    print("Weights & Biases Integration Examples")
    print("=" * 50)
    
    print("\n1. Basic W&B Integration")
    example_wandb_basic()
    
    print("\n2. W&B Context Manager")
    example_wandb_context_manager()
    
    print("\n3. Both MLflow and W&B")
    example_both_mlflow_and_wandb()
    
    print("\n4. W&B with Pipeline Instance")
    example_wandb_with_pipeline_instance()
    
    print("\n5. Auto-detect Backend")
    example_auto_detect()

