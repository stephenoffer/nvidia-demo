"""Example: Batch inference with MLOps integration and data quality checks.

Demonstrates:
- Batch inference with model registry integration
- Data quality checks (schema validation, profiling, drift detection)
- Feature engineering and transformation
- MLflow integration for tracking
"""

from pipeline import MultimodalPipeline, PipelineConfig
from pipeline.stages import (
    BatchInferenceStage,
    SchemaValidator,
    DataProfiler,
    DriftDetector,
    FeatureEngineeringStage,
    DataTransformer,
)
from pipeline.integrations import create_model_registry, create_feature_store


def main():
    """Run batch inference pipeline with MLOps integration."""
    
    # Create pipeline configuration
    config = PipelineConfig(
        input_paths=["s3://bucket/robotics_data/"],
        output_path="s3://bucket/inference_results/",
        num_gpus=4,
        batch_size=128,
        enable_observability=True,
    )
    
    pipeline = MultimodalPipeline(config)
    
    # Add schema validation
    schema_validator = SchemaValidator(
        expected_schema={
            "image": list,
            "text": str,
            "sensor_data": dict,
            "timestamp": float,
        },
        strict=True,
    )
    pipeline.add_stage(schema_validator)
    
    # Add data profiling
    profiler = DataProfiler(
        profile_columns=["image", "sensor_data"],
        compute_statistics=True,
        detect_outliers=True,
        output_path="s3://bucket/profiling_report.json",
    )
    pipeline.add_stage(profiler)
    
    # Add drift detection
    drift_detector = DriftDetector(
        reference_statistics={
            "sensor_data": {
                "mean": 0.0,
                "std": 1.0,
                "histogram": [10] * 10,
            }
        },
        columns_to_check=["sensor_data"],
        drift_threshold=0.1,
        method="ks_test",
    )
    pipeline.add_stage(drift_detector)
    
    # Add feature engineering
    def extract_image_features(item: dict) -> dict:
        """Extract features from image."""
        import numpy as np
        image = item.get("image", [])
        if isinstance(image, list) and len(image) > 0:
            img_array = np.array(image[0]) if isinstance(image[0], (list, np.ndarray)) else np.array(image)
            return {
                "image_mean": float(np.mean(img_array)),
                "image_std": float(np.std(img_array)),
                "image_shape": list(img_array.shape) if hasattr(img_array, 'shape') else [],
            }
        return {"image_mean": 0.0, "image_std": 0.0, "image_shape": []}
    
    feature_engineering = FeatureEngineeringStage(
        feature_functions={
            "image_stats": lambda x: extract_image_features(x),
        },
    )
    pipeline.add_stage(feature_engineering)
    
    # Add batch inference
    model_registry = create_model_registry(
        registry_type="mlflow",
        tracking_uri="http://mlflow-server:5000",
    )
    
    model_uri = model_registry.get_model(
        model_name="groot-vision-model",
        stage="Production",
    )
    
    inference_stage = BatchInferenceStage(
        model_uri=model_uri,
        input_column="image",
        output_column="predictions",
        batch_size=64,
        use_gpu=True,
        num_gpus=1,
        log_predictions=True,
        prediction_metadata={
            "pipeline_version": "1.0",
            "model_stage": "Production",
        },
    )
    pipeline.add_stage(inference_stage)
    
    # Add data transformation to filter low-confidence predictions
    def filter_high_confidence(item: dict) -> dict:
        """Filter items with high confidence predictions."""
        predictions = item.get("predictions", [])
        if isinstance(predictions, list) and len(predictions) > 0:
            confidence = predictions[0].get("confidence", 0.0) if isinstance(predictions[0], dict) else 0.0
            return confidence > 0.8
        return False
    
    transformer = DataTransformer(
        transform_func=lambda x: x,
        filter_func=filter_high_confidence,
    )
    pipeline.add_stage(transformer)
    
    # Run pipeline
    with pipeline:
        results = pipeline.run()
        print(f"Pipeline completed: {results}")


if __name__ == "__main__":
    main()

