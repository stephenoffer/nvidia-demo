"""Fluent API for building pipelines with decorator-based task definitions.

Inspired by Prefect, Metaflow, and MLflow's intuitive APIs. Provides a
Pythonic, decorator-based interface for defining and composing pipeline stages.
"""

from __future__ import annotations

import functools
import logging
from contextlib import contextmanager
from typing import Any, Callable, Iterator, Optional, TypeVar, Union

from ray.data import Dataset

from pipeline.stages.base import PipelineStage, ProcessorBase
from pipeline.stages.inference.batch_inference import BatchInferenceStage
from pipeline.stages.analyzers.data_profiler import DataProfiler
from pipeline.stages.processors.data_transformer import DataAggregator, DataTransformer
from pipeline.stages.analyzers.drift_detector import DriftDetector
from pipeline.stages.processors.feature_engineering import FeatureEngineeringStage
from pipeline.stages.validators.schema_validator import SchemaValidator

logger = logging.getLogger(__name__)

T = TypeVar("T")


def task(
    name: Optional[str] = None,
    retries: int = 0,
    timeout: Optional[float] = None,
    tags: Optional[list[str]] = None,
    **task_kwargs: Any,
) -> Callable:
    """Decorator to define a pipeline task.

    Inspired by Prefect and Metaflow's @task decorator. Makes any function
    a pipeline stage that can be composed and executed.

    Args:
        name: Task name (defaults to function name)
        retries: Number of retries on failure
        timeout: Task timeout in seconds
        tags: Tags for task organization
        **task_kwargs: Additional task configuration

    Example:
        ```python
        @task(name="process_video", retries=2, tags=["video", "gpu"])
        def process_video_batch(batch: dict) -> dict:
            # Process video batch
            return processed_batch

        # Use in pipeline
        pipeline = Pipeline()
        pipeline.add_task(process_video_batch)
        ```
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            return func(*args, **kwargs)
        
        wrapper._is_task = True
        wrapper._task_name = name or func.__name__
        wrapper._task_retries = retries
        wrapper._task_timeout = timeout
        wrapper._task_tags = tags or []
        wrapper._task_kwargs = task_kwargs
        return wrapper
    return decorator


def stage(
    stage_class: type[PipelineStage],
    name: Optional[str] = None,
    **stage_kwargs: Any,
) -> PipelineStage:
    """Create a pipeline stage instance with fluent configuration.

    Args:
        stage_class: Stage class to instantiate
        name: Stage name (defaults to class name)
        **stage_kwargs: Stage configuration

    Returns:
        Configured stage instance

    Example:
        ```python
        profiler = stage(
            DataProfiler,
            name="data_quality_check",
            profile_columns=["image", "sensor_data"],
            use_gpu=True,
        )
        ```
    """
    instance = stage_class(**stage_kwargs)
    if name:
        instance.name = name
    return instance


class PipelineBuilder:
    """Fluent builder for constructing pipelines.

    Inspired by MLflow's fluent API and Prefect's flow builder pattern.
    Provides method chaining for intuitive pipeline construction.

    Example:
        ```python
        # Basic usage (short names)
        p = (
            PipelineBuilder()
            .source("video", "s3://bucket/videos/")
            .source("mcap", "s3://bucket/rosbags/")
            .stage(DataProfiler(profile_columns=["image"]))
            .stage(BatchInferenceStage(model_uri="models:/model/Production"))
            .output("s3://bucket/output/")
            .gpu()
            .build()
        )
        
        # With convenience methods (short names)
        p = (
            PipelineBuilder()
            .source("auto", "s3://bucket/data/")
            .profile(profile_columns=["image"], use_gpu=True)
            .validate(expected_schema={"image": list})
            .infer(model_uri="models:/model/Production")
            .output("s3://bucket/output/")
            .build()
        )
        ```
    """

    def __init__(self) -> None:
        """Initialize pipeline builder."""
        self.sources: list[dict[str, Any]] = []
        self.stages: list[PipelineStage] = []
        self.output_path: Optional[str] = None
        self.config: dict[str, Any] = {}

    def source(
        self,
        source_type: str,
        path: str,
        **source_kwargs: Any,
    ) -> PipelineBuilder:
        """Add a data source to the pipeline.

        Args:
            source_type: Type of data source (video, mcap, hdf5, etc.)
            path: Path to data source
            **source_kwargs: Additional source configuration

        Returns:
            Self for method chaining
        """
        self.sources.append({
            "type": source_type,
            "path": path,
            **source_kwargs,
        })
        return self

    def stage(
        self,
        stage: Union[PipelineStage, Callable],
        name: Optional[str] = None,
    ) -> PipelineBuilder:
        """Add a processing stage to the pipeline.

        Args:
            stage: Stage instance or task function
            name: Optional stage name

        Returns:
            Self for method chaining
        """
        if callable(stage) and hasattr(stage, "_is_task"):
            # Convert task function to stage
            from pipeline.stages.base import ProcessorBase
            
            class TaskStage(ProcessorBase):
                def __init__(self, task_func: Callable, name: str, batch_size: int = 100, **kwargs: Any):
                    super().__init__(batch_size=batch_size, **kwargs)
                    self.task_func = task_func
                    self.name = name
                
                def process(self, dataset: Dataset) -> Dataset:
                    def process_batch(batch: dict[str, Any]) -> dict[str, Any]:
                        return self.task_func(batch)
                    return dataset.map_batches(
                        process_batch,
                        batch_size=self.batch_size,
                        batch_format="pandas",
                    )
            
            # Get batch_size from config if available
            task_batch_size = self.config.get("batch_size", 100)
            stage = TaskStage(
                stage,
                name or stage._task_name,
                batch_size=task_batch_size,
            )
        
        if name and hasattr(stage, "name"):
            stage.name = name
        
        self.stages.append(stage)
        return self

    def profile(
        self,
        profile_columns: Optional[list[str]] = None,
        use_gpu: bool = False,
        **kwargs: Any,
    ) -> PipelineBuilder:
        """Add data profiling stage.

        Args:
            profile_columns: Columns to profile
            use_gpu: Use GPU acceleration
            **kwargs: Additional profiler configuration

        Returns:
            Self for method chaining
        """
        return self.stage(
            DataProfiler(
                profile_columns=profile_columns,
                use_gpu=use_gpu,
                **kwargs,
            ),
            name="data_profiler",
        )

    def validate(
        self,
        expected_schema: dict[str, Any],
        strict: bool = True,
        **kwargs: Any,
    ) -> PipelineBuilder:
        """Add schema validation stage.

        Args:
            expected_schema: Expected data schema
            strict: Whether to reject invalid items
            **kwargs: Additional validator configuration

        Returns:
            Self for method chaining
        """
        return self.stage(
            SchemaValidator(
                expected_schema=expected_schema,
                strict=strict,
                **kwargs,
            ),
            name="schema_validator",
        )

    def infer(
        self,
        model_uri: str,
        input_column: str = "data",
        use_tensorrt: bool = False,
        **kwargs: Any,
    ) -> PipelineBuilder:
        """Add batch inference stage.

        Args:
            model_uri: Model URI or path
            input_column: Input column name
            use_tensorrt: Use TensorRT optimization
            **kwargs: Additional inference configuration

        Returns:
            Self for method chaining
        """
        return self.stage(
            BatchInferenceStage(
                model_uri=model_uri,
                input_column=input_column,
                use_tensorrt=use_tensorrt,
                **kwargs,
            ),
            name="batch_inference",
        )

    def drift(
        self,
        reference_statistics: dict[str, Any],
        drift_threshold: float = 0.1,
        **kwargs: Any,
    ) -> PipelineBuilder:
        """Add drift detection stage.

        Args:
            reference_statistics: Reference distribution statistics
            drift_threshold: Drift detection threshold
            **kwargs: Additional detector configuration

        Returns:
            Self for method chaining
        """
        return self.stage(
            DriftDetector(
                reference_statistics=reference_statistics,
                drift_threshold=drift_threshold,
                **kwargs,
            ),
            name="drift_detector",
        )

    def features(
        self,
        feature_functions: dict[str, Callable],
        **kwargs: Any,
    ) -> PipelineBuilder:
        """Add feature engineering stage.

        Args:
            feature_functions: Dictionary of feature extraction functions
            **kwargs: Additional feature engineering configuration

        Returns:
            Self for method chaining
        """
        return self.stage(
            FeatureEngineeringStage(
                feature_functions=feature_functions,
                **kwargs,
            ),
            name="feature_engineering",
        )
    
    def transform(
        self,
        transform_func: Callable,
        **kwargs: Any,
    ) -> PipelineBuilder:
        """Add data transformation stage.

        Args:
            transform_func: Transformation function
            **kwargs: Additional transformer configuration

        Returns:
            Self for method chaining
        """
        return self.stage(
            DataTransformer(
                transform_func=transform_func,
                **kwargs,
            ),
            name="data_transformer",
        )
    
    def aggregate(
        self,
        group_by: list[str],
        aggregations: dict[str, str],
        **kwargs: Any,
    ) -> PipelineBuilder:
        """Add data aggregation stage.

        Args:
            group_by: Columns to group by
            aggregations: Aggregation operations (e.g., {"value": "mean"})
            **kwargs: Additional aggregator configuration

        Returns:
            Self for method chaining
        """
        return self.stage(
            DataAggregator(
                group_by=group_by,
                aggregations=aggregations,
                **kwargs,
            ),
            name="data_aggregator",
        )
    
    def isaac(
        self,
        simulation_path: str,
        robot_type: str = "humanoid",
        use_gpu: bool = True,
        **kwargs: Any,
    ) -> PipelineBuilder:
        """Add Isaac Lab simulation data source.

        Args:
            simulation_path: Path to Isaac Lab simulation data
            robot_type: Type of robot (humanoid, quadruped, etc.)
            use_gpu: Use GPU acceleration for loading
            **kwargs: Additional Isaac Lab loader configuration

        Returns:
            Self for method chaining
        """
        from pipeline.integrations.isaac_lab import IsaacLabLoader
        
        loader = IsaacLabLoader(
            simulation_path=simulation_path,
            robot_type=robot_type,
            use_gpu=use_gpu,
            use_gpu_object_store=use_gpu,
            **kwargs,
        )
        
        # Add as source
        return self.source(
            "isaac_lab",
            simulation_path,
            robot_type=robot_type,
            loader=loader,
            **kwargs,
        )
    
    def omniverse(
        self,
        omniverse_path: str,
        use_replicator: bool = False,
        use_gpu: bool = True,
        **kwargs: Any,
    ) -> PipelineBuilder:
        """Add Omniverse USD/Replicator data source.

        Args:
            omniverse_path: Path to Omniverse USD files or Replicator output
            use_replicator: Whether to use Replicator data
            use_gpu: Use GPU acceleration for USD processing
            **kwargs: Additional Omniverse loader configuration

        Returns:
            Self for method chaining
        """
        from pipeline.integrations.omniverse import OmniverseLoader
        
        loader = OmniverseLoader(
            omniverse_path=omniverse_path,
            use_replicator=use_replicator,
            use_gpu=use_gpu,
            use_gpu_object_store=use_gpu,
            **kwargs,
        )
        
        return self.source(
            "omniverse",
            omniverse_path,
            use_replicator=use_replicator,
            loader=loader,
            **kwargs,
        )
    
    def cosmos(
        self,
        dreams_path: str,
        model_name: str = "groot-dreams-v1",
        **kwargs: Any,
    ) -> PipelineBuilder:
        """Add Cosmos Dreams synthetic video data source.

        Args:
            dreams_path: Path to Cosmos Dreams output
            model_name: Name of the video world model
            **kwargs: Additional Cosmos loader configuration

        Returns:
            Self for method chaining
        """
        return self.source(
            "cosmos_dreams",
            dreams_path,
            model_name=model_name,
            **kwargs,
        )

    def output(self, output_path: str) -> PipelineBuilder:
        """Set pipeline output path.

        Args:
            output_path: Output path for processed data

        Returns:
            Self for method chaining
        """
        if not isinstance(output_path, str) or not output_path.strip():
            raise ValueError(f"output_path must be a non-empty string, got {type(output_path).__name__}: {output_path}")
        self.output_path = output_path
        return self

    def gpu(self, num_gpus: int = 1) -> PipelineBuilder:
        """Enable GPU acceleration.

        Args:
            num_gpus: Number of GPUs to use

        Returns:
            Self for method chaining

        Raises:
            ValueError: If num_gpus is not positive
        """
        if not isinstance(num_gpus, int) or num_gpus <= 0:
            raise ValueError(f"num_gpus must be a positive integer, got {num_gpus}")
        
        self.config["enable_gpu"] = True
        self.config["num_gpus"] = num_gpus
        return self
    
    def cpu(self) -> PipelineBuilder:
        """Use CPU only (disable GPU).

        Returns:
            Self for method chaining
        """
        self.config["enable_gpu"] = False
        self.config["num_gpus"] = 0
        return self
    
    def batch(self, batch_size: int) -> PipelineBuilder:
        """Set default batch size.

        Args:
            batch_size: Default batch size for stages

        Returns:
            Self for method chaining

        Raises:
            ValueError: If batch_size is not positive
        """
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError(f"batch_size must be a positive integer, got {batch_size}")
        
        self.config["batch_size"] = batch_size
        return self
    
    def streaming(self, enabled: bool = True) -> PipelineBuilder:
        """Enable or disable streaming execution mode.

        Args:
            enabled: Whether to enable streaming

        Returns:
            Self for method chaining
        """
        self.config["streaming"] = enabled
        return self
    
    def gpu_store(self) -> PipelineBuilder:
        """Enable GPU object store for RDMA transfers.

        Returns:
            Self for method chaining
        """
        self.config["use_gpu_object_store"] = True
        if not self.config.get("enable_gpu", False):
            self.gpu()
        return self
    
    def checkpoint(self, interval: int) -> PipelineBuilder:
        """Set checkpoint interval for fault tolerance.

        Args:
            interval: Checkpoint interval (number of batches)

        Returns:
            Self for method chaining
        """
        if not isinstance(interval, int) or interval <= 0:
            raise ValueError(f"interval must be a positive integer, got {interval}")
        
        self.config["checkpoint_interval"] = interval
        return self
    
    def dedup(
        self,
        method: str = "fuzzy",
        similarity_threshold: float = 0.95,
    ) -> PipelineBuilder:
        """Set deduplication method and threshold.

        Args:
            method: Deduplication method ("fuzzy", "semantic", "both")
            similarity_threshold: Similarity threshold (0.0-1.0)

        Returns:
            Self for method chaining
        """
        if method not in {"fuzzy", "semantic", "both"}:
            raise ValueError(f"method must be 'fuzzy', 'semantic', or 'both', got {method}")
        
        if not isinstance(similarity_threshold, (int, float)) or not (0.0 <= similarity_threshold <= 1.0):
            raise ValueError(f"similarity_threshold must be between 0.0 and 1.0, got {similarity_threshold}")
        
        self.config["dedup_method"] = method
        self.config["similarity_threshold"] = similarity_threshold
        return self

    def config(self, **kwargs: Any) -> PipelineBuilder:
        """Set additional configuration.

        Args:
            **kwargs: Configuration options

        Returns:
            Self for method chaining
        """
        self.config.update(kwargs)
        return self

    def build(self) -> Any:
        """Build and return pipeline instance.

        Returns:
            Configured Pipeline instance

        Raises:
            ValueError: If required configuration is missing or invalid
        """
        if not self.sources:
            raise ValueError(
                "At least one source must be added. "
                "Use .source() or convenience methods like .isaac()"
            )
        
        if not self.output_path:
            raise ValueError(
                "Output path must be set. Use .output(path) to specify output location."
            )
        
        if not isinstance(self.output_path, str) or not self.output_path.strip():
            raise ValueError(
                f"Output path must be a non-empty string, got {type(self.output_path).__name__}: {self.output_path}"
            )
        
        # Validate sources
        for i, source in enumerate(self.sources):
            if not isinstance(source, dict):
                raise ValueError(
                    f"Source {i} must be a dict, got {type(source).__name__}. "
                    f"Use .source() or convenience methods like .isaac()"
                )
            if "type" not in source or "path" not in source:
                raise ValueError(
                    f"Source {i} must have 'type' and 'path' keys. "
                    f"Got keys: {list(source.keys())}"
                )
            if not isinstance(source["path"], str) or not source["path"].strip():
                raise ValueError(
                    f"Source {i} path must be a non-empty string, "
                    f"got {type(source['path']).__name__}: {source['path']}"
                )
        
        # Validate stages
        for i, stage in enumerate(self.stages):
            if not isinstance(stage, PipelineStage):
                raise ValueError(
                    f"Stage {i} must be a PipelineStage instance, got {type(stage).__name__}. "
                    f"Use .stage() or convenience methods like .profile()"
                )

        from pipeline.api.declarative import Pipeline

        try:
            pipeline = Pipeline(
                sources=self.sources,
                output=self.output_path,
                **self.config,
            )

            # Add custom stages
            for stage in self.stages:
                pipeline.pipeline.stage(stage)

            return pipeline
        except Exception as e:
            logger.error(f"Failed to build pipeline: {e}", exc_info=True)
            raise ValueError(
                f"Failed to build pipeline: {e}. "
                f"Check that all sources and stages are properly configured."
            ) from e
    
    def __enter__(self) -> "PipelineBuilder":
        """Context manager entry for fluent API.
        
        Allows using PipelineBuilder as a context manager:
        
        Example:
            ```python
            with PipelineBuilder() as builder:
                builder.source("video", "s3://bucket/videos/")
                pipeline = builder.build()
            ```
        """
        return self
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit - no cleanup needed."""
        pass
    
    def __repr__(self) -> str:
        """String representation of builder.
        
        Returns:
            Human-readable builder description
        """
        sources_count = len(self.sources)
        stages_count = len(self.stages)
        output = self.output_path or "not set"
        gpu_enabled = self.config.get("enable_gpu", False)
        
        return (
            f"PipelineBuilder(sources={sources_count}, "
            f"stages={stages_count}, "
            f"output={output}, "
            f"gpu={'enabled' if gpu_enabled else 'disabled'})"
        )
    
    def __len__(self) -> int:
        """Get number of stages (supports len() built-in).
        
        Returns:
            Number of stages added to builder
            
        Example:
            ```python
            builder = PipelineBuilder().source(...).profile(...)
            num_stages = len(builder)  # Standard Python len() support
            ```
        """
        return len(self.stages)
    
    def __iter__(self) -> Iterator[PipelineStage]:
        """Iterate over stages (supports iter() and for loops).
        
        Yields:
            PipelineStage instances
            
        Example:
            ```python
            builder = PipelineBuilder().source(...).profile(...)
            for stage in builder:  # Standard Python iteration
                print(stage.name)
            ```
        """
        return iter(self.stages)
    
    def __bool__(self) -> bool:
        """Check if builder has sources (supports bool() and if statements).
        
        Returns:
            True if builder has at least one source, False otherwise
            
        Example:
            ```python
            builder = PipelineBuilder()
            if builder:  # Standard Python truthiness
                pipeline = builder.build()
            ```
        """
        return bool(self.sources)
    
    def __contains__(self, item: Any) -> bool:
        """Check if stage exists (supports 'in' operator).
        
        Args:
            item: Stage instance or stage name to search for
            
        Returns:
            True if stage found, False otherwise
            
        Example:
            ```python
            builder = PipelineBuilder().profile(...)
            if "data_profiler" in builder:  # Check if stage exists
                print("Profiler found")
            ```
        """
        if isinstance(item, str):
            # Check if stage name exists
            return any(getattr(stage, "name", None) == item for stage in self.stages)
        elif isinstance(item, PipelineStage):
            # Check if stage instance exists
            return item in self.stages
        return False


@contextmanager
def experiment(
    name: str,
    tracking_backend: str = "auto",
    tracking_uri: Optional[str] = None,
    tags: Optional[dict[str, str]] = None,
    project: Optional[str] = None,
    entity: Optional[str] = None,
    **kwargs: Any,
):
    """Context manager for experiment tracking.

    Supports both MLflow and Weights & Biases (wandb). Automatically tracks
    pipeline runs, parameters, and metrics.

    Args:
        name: Experiment name
        tracking_backend: Tracking backend ("mlflow", "wandb", "both", or "auto")
        tracking_uri: MLflow tracking URI (for MLflow backend)
        tags: Experiment tags
        project: W&B project name (for wandb backend)
        entity: W&B entity/team name (for wandb backend)
        **kwargs: Additional experiment configuration

    Example:
        ```python
        # MLflow (default)
        with experiment("groot_training_v1", tags={"model": "groot"}):
            pipeline = PipelineBuilder()...
            results = pipeline.run()
        
        # Weights & Biases
        with experiment(
            "groot_training_v1",
            tracking_backend="wandb",
            project="groot-pipeline",
            tags={"model": "groot"},
        ):
            pipeline = PipelineBuilder()...
            results = pipeline.run()
        
        # Both MLflow and W&B
        with experiment(
            "groot_training_v1",
            tracking_backend="both",
            project="groot-pipeline",
        ):
            pipeline = PipelineBuilder()...
            results = pipeline.run()
        ```
    """
    mlflow_run = None
    wandb_run = None
    
    # Determine which backends to use
    use_mlflow = tracking_backend in ("mlflow", "both", "auto")
    use_wandb = tracking_backend in ("wandb", "both")
    
    if tracking_backend == "auto":
        # Try wandb first, fall back to MLflow
        try:
            import wandb
            use_wandb = True
            use_mlflow = False
        except ImportError:
            use_wandb = False
            use_mlflow = True
    
    # Initialize MLflow
    if use_mlflow:
        try:
            import mlflow
            
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            
            mlflow.set_experiment(name)
            mlflow_run = mlflow.start_run(tags=tags)
            mlflow_run.__enter__()
            
            logger.info(f"Started MLflow experiment: {name}")
        except ImportError:
            logger.warning("MLflow not available, skipping MLflow tracking")
            use_mlflow = False
        except Exception as e:
            logger.warning(f"MLflow initialization failed: {e}, skipping MLflow tracking")
            use_mlflow = False
    
    # Initialize W&B
    if use_wandb:
        try:
            import wandb
            
            wandb_kwargs = {
                "name": name,
                "tags": list(tags.values()) if tags else [],
                **kwargs,
            }
            if project:
                wandb_kwargs["project"] = project
            if entity:
                wandb_kwargs["entity"] = entity
            
            wandb_run = wandb.init(**wandb_kwargs)
            logger.info(f"Started W&B experiment: {name} (project: {project or 'default'})")
        except ImportError:
            logger.warning("wandb not available, skipping W&B tracking")
            use_wandb = False
            wandb_run = None
        except Exception as e:
            logger.warning(f"W&B initialization failed: {e}, skipping W&B tracking")
            use_wandb = False
            wandb_run = None
    
    try:
        yield {
            "mlflow": mlflow_run if use_mlflow else None,
            "wandb": wandb_run if use_wandb else None,
        }
    finally:
        # Cleanup MLflow
        if mlflow_run:
            try:
                mlflow_run.__exit__(None, None, None)
            except Exception as e:
                logger.warning(f"MLflow cleanup failed: {e}")
        
        # Cleanup W&B
        if wandb_run:
            try:
                wandb.finish()
            except Exception as e:
                logger.warning(f"W&B cleanup failed: {e}")


def run_pipeline(
    pipeline: Any,
    experiment_name: Optional[str] = None,
    tracking_backend: str = "auto",
    **run_kwargs: Any,
) -> Any:
    """Run pipeline with optional experiment tracking.

    Supports both MLflow and Weights & Biases.

    Args:
        pipeline: Pipeline instance to run
        experiment_name: Optional experiment name for tracking
        tracking_backend: Tracking backend ("mlflow", "wandb", "both", or "auto")
        **run_kwargs: Additional run configuration (tags, project, entity, etc.)

    Returns:
        Pipeline execution results

    Example:
        ```python
        # MLflow tracking
        results = run_pipeline(
            pipeline,
            experiment_name="experiment_1",
            tracking_backend="mlflow",
        )
        
        # W&B tracking
        results = run_pipeline(
            pipeline,
            experiment_name="experiment_1",
            tracking_backend="wandb",
            project="my-project",
        )
        ```
    """
    if experiment_name:
        with experiment(experiment_name, tracking_backend=tracking_backend, **run_kwargs):
            return pipeline.run()
    else:
        return pipeline.run()

