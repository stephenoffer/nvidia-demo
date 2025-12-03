"""High-level declarative API for multimodal data pipeline.

Supports both Python and YAML-based configuration for easy pipeline setup.
Provides single-pipeline and multi-pipeline orchestration similar to modern
multimodal ETL tools (Airflow, Dagster, Prefect, NVIDIA NeMo Data Curator),
while tightly integrating with Ray Data and NVIDIA robotics tooling such as
Isaac Lab and Cosmos (GR00T Dreams).
"""

import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

import yaml  # https://pyyaml.org/

from pipeline.api.multipipeline import MultiPipelineRunner
from pipeline.api.types import CombineConfig, PipelineSpecConfig
from pipeline.config import PipelineConfig
from pipeline.core import MultimodalPipeline

logger = logging.getLogger(__name__)


class Pipeline:
    """High-level declarative API for multimodal data pipeline.

    Provides a simple, declarative interface for configuring and running
    data curation pipelines. Supports both programmatic (Python) and
    declarative (YAML) configuration.

    Example:
        ```python
        # Python API
        pipeline = Pipeline(
            sources=[
                {"type": "video", "path": "s3://bucket/videos/"},
                {"type": "mcap", "path": "s3://bucket/rosbags/"},
            ],
            output="s3://bucket/curated/",
            enable_gpu=True,
        )
        results = pipeline.run()
        ```

        ```yaml
        # YAML API
        sources:
          - type: video
            path: s3://bucket/videos/
          - type: mcap
            path: s3://bucket/rosbags/
        output: s3://bucket/curated/
        enable_gpu: true
        ```
    """

    def __init__(
        self,
        sources: List[Dict[str, Any]],
        output: str,
        enable_gpu: bool = False,
        num_gpus: int = 0,
        num_cpus: Optional[int] = None,
        compute_mode: str = "auto",
        **kwargs: Any,
    ):
        """Initialize declarative pipeline.

        Args:
            sources: List of data source configurations
                    Each source dict should have 'type' and 'path' keys
            output: Output path for curated data
            enable_gpu: Enable GPU acceleration
            num_gpus: Number of GPUs to use
            num_cpus: Number of CPUs to use (auto if None)
            **kwargs: Additional configuration options
        """
        compute_mode = (compute_mode or ("gpu" if enable_gpu else "auto")).lower()
        if compute_mode not in {"auto", "cpu", "gpu"}:
            raise ValueError("compute_mode must be 'auto', 'cpu', or 'gpu'")

        if compute_mode == "cpu":
            enable_gpu = False
            num_gpus = 0
        elif compute_mode == "gpu":
            enable_gpu = True
            if num_gpus <= 0:
                num_gpus = 1
        else:  # auto
            if num_gpus > 0:
                enable_gpu = True
            else:
                enable_gpu = False
                num_gpus = 0

        self.sources = sources
        self.output = output
        self.enable_gpu = enable_gpu
        self.num_gpus = num_gpus
        self.num_cpus = num_cpus
        self.compute_mode = compute_mode
        self.kwargs = kwargs

        # Build PipelineConfig from declarative config
        # Extract paths and validate
        input_paths = []
        for i, source in enumerate(sources):
            if not isinstance(source, dict):
                raise TypeError(f"Source {i} must be a dictionary")
            if "path" not in source:
                raise ValueError(f"Source {i} missing required 'path' field")
            path = source["path"]
            if not isinstance(path, str):
                raise TypeError(f"Source {i} path must be a string")
            input_paths.append(path)

        if not input_paths:
            raise ValueError("At least one source path must be provided")

        self.config = PipelineConfig(
            input_paths=input_paths,
            output_path=output,
            enable_gpu_dedup=enable_gpu,
            num_gpus=num_gpus,
            num_cpus=num_cpus,
            compute_mode=compute_mode,
            **kwargs,
        )

        # Create pipeline instance
        self.pipeline = MultimodalPipeline(self.config)

        # Configure data sources based on types
        self._configure_sources()
        
        # Store MLflow tracking URI if provided
        self.mlflow_tracking_uri = kwargs.get("mlflow_tracking_uri")
    
    @property
    def status(self) -> Dict[str, Any]:
        """Get current pipeline status (property accessor).
        
        Returns:
            Dictionary with pipeline status information
            
        Example:
            ```python
            pipeline = Pipeline(...)
            print(pipeline.status)  # Access as property
            ```
        """
        return self.get_status()
    
    @property
    def stage_names(self) -> List[str]:
        """Get list of stage names (property accessor).
        
        Returns:
            List of stage names in execution order
            
        Example:
            ```python
            pipeline = Pipeline(...)
            print(pipeline.stage_names)  # Access as property
            ```
        """
        return self.get_stage_names()
    
    def __repr__(self) -> str:
        """String representation of pipeline.
        
        Returns:
            Human-readable pipeline description
        """
        sources_str = ", ".join([f"{s.get('type', 'auto')}:{s.get('path', 'N/A')}" for s in self.sources[:3]])
        if len(self.sources) > 3:
            sources_str += f", ... (+{len(self.sources) - 3} more)"
        
        return (
            f"Pipeline(sources=[{sources_str}], "
            f"output={self.output}, "
            f"gpu={'enabled' if self.enable_gpu else 'disabled'}, "
            f"num_gpus={self.num_gpus})"
        )
    
    def __len__(self) -> int:
        """Get number of sources (supports len() built-in).
        
        Returns:
            Number of data sources
            
        Example:
            ```python
            pipeline = Pipeline(...)
            num_sources = len(pipeline)  # Standard Python len() support
            ```
        """
        return len(self.sources)
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over sources (supports iter() and for loops).
        
        Yields:
            Source configuration dictionaries
            
        Example:
            ```python
            pipeline = Pipeline(...)
            for source in pipeline:  # Standard Python iteration
                print(source["path"])
            ```
        """
        return iter(self.sources)
    
    def __bool__(self) -> bool:
        """Check if pipeline is configured (supports bool() and if statements).
        
        Returns:
            True if pipeline has sources and output configured, False otherwise
            
        Example:
            ```python
            pipeline = Pipeline(...)
            if pipeline:  # Standard Python truthiness
                pipeline.run()
            ```
        """
        return bool(self.sources) and bool(self.output)
    
    def __contains__(self, item: Any) -> bool:
        """Check if source exists (supports 'in' operator).
        
        Args:
            item: Source path (str) or source dict to search for
            
        Returns:
            True if source found, False otherwise
            
        Example:
            ```python
            pipeline = Pipeline(...)
            if "s3://bucket/data/" in pipeline:  # Check if path exists
                print("Source found")
            ```
        """
        if isinstance(item, str):
            # Check if path exists in sources
            return any(source.get("path") == item for source in self.sources)
        elif isinstance(item, dict):
            # Check if source dict exists
            return item in self.sources
        return False
    
    def __add__(self, other: "Pipeline") -> "Pipeline":
        """Combine pipelines (supports + operator).
        
        Creates a new pipeline that combines sources from both pipelines.
        Useful for building pipelines incrementally.
        
        Args:
            other: Other Pipeline to combine with
            
        Returns:
            New Pipeline with combined sources
            
        Raises:
            TypeError: If other is not a Pipeline
            
        Example:
            ```python
            pipeline1 = Pipeline(...)
            pipeline2 = Pipeline(...)
            combined = pipeline1 + pipeline2  # Combine sources
            ```
        """
        if not isinstance(other, Pipeline):
            return NotImplemented
        
        # Combine sources
        combined_sources = self.sources + other.sources
        
        # Use output from first pipeline (or could raise error if different)
        if self.output != other.output and other.output:
            logger.warning(
                f"Combining pipelines with different outputs: "
                f"{self.output} and {other.output}. Using {self.output}."
            )
        
        # Merge configurations (prefer self's config)
        combined_config = {**other.kwargs, **self.kwargs}
        
        # Remove config keys that are passed as explicit arguments
        combined_config.pop("enable_gpu", None)
        combined_config.pop("num_gpus", None)
        combined_config.pop("num_cpus", None)
        combined_config.pop("compute_mode", None)
        
        return Pipeline(
            sources=combined_sources,
            output=self.output,
            enable_gpu=self.enable_gpu or other.enable_gpu,
            num_gpus=max(self.num_gpus, other.num_gpus),
            num_cpus=self.num_cpus or other.num_cpus,
            compute_mode=self.compute_mode,
            **combined_config,
        )

    @classmethod
    def quick_start(
        cls,
        input_paths: Union[str, List[str]],
        output_path: str,
        enable_gpu: bool = False,
        num_gpus: int = 0,
        **kwargs: Any,
    ) -> "Pipeline":
        """Quick start constructor for simple pipeline creation.

        Creates a pipeline with minimal configuration for common use cases.
        
        This is the simplest way to create a pipeline - just provide input and output paths.
        The pipeline will auto-detect data formats and use sensible defaults.

        Args:
            input_paths: Single path or list of input paths
            output_path: Output path for processed data
            enable_gpu: Enable GPU acceleration (deprecated, use num_gpus instead)
            num_gpus: Number of GPUs to use (0 = CPU only, overrides enable_gpu)
            **kwargs: Additional pipeline configuration

        Returns:
            Configured Pipeline instance

        Example:
            ```python
            # Single path
            pipeline = Pipeline.quick_start(
                input_paths="s3://bucket/data/",
                output_path="s3://bucket/output/",
            )
            results = pipeline.run()
            
            # Multiple paths with GPU
            pipeline = Pipeline.quick_start(
                input_paths=["s3://bucket/videos/", "s3://bucket/rosbags/"],
                output_path="s3://bucket/output/",
                num_gpus=4,
            )
            results = pipeline.run()
            ```
        """
        if isinstance(input_paths, str):
            input_paths = [input_paths]
        
        if not input_paths:
            raise ValueError("At least one input path must be provided")
        
        # Use num_gpus if provided, otherwise use enable_gpu
        if num_gpus > 0:
            enable_gpu = True
        elif enable_gpu and num_gpus == 0:
            num_gpus = 1  # Default to 1 GPU if enable_gpu=True but num_gpus not specified
        
        sources = [{"type": "auto", "path": path} for path in input_paths]
        
        # Set sensible defaults if not provided
        if "num_cpus" not in kwargs:
            kwargs["num_cpus"] = None  # Auto-detect
        if "batch_size" not in kwargs:
            kwargs["batch_size"] = 1000  # Default batch size
        
        return cls(
            sources=sources,
            output=output_path,
            enable_gpu=enable_gpu,
            num_gpus=num_gpus,
            **kwargs,
        )

    @classmethod
    def cpu_demo(
        cls,
        sources: List[Dict[str, Any]],
        output: str,
        **kwargs: Any,
    ) -> "Pipeline":
        """Convenience constructor for CPU-only local demos.
        
        Args:
            sources: List of data source configurations
            output: Output path for curated data
            **kwargs: Additional configuration options
            
        Returns:
            Configured Pipeline instance for CPU-only execution
            
        Example:
            ```python
            pipeline = Pipeline.cpu_demo(
                sources=[{"type": "video", "path": "local/videos/"}],
                output="local/output/",
            )
            results = pipeline.run()
            ```
        """
        return cls(
            sources=sources,
            output=output,
            enable_gpu=False,
            num_gpus=0,
            compute_mode="cpu",
            **kwargs,
        )
    
    def get_status(self) -> Dict[str, Any]:
        """Get current pipeline status.
        
        Returns:
            Dictionary with pipeline status information
        """
        status = self.pipeline.get_status()
        status["num_sources"] = len(self.sources)
        status["output_path"] = self.output
        status["compute_mode"] = self.compute_mode
        status["gpu_enabled"] = self.enable_gpu
        status["num_gpus"] = self.num_gpus
        return status
    
    def get_stage_names(self) -> List[str]:
        """Get list of stage names.
        
        Returns:
            List of stage names in execution order
        """
        return self.pipeline.get_stage_names()

    def _configure_sources(self) -> None:
        """Configure data sources based on their types."""
        if not isinstance(self.sources, list):
            raise TypeError("sources must be a list")

        for i, source in enumerate(self.sources):
            if not isinstance(source, dict):
                raise TypeError(f"Source {i} must be a dictionary")

            if "path" not in source:
                raise ValueError(f"Source {i} missing required 'path' field")

            source_type = source.get("type", "auto")
            path = source["path"]

            if not isinstance(path, str):
                raise TypeError(f"Source {i} path must be a string, got {type(path)}")

            # Configure NVIDIA integrations
            if source_type == "isaac_lab":
                from pipeline.integrations.isaac_lab import IsaacLabLoader

                loader = IsaacLabLoader(
                    simulation_path=path,
                    robot_type=source.get("robot_type", "humanoid"),
                    include_observations=source.get("include_observations", True),
                    include_actions=source.get("include_actions", True),
                    include_rewards=source.get("include_rewards", True),
                )
                self.pipeline.add_simulation_data(loader)

            elif source_type == "cosmos_dreams":
                from pipeline.integrations.cosmos import CosmosDreamsLoader

                loader = CosmosDreamsLoader(
                    dreams_path=path,
                    include_metadata=source.get("include_metadata", True),
                )
                self.pipeline.add_synthetic_data(loader)

            # Note: MCAP uses Ray Data's native read_mcap() with predicate pushdown
            # Other source types are handled by MultimodalLoader which auto-detects formats

    def run(
        self,
        experiment_name: Optional[str] = None,
        tags: Optional[dict[str, str]] = None,
        tracking_backend: str = "auto",
        project: Optional[str] = None,
        entity: Optional[str] = None,
        **run_kwargs: Any,
    ) -> Dict[str, Any]:
        """Run the pipeline.

        Supports both MLflow and Weights & Biases for experiment tracking.

        Args:
            experiment_name: Optional experiment name for tracking
            tags: Optional tags for experiment tracking
            tracking_backend: Tracking backend ("mlflow", "wandb", "both", or "auto")
            project: W&B project name (for wandb backend)
            entity: W&B entity/team name (for wandb backend)
            **run_kwargs: Additional run configuration

        Returns:
            Dictionary containing execution results and metrics

        Raises:
            ValueError: If no sources are configured
            RuntimeError: If pipeline execution fails
            
        Example:
            ```python
            # Simple run
            pipeline = Pipeline(
                sources=[{"type": "video", "path": "s3://bucket/videos/"}],
                output="s3://bucket/curated/",
            )
            results = pipeline.run()
            
            # With MLflow tracking
            results = pipeline.run(
                experiment_name="groot_training_v1",
                tracking_backend="mlflow",
                tags={"model": "groot", "version": "1.0"},
            )
            
            # With W&B tracking
            results = pipeline.run(
                experiment_name="groot_training_v1",
                tracking_backend="wandb",
                project="groot-pipeline",
                tags={"model": "groot"},
            )
            
            # With both MLflow and W&B
            results = pipeline.run(
                experiment_name="groot_training_v1",
                tracking_backend="both",
                project="groot-pipeline",
                tags={"model": "groot"},
            )
            ```
        """
        if not self.sources:
            raise ValueError("No sources configured for pipeline")

        logger.info(f"Running declarative pipeline with {len(self.sources)} sources")
        
        if experiment_name:
            # Use unified experiment tracking
            from pipeline.api.fluent import experiment
            
            try:
                with experiment(
                    experiment_name,
                    tracking_backend=tracking_backend,
                    tracking_uri=getattr(self, "mlflow_tracking_uri", None),
                    tags=tags,
                    project=project,
                    entity=entity,
                    **run_kwargs,
                ):
                    # Log parameters to both backends
                    params = {
                        "num_sources": len(self.sources),
                        "num_gpus": self.num_gpus,
                        "compute_mode": self.compute_mode,
                    }
                    
                    # Log to MLflow
                    try:
                        import mlflow
                        mlflow.log_params(params)
                    except (ImportError, Exception):
                        pass
                    
                    # Log to W&B
                    try:
                        import wandb
                        wandb.config.update(params)
                    except (ImportError, Exception):
                        pass
                    
                    results = self.pipeline.run()
                    
                    # Log metrics to both backends
                    if isinstance(results, dict):
                        metrics = results.get("metrics", {})
                        
                        # Log to MLflow
                        try:
                            import mlflow
                            mlflow.log_metrics(metrics)
                        except (ImportError, Exception):
                            pass
                        
                        # Log to W&B
                        try:
                            import wandb
                            wandb.log(metrics)
                        except (ImportError, Exception):
                            pass
                    
                    logger.info("Pipeline execution completed successfully")
                    return results
            except Exception as e:
                logger.warning(f"Experiment tracking failed: {e}, continuing without tracking")
                results = self.pipeline.run()
                logger.info("Pipeline execution completed successfully")
                return results
        else:
            try:
                results = self.pipeline.run()
                logger.info("Pipeline execution completed successfully")
                return results
            except Exception as e:
                logger.error(f"Pipeline execution failed: {e}", exc_info=True)
                raise RuntimeError(f"Pipeline execution failed: {e}") from e

    def to_yaml(self, output_path: str) -> None:
        """Export pipeline configuration to YAML.

        Args:
            output_path: Path to write YAML configuration

        Raises:
            ValueError: If output_path is invalid
            IOError: If file cannot be written
        """
        if not output_path:
            raise ValueError("output_path must be specified")

        try:
            config_dict = {
                "sources": self.sources,
                "output": self.output,
                "enable_gpu": self.enable_gpu,
                "num_gpus": self.num_gpus,
                "compute_mode": self.compute_mode,
            }
            if self.num_cpus:
                config_dict["num_cpus"] = self.num_cpus
            config_dict.update(self.kwargs)

            output_path_obj = Path(output_path)
            # Create parent directory if it doesn't exist
            output_path_obj.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path_obj, "w") as f:
                yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

            logger.info(f"Exported pipeline configuration to {output_path}")
        except (IOError, OSError, ValueError, yaml.YAMLError) as e:
            logger.error(f"Failed to export YAML config to {output_path}: {e}", exc_info=True)
            raise IOError(f"Failed to export YAML config: {e}") from e
    
    def __enter__(self) -> "Pipeline":
        """Context manager entry.
        
        Returns:
            Self for use in 'with' statement
        """
        return self
    
    def __exit__(self, exc_type: Optional[type[Exception]], exc_val: Optional[Exception], exc_tb: Optional[Any]) -> None:
        """Context manager exit.
        
        Ensures cleanup happens even if exceptions occur.
        
        Args:
            exc_type: Exception type if exception occurred
            exc_val: Exception value if exception occurred
            exc_tb: Exception traceback if exception occurred
        """
        try:
            self.pipeline.shutdown()
        except Exception:
            pass


def load_from_yaml(yaml_path: str) -> Pipeline:
    """Load pipeline configuration from YAML file.

    Args:
        yaml_path: Path to YAML configuration file

    Returns:
        Configured Pipeline instance

    Example YAML:
        ```yaml
        sources:
          - type: video
            path: s3://bucket/videos/
          - type: mcap
            path: s3://bucket/rosbags/
            topics: ["/camera/image_raw", "/imu/data"]
          - type: hdf5
            path: s3://bucket/sensor_data/
        output: s3://bucket/curated/
        enable_gpu: true
        num_gpus: 4
        ```
    """
    yaml_path_obj = Path(yaml_path)
    if not yaml_path_obj.exists():
        raise FileNotFoundError(f"YAML config file not found: {yaml_path}")

    with open(yaml_path_obj) as f:
        config_dict = yaml.safe_load(f)

    if not config_dict:
        raise ValueError(f"Empty or invalid YAML configuration: {yaml_path}")

    # Multi-pipeline configuration (pipelines + optional multi_pipeline combine)
    if "pipelines" in config_dict:
        pipeline_specs_raw = config_dict.pop("pipelines") or []
        if not pipeline_specs_raw:
            raise ValueError("pipelines list cannot be empty")

        pipeline_defaults = config_dict.pop("pipeline_defaults", {})
        combine_cfg_raw = config_dict.pop(
            "multi_pipeline", config_dict.pop("combine", None)
        )
        multi_name = config_dict.pop("name", "multi_pipeline")

        normalized_specs = [
            PipelineSpecConfig.from_dict(raw_spec, pipeline_defaults, idx)
            for idx, raw_spec in enumerate(pipeline_specs_raw)
        ]

        return MultiPipelineRunner(
            pipeline_specs=normalized_specs,
            combine_config=CombineConfig.from_dict(combine_cfg_raw),
            name=multi_name,
        )

    # Single-pipeline configuration
    sources = config_dict.pop("sources", [])
    if not sources:
        raise ValueError("YAML config must contain 'sources' list")

    output = config_dict.pop("output", None)
    if not output:
        raise ValueError("YAML config must contain 'output' path")

    # Extract compute / resource settings
    compute_mode = config_dict.pop("compute_mode", "auto")
    enable_gpu = config_dict.pop("enable_gpu", compute_mode == "gpu")
    num_gpus = config_dict.pop("num_gpus", 1 if enable_gpu else 0)
    num_cpus = config_dict.pop("num_cpus", None)

    # Remaining kwargs are passed as additional config
    return Pipeline(
        sources=sources,
        output=output,
        enable_gpu=enable_gpu,
        num_gpus=num_gpus,
        num_cpus=num_cpus,
        compute_mode=compute_mode,
        **config_dict,
    )

