"""Main pipeline orchestrator.

Coordinates data loading, stage execution, and output generation.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Iterator, Optional

import ray
from ray.data import Dataset

from pipeline.config import PipelineConfig
from pipeline.config.ray_data import RayDataConfig
from pipeline.core.execution import PipelineExecutor
from pipeline.core.lifecycle import PipelineLifecycleManager
from pipeline.core.visualization import PipelineVisualizationManager
from pipeline.dedup.gpu_dedup import GPUDeduplicator
from pipeline.exceptions import (
    CheckpointError,
    ConfigurationError,
    DataLoadError,
    MetricsError,
    PipelineError,
    RayError,
    StorageError,
)
from pipeline.integrations.cosmos import CosmosDreamsLoader
from pipeline.integrations.isaac_lab import IsaacLabLoader
from pipeline.loaders.multimodal import MultimodalLoader
from pipeline.observability.metrics import PipelineMetrics
from pipeline.stages import (
    CompletenessValidator,
    CrossModalValidator,
    DataQualityScorer,
    EpisodeBoundaryDetector,
    GPUAnalyticsStage,
    InstructionGroundingStage,
    PhysicsValidator,
    SensorProcessor,
    SequenceNormalizer,
    TemporalAlignmentStage,
    TextProcessor,
    TransitionAlignmentStage,
    VideoProcessor,
)
from pipeline.stages.base import PipelineStage

logger = logging.getLogger(__name__)


class MultimodalPipeline:
    """Main pipeline orchestrator for multimodal data curation.

    Coordinates data loading, processing stages, and output generation.
    """

    def __init__(self, config: PipelineConfig) -> None:
        """Initialize the multimodal pipeline.

        Args:
            config: Pipeline configuration
        """
        # Sanitize and validate configuration
        from pipeline.utils.validation.input_validation import InputValidator

        sanitizer = InputValidator(strict=True)
        try:
            is_valid, errors = sanitizer.validate_config(config.__dict__)
            if not is_valid:
                error_msg = "; ".join(errors)
                logger.error(f"Configuration validation failed: {error_msg}")
                raise ValueError(f"Invalid configuration: {error_msg}")
        except (ValueError, TypeError) as e:
            logger.error(f"Configuration sanitization failed: {e}")
            raise

        config.validate()
        self.config = config

        self.lifecycle = PipelineLifecycleManager(config)
        self.lifecycle.initialize()

        self._configure_ray_data(config)

        self.loader = MultimodalLoader(config)
        self.metrics = PipelineMetrics(
            enabled=config.enable_observability,
            enable_prometheus=getattr(config, 'enable_prometheus', True),
        )
        self.executor = PipelineExecutor(config, self.metrics)
        self.visualization = PipelineVisualizationManager()

        self.stages: list[PipelineStage] = []
        self._setup_default_stages()

        self.simulation_loaders: list[IsaacLabLoader] = []
        self.synthetic_loaders: list[CosmosDreamsLoader] = []
        self.omniverse_loaders: list[Any] = []  # TODO: Create OmniverseLoader type
        self.synthetic_generators: list[Any] = []  # TODO: Create SyntheticGenerator type

    def _configure_ray_data(self, config: PipelineConfig) -> None:
        """Configure Ray Data execution options.
        
        Note: This method is kept for backward compatibility but Ray Data context
        is now configured in RayContext.initialize() with best practices including
        eager_free=True. This method will be deprecated in favor of using
        RayContext directly.

        Args:
            config: Pipeline configuration
        """
        # Ray Data context is now configured in RayContext.initialize()
        # This method is kept for backward compatibility
        logger.info("Ray Data context configuration is handled by RayContext.initialize()")
        
        # Verify context is configured
        try:
            from pipeline.utils.ray.context import RayContext
            ctx = RayContext.get_data_context()
            if ctx:
                logger.info(f"Ray Data context verified: eager_free={ctx.eager_free}")
        except (AttributeError, ImportError, RayError) as e:
            logger.warning(f"Could not verify Ray Data context: {e}")

    def _setup_default_stages(self) -> None:
        """Set up default processing stages."""
        self.stages.append(EpisodeBoundaryDetector())
        self.stages.append(TemporalAlignmentStage())
        self.stages.append(CompletenessValidator(allow_partial=True))
        self.stages.append(VideoProcessor(config=self.config, **self.config.video_config))
        self.stages.append(TextProcessor(**self.config.text_config))
        self.stages.append(SensorProcessor(**self.config.sensor_config))
        self.stages.append(InstructionGroundingStage())
        self.stages.append(PhysicsValidator(reject_invalid=False))
        self.stages.append(CrossModalValidator())
        self.stages.append(DataQualityScorer(filter_low_quality=False))
        self.stages.append(SequenceNormalizer())
        self.stages.append(TransitionAlignmentStage())

        gpu_cfg = self.config.gpu_analytics_config or {}
        if gpu_cfg.get("enabled"):
            self.stages.append(
                GPUAnalyticsStage(
                    target_columns=gpu_cfg.get("target_columns", []),
                    metrics=gpu_cfg.get("metrics", ["mean", "std"]),
                    normalize=gpu_cfg.get("normalize", True),
                    num_gpus=gpu_cfg.get("num_gpus", 1),
                )
            )

        if self.config.enable_gpu_dedup:
            self.stages.append(
                GPUDeduplicator(
                    method=self.config.dedup_method,
                    similarity_threshold=self.config.similarity_threshold,
                    num_gpus=self.config.num_gpus,
                )
            )

    def add_stage(self, stage: PipelineStage) -> "MultimodalPipeline":
        """Add a custom processing stage to the pipeline.

        Args:
            stage: Processing stage instance
            
        Returns:
            Self for method chaining
            
        Raises:
            TypeError: If stage is not a PipelineStage instance
            
        Example:
            ```python
            pipeline = MultimodalPipeline(config)
            pipeline.add_stage(custom_stage).add_stage(another_stage)
            ```
        """
        if not isinstance(stage, PipelineStage):
            raise TypeError(f"stage must be a PipelineStage instance, got {type(stage)}")
        self.stages.append(stage)
        return self

    def add_simulation_data(self, loader: IsaacLabLoader) -> "MultimodalPipeline":
        """Add Isaac Lab simulation data loader.

        Args:
            loader: Isaac Lab loader instance
            
        Returns:
            Self for method chaining
            
        Raises:
            TypeError: If loader is not an IsaacLabLoader instance
            
        Example:
            ```python
            from pipeline.integrations.isaac_lab import IsaacLabLoader
            
            loader = IsaacLabLoader(simulation_path="path/to/data")
            pipeline.add_simulation_data(loader).run()
            ```
        """
        if not isinstance(loader, IsaacLabLoader):
            raise TypeError(f"loader must be an IsaacLabLoader instance, got {type(loader)}")
        self.simulation_loaders.append(loader)
        logger.info(f"Added Isaac Lab simulation data loader: {loader.robot_type}")
        return self

    def add_synthetic_data(self, loader: CosmosDreamsLoader) -> "MultimodalPipeline":
        """Add Cosmos Dreams synthetic data loader.

        Args:
            loader: Cosmos Dreams loader instance
            
        Returns:
            Self for method chaining
            
        Raises:
            TypeError: If loader is not a CosmosDreamsLoader instance
            
        Example:
            ```python
            from pipeline.integrations.cosmos import CosmosDreamsLoader
            
            loader = CosmosDreamsLoader(dreams_path="path/to/dreams")
            pipeline.add_synthetic_data(loader).run()
            ```
        """
        if not isinstance(loader, CosmosDreamsLoader):
            raise TypeError(f"loader must be a CosmosDreamsLoader instance, got {type(loader)}")
        self.synthetic_loaders.append(loader)
        logger.info(f"Added Cosmos Dreams synthetic data loader: {loader.model_name}")
        return self

    def add_omniverse_data(self, loader: Any) -> None:  # TODO: Create OmniverseLoader type
        """Add Omniverse USD/Replicator data loader.

        Args:
            loader: OmniverseLoader instance
        """
        from pipeline.integrations.omniverse import OmniverseLoader
        
        if not isinstance(loader, OmniverseLoader):
            raise TypeError(f"Loader must be OmniverseLoader, got {type(loader)}")
        
        self.omniverse_loaders.append(loader)
        logger.info(f"Added Omniverse data loader: {loader.omniverse_path}")

    def add_synthetic_generator(self, generator: Any) -> None:  # TODO: Create SyntheticGenerator type
        """Add a synthetic data generator.

        Args:
            generator: Synthetic data generator instance
            
        Raises:
            TypeError: If generator is None or invalid
        """
        if generator is None:
            raise TypeError("generator cannot be None")
        if not hasattr(generator, 'generate'):
            raise TypeError(f"generator must have a 'generate' method, got {type(generator)}")
        self.synthetic_generators.append(generator)
        logger.info(f"Added synthetic data generator: {generator.__class__.__name__}")

    def enable_visualization(
        self,
        video_resolution: tuple[int, int] = (1280, 720),
        video_fps: int = 30,
        dashboard_mode: str = "local",
    ) -> "MultimodalPipeline":
        """Enable visualization features.

        Args:
            video_resolution: Resolution for generated videos (width, height)
            video_fps: Frames per second for videos
            dashboard_mode: Dashboard mode ('local' or 'web')
            
        Returns:
            Self for method chaining
            
        Raises:
            ValueError: If video_resolution or video_fps are invalid
            
        Example:
            ```python
            pipeline.enable_visualization(
                video_resolution=(1920, 1080),
                video_fps=60
            ).run()
            ```
        """
        if not isinstance(video_resolution, tuple) or len(video_resolution) != 2:
            raise ValueError(f"video_resolution must be a tuple of (width, height), got {video_resolution}")
        if not all(isinstance(dim, int) and dim > 0 for dim in video_resolution):
            raise ValueError(f"video_resolution dimensions must be positive integers, got {video_resolution}")
        if not isinstance(video_fps, int) or video_fps <= 0:
            raise ValueError(f"video_fps must be a positive integer, got {video_fps}")
        if dashboard_mode not in {"local", "web"}:
            raise ValueError(f"dashboard_mode must be 'local' or 'web', got {dashboard_mode}")
        self.visualization.enable_visualization(video_resolution, video_fps, dashboard_mode)
        return self

    def run(self, resume_from_checkpoint: Optional[str] = None) -> dict[str, Any]:
        """Execute the complete pipeline.

        Args:
            resume_from_checkpoint: Checkpoint name to resume from (None = start fresh)

        Returns:
            Dictionary containing pipeline results and metrics
            
        Raises:
            PipelineError: If pipeline execution fails
            CheckpointError: If checkpoint resume fails
            DataLoadError: If data loading fails
            StorageError: If output writing fails
            ValueError: If resume_from_checkpoint is provided but checkpoint_dir not configured
            
        Example:
            ```python
            from pipeline import MultimodalPipeline, PipelineConfig
            
            config = PipelineConfig(
                input_paths=["s3://bucket/input/"],
                output_path="s3://bucket/output/",
            )
            pipeline = MultimodalPipeline(config)
            results = pipeline.run()
            ```
        """
        if resume_from_checkpoint and not getattr(self.config, 'checkpoint_dir', None):
            raise ValueError("resume_from_checkpoint requires checkpoint_dir to be configured")
        
        logger.info("Starting multimodal data curation pipeline")
        
        # Set random seed for reproducibility if configured
        import random
        import numpy as np
        seed = getattr(self.config, 'random_seed', None)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            logger.info(f"Set random seed to {seed} for reproducibility")
        
        self.metrics.start()

        # Resume from checkpoint if specified
        dataset = None
        start_stage_index = 0
        if resume_from_checkpoint:
            try:
                from pipeline.utils.execution.checkpoint import PipelineCheckpoint
                
                checkpoint_dir = getattr(self.config, 'checkpoint_dir', None)
                if not checkpoint_dir:
                    raise ValueError("checkpoint_dir not configured, cannot resume")
                
                checkpoint = PipelineCheckpoint(checkpoint_dir=checkpoint_dir)
                dataset, start_stage_index = checkpoint.resume_from_checkpoint(
                    checkpoint_name=resume_from_checkpoint,
                    stages=self.stages,
                )
                logger.info(f"Resuming from checkpoint: {resume_from_checkpoint}, stage {start_stage_index}")
            except (CheckpointError, StorageError, IOError, OSError) as e:
                logger.error(f"Failed to resume from checkpoint: {e}")
                raise CheckpointError(f"Failed to resume from checkpoint: {e}") from e

        # Create version record if not resuming
        if not resume_from_checkpoint:
            self._create_version_record()

        try:
            # Load data if not resuming
            if dataset is None:
                dataset = self._load_all_data()
                
                # Add data lineage tracking if enabled
                if getattr(self.config, 'enable_lineage_tracking', True):
                    from pipeline.integrations.openlineage import create_openlineage_tracker
                    
                    lineage_tracker = create_openlineage_tracker(enabled=True)
                    source_paths = self.config.input_paths or ["unknown"]
                    output_path = self.config.output_path or "unknown"
                    
                    run_id = lineage_tracker.track_lineage(
                        stage_name="data-loading",
                        input_paths=source_paths,
                        output_path=output_path,
                        metadata={"pipeline_config": str(self.config.__dict__)},
                    )
                    logger.info(f"Tracked data lineage: run_id={run_id}")
            
            # Execute remaining stages
            remaining_stages = self.stages[start_stage_index:]
            dataset = self.executor.execute_stages(dataset, remaining_stages)
            self.executor.write_output(dataset)

            results = self._collect_results()

            if self.config.enable_observability:
                self.visualization.generate_visualizations(results, dataset, self.config.output_path)
                
                # Export Grafana dashboard if enabled
                if getattr(self.config, 'enable_grafana', True):
                    from pipeline.observability.grafana import create_grafana_dashboard
                    from pathlib import Path
                    
                    dashboard_path = Path(self.config.output_path).parent / "grafana_dashboard.json"
                    create_grafana_dashboard(
                        datasource_name="Prometheus",
                        output_path=str(dashboard_path),
                    )
                    logger.info(f"Grafana dashboard exported to {dashboard_path}")

            logger.info("Pipeline completed successfully")
            return results

        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            self.metrics.record_error(str(e))
            if isinstance(e, PipelineError):
                raise
            raise PipelineError(f"Pipeline execution failed: {e}") from e
        finally:
            self.metrics.stop()

    def _create_version_record(self) -> None:
        """Create data version record using MLflow."""
        from pipeline.integrations.mlflow import create_mlflow_tracker
        
        mlflow_tracker = create_mlflow_tracker(enabled=True)
        if mlflow_tracker.enabled:
            mlflow_tracker.start_run()
            mlflow_tracker.log_data_version(
                input_paths=self.config.input_paths or [],
                output_path=self.config.output_path or "",
            )
            pipeline_config_dict = {
                "num_gpus": self.config.num_gpus,
                "num_cpus": self.config.num_cpus,
                "batch_size": self.config.batch_size,
                "dedup_method": self.config.dedup_method,
            }
            mlflow_tracker.log_params(pipeline_config_dict)
            logger.info("Logged data version to MLflow")

    def _load_all_data(self) -> Dataset:
        """Load data from all sources.

        Returns:
            Combined dataset from all sources
            
        Raises:
            DataLoadError: If data loading fails
        """
        logger.info(f"Loading data from {len(self.config.input_paths)} sources")
        
        dataset = self.loader.load(self.config.input_paths)
        dataset = self._load_simulation_data(dataset)
        dataset = self._load_synthetic_data(dataset)
        dataset = self._load_omniverse_data(dataset)
        dataset = self._load_synthetic_generators(dataset)

        return dataset

    def _load_simulation_data(self, dataset: Dataset) -> Dataset:
        """Load Isaac Lab simulation data.

        Args:
            dataset: Current dataset

        Returns:
            Dataset with simulation data added
            
        Raises:
            DataLoadError: If simulation data loading fails
        """
        if not self.simulation_loaders:
            return dataset

        simulation_datasets = []
        for loader in self.simulation_loaders:
            sim_ds = loader.load()
            simulation_datasets.append(sim_ds)

        return self._add_datasets(dataset, simulation_datasets, "Isaac Lab")

    def _load_synthetic_data(self, dataset: Dataset) -> Dataset:
        """Load Cosmos Dreams synthetic data.

        Args:
            dataset: Current dataset

        Returns:
            Dataset with synthetic data added
            
        Raises:
            DataLoadError: If synthetic data loading fails
        """
        if not self.synthetic_loaders:
            return dataset

        synthetic_datasets = []
        for loader in self.synthetic_loaders:
            synth_ds = loader.load()
            synthetic_datasets.append(synth_ds)

        return self._add_datasets(dataset, synthetic_datasets, "Cosmos Dreams")

    def _load_omniverse_data(self, dataset: Dataset) -> Dataset:
        """Load Omniverse USD and Replicator data.

        Args:
            dataset: Current dataset

        Returns:
            Dataset with Omniverse data added
            
        Raises:
            DataLoadError: If Omniverse data loading fails
        """
        if not self.omniverse_loaders:
            return dataset

        omniverse_datasets = []
        for loader in self.omniverse_loaders:
            from pathlib import Path
            import ray.data

            omniverse_items = []
            usd_files = list(Path(loader.omniverse_path).glob("*.usd")) + list(Path(loader.omniverse_path).glob("*.usda")) + list(Path(loader.omniverse_path).glob("*.usdc"))
            for usd_file in usd_files:
                items = loader.load_usd_scene(str(usd_file))
                omniverse_items.extend(items)
            
            replicator_path = Path(loader.omniverse_path) / "replicator_output"
            if replicator_path.exists():
                replicator_items = loader.load_replicator_data(str(replicator_path))
                omniverse_items.extend(replicator_items)

            if omniverse_items:
                omniverse_dataset = ray.data.from_items(omniverse_items)
                omniverse_datasets.append(omniverse_dataset)

        if omniverse_datasets:
            return self._add_datasets(dataset, omniverse_datasets, "Omniverse")
        
        return dataset

    def _load_synthetic_generators(self, dataset: Dataset) -> Dataset:
        """Load synthetic generator data.

        Args:
            dataset: Current dataset

        Returns:
            Dataset with generator data added
            
        Raises:
            DataLoadError: If synthetic generation fails
        """
        if not self.synthetic_generators:
            return dataset

        generator_datasets = []
        for generator in self.synthetic_generators:
            gen_ds = generator.generate()
            generator_datasets.append(gen_ds)

        return self._add_datasets(dataset, generator_datasets, "synthetic generators")

    def _add_datasets(
        self,
        main_dataset: Dataset,
        datasets: list[Dataset],
        source_name: str,
    ) -> Dataset:
        """Add multiple datasets to main dataset using Ray Data union.

        Args:
            main_dataset: Main dataset to add to
            datasets: List of datasets to add
            source_name: Name of data source for logging

        Returns:
            Combined dataset
        """
        if not datasets:
            return main_dataset

        logger.info(f"Loading {len(datasets)} {source_name} datasets")

        valid_datasets = [ds for ds in datasets if ds is not None]
        if not valid_datasets:
            logger.warning(f"No valid {source_name} datasets to add")
            return main_dataset

        all_datasets = [main_dataset] + valid_datasets
        if len(all_datasets) == 1:
            return all_datasets[0]

        return ray.data.union(*all_datasets)

    def get_metrics(self) -> dict[str, Any]:
        """Get current pipeline metrics.

        Returns:
            Dictionary of current metrics
        """
        return self.metrics.get_current_metrics()

    def _collect_results(self) -> dict[str, Any]:
        """Collect final pipeline results.

        Returns:
            Dictionary of pipeline results
        """
        results = self.metrics.finish()
        results["output_path"] = self.config.output_path
        results["total_stages"] = len(self.stages)
        return results

    def shutdown(self) -> None:
        """Shutdown pipeline and cleanup resources.
        
        Ensures all resources are properly cleaned up including:
        - Ray actors and tasks
        - GPU memory
        - File handles and temporary files
        - Health check servers
        """
        try:
            self.lifecycle.shutdown()
        except Exception as e:
            logger.error(f"Error during pipeline shutdown: {e}", exc_info=True)
            # Continue cleanup even if errors occur
    
    def __enter__(self) -> "MultimodalPipeline":
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
        self.shutdown()
    
    @classmethod
    def create(cls, input_paths: list[str], output_path: str, **kwargs: Any) -> "MultimodalPipeline":
        """Create a pipeline with a simple configuration.
        
        Convenience factory method for quick pipeline setup.
        
        Args:
            input_paths: List of input data paths
            output_path: Output path for curated data
            **kwargs: Additional configuration options (num_gpus, batch_size, etc.)
            
        Returns:
            Configured MultimodalPipeline instance
            
        Example:
            ```python
            pipeline = MultimodalPipeline.create(
                input_paths=["s3://bucket/input/"],
                output_path="s3://bucket/output/",
                num_gpus=4,
            )
            results = pipeline.run()
            ```
        """
        from pipeline.config import PipelineConfig
        
        config = PipelineConfig(
            input_paths=input_paths,
            output_path=output_path,
            **kwargs,
        )
        return cls(config)
    
    
    def get_status(self) -> dict[str, Any]:
        """Get current pipeline status.
        
        Returns:
            Dictionary with pipeline status information
        """
        return {
            "num_stages": len(self.stages),
            "output_path": self.config.output_path,
        }

