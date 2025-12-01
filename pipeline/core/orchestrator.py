"""Main pipeline orchestrator.

Coordinates data loading, stage execution, and output generation.
"""

from __future__ import annotations

import logging
from typing import Any

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
from pipeline.stages.completeness_validator import CompletenessValidator
from pipeline.stages.cross_modal_validator import CrossModalValidator
from pipeline.stages.episode_detector import EpisodeBoundaryDetector
from pipeline.stages.gpu_analytics import GPUAnalyticsStage
from pipeline.stages.instruction_grounding import InstructionGroundingStage
from pipeline.stages.physics_validator import PhysicsValidator
from pipeline.stages.quality_scorer import DataQualityScorer
from pipeline.stages.sensor import SensorProcessor
from pipeline.stages.sequence_normalizer import SequenceNormalizer
from pipeline.stages.temporal_alignment import TemporalAlignmentStage
from pipeline.stages.text import TextProcessor
from pipeline.stages.transition_alignment import TransitionAlignmentStage
from pipeline.stages.video import VideoProcessor

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
        from pipeline.utils.input_validation import InputValidator

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

    def add_stage(self, stage: PipelineStage) -> None:
        """Add a custom processing stage to the pipeline.

        Args:
            stage: Processing stage instance
        """
        self.stages.append(stage)

    def add_simulation_data(self, loader: IsaacLabLoader) -> None:
        """Add Isaac Lab simulation data loader.

        Args:
            loader: Isaac Lab loader instance
        """
        self.simulation_loaders.append(loader)
        logger.info(f"Added Isaac Lab simulation data loader: {loader.robot_type}")

    def add_synthetic_data(self, loader: CosmosDreamsLoader) -> None:
        """Add Cosmos Dreams synthetic data loader.

        Args:
            loader: Cosmos Dreams loader instance
        """
        self.synthetic_loaders.append(loader)
        logger.info(f"Added Cosmos Dreams synthetic data loader: {loader.model_name}")

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
        """
        self.synthetic_generators.append(generator)
        logger.info(f"Added synthetic data generator: {generator.__class__.__name__}")

    def enable_visualization(
        self,
        video_resolution: tuple = (1280, 720),
        video_fps: int = 30,
        dashboard_mode: str = "local",
    ) -> None:
        """Enable visualization features.

        Args:
            video_resolution: Resolution for generated videos
            video_fps: Frames per second for videos
            dashboard_mode: Dashboard mode ('local' or 'web')
        """
        self.visualization.enable_visualization(video_resolution, video_fps, dashboard_mode)

    def run(self, resume_from_checkpoint: Optional[str] = None) -> dict[str, Any]:
        """Execute the complete pipeline.

        Args:
            resume_from_checkpoint: Checkpoint name to resume from (None = start fresh)

        Returns:
            Dictionary containing pipeline results and metrics
        """
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
                from pipeline.utils.checkpoint import PipelineCheckpoint
                
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
                    try:
                        from pipeline.integrations.openlineage import create_openlineage_tracker
                        import uuid
                        
                        lineage_tracker = create_openlineage_tracker(enabled=True)
                        source_paths = self.config.input_paths or ["unknown"]
                        output_path = self.config.output_path or "unknown"
                        
                        # Track lineage for this pipeline run
                        run_id = lineage_tracker.track_lineage(
                            stage_name="data-loading",
                            input_paths=source_paths,
                            output_path=output_path,
                            metadata={"pipeline_config": str(self.config.__dict__)},
                        )
                        logger.info(f"Tracked data lineage: run_id={run_id}")
                    except ImportError:
                        logger.warning("OpenLineage not available, skipping lineage tracking")
                    except (IOError, OSError, RuntimeError) as e:
                        logger.warning(f"Failed to track lineage: {e}")
            
            # Execute remaining stages
            remaining_stages = self.stages[start_stage_index:]
            dataset = self.executor.execute_stages(dataset, remaining_stages)
            self.executor.write_output(dataset)

            results = self._collect_results()

            if self.config.enable_observability:
                self.visualization.generate_visualizations(results, dataset, self.config.output_path)
                
                # Export Grafana dashboard if enabled
                if getattr(self.config, 'enable_grafana', True):
                    try:
                        from pipeline.observability.grafana import create_grafana_dashboard
                        from pathlib import Path
                        
                        dashboard_path = Path(self.config.output_path).parent / "grafana_dashboard.json"
                        create_grafana_dashboard(
                            datasource_name="Prometheus",
                            output_path=str(dashboard_path),
                        )
                        logger.info(f"Grafana dashboard exported to {dashboard_path}")
                    except (IOError, OSError, RuntimeError) as e:
                        logger.warning(f"Failed to export Grafana dashboard: {e}")

            logger.info("Pipeline completed successfully")
            return results

        except (PipelineError, ValueError, RuntimeError, IOError, OSError) as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            self.metrics.record_error(str(e))
            if isinstance(e, PipelineError):
                raise
            raise PipelineError(f"Pipeline execution failed: {e}") from e

    def _create_version_record(self) -> None:
        """Create data version record using MLflow."""
        try:
            # Use MLflow for data versioning if available
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
            else:
                # Fallback: log warning if MLflow not available
                logger.warning("MLflow not available, skipping data versioning")
        except (ImportError, IOError, OSError, StorageError) as e:
            logger.warning(f"Failed to create version record: {e}")

    def _load_all_data(self) -> Dataset:
        """Load data from all sources with graceful degradation.

        Returns:
            Combined dataset from all sources
        """
        # Apply incremental processing if enabled
        input_paths = self.config.input_paths
        enable_incremental = getattr(self.config, "enable_incremental_processing", True)
        
        if enable_incremental:
            try:
                from pipeline.utils.incremental import create_incremental_processor
                from pathlib import Path
                
                state_dir = str(Path(self.config.output_path).parent / ".incremental_state")
                incremental_processor = create_incremental_processor(
                    state_dir=state_dir,
                    enable_incremental=True,
                )
                
                # Filter to only unprocessed files
                input_paths = incremental_processor.filter_unprocessed(
                    input_paths,
                    compute_hashes=True,
                )
                
                # Save state after filtering
                incremental_processor.save()
                
                logger.info(
                    f"Incremental processing: {len(self.config.input_paths)} total paths, "
                    f"{len(input_paths)} unprocessed"
                )
            except (IOError, OSError, StorageError, RuntimeError) as e:
                logger.warning(f"Incremental processing failed, processing all files: {e}")
                input_paths = self.config.input_paths
        
        logger.info(f"Loading data from {len(input_paths)} sources")
        
        # Load main data sources with graceful degradation
        from pipeline.utils.partial_failure import PartialFailureHandler

        failure_handler = PartialFailureHandler(continue_on_failure=True)

        # Load main data sources
        dataset = failure_handler.execute_with_fallback(
            func=lambda: self.loader.load(input_paths),
            fallback_value=ray.data.from_items([]),
            error_context="Main data loading",
        )

        # Load additional sources with graceful degradation
        dataset = self._load_simulation_data(dataset, failure_handler)
        dataset = self._load_synthetic_data(dataset, failure_handler)
        dataset = self._load_omniverse_data(dataset, failure_handler)
        dataset = self._load_synthetic_generators(dataset, failure_handler)

        # Log available modalities without materializing dataset
        # Use iter_batches with limit=1 to sample without full materialization
        try:
            sample_batch = next(dataset.iter_batches(batch_size=1, prefetch_batches=0), None)
            if sample_batch is not None:
                sample = sample_batch.iloc[0].to_dict() if hasattr(sample_batch, 'iloc') else sample_batch[0] if isinstance(sample_batch, list) else sample_batch
                from pipeline.utils.data_types import detect_modalities
                modalities = detect_modalities(sample)
                logger.info(f"Loaded data with modalities: {[m.value for m in modalities]}")
        except (StopIteration, IndexError, AttributeError, TypeError) as e:
            logger.debug(f"Could not sample dataset for modality detection: {e}")

        return dataset

    def _load_simulation_data(self, dataset: Dataset, failure_handler: PartialFailureHandler) -> Dataset:
        """Load Isaac Lab simulation data.

        Args:
            dataset: Current dataset
            failure_handler: Failure handler instance

        Returns:
            Dataset with simulation data added
        """
        if not self.simulation_loaders:
            return dataset

        simulation_datasets = []
        for loader in self.simulation_loaders:
            sim_ds = failure_handler.execute_with_fallback(
                func=lambda l=loader: l.load(),
                fallback_value=None,
                error_context=f"Isaac Lab load from {loader.simulation_path}",
            )
            if sim_ds is not None:
                simulation_datasets.append(sim_ds)
            else:
                self.metrics.record_error(f"Isaac Lab load failed from {loader.simulation_path}")

        if simulation_datasets:
            return self._add_datasets(dataset, simulation_datasets, "Isaac Lab")
        return dataset

    def _load_synthetic_data(self, dataset: Dataset, failure_handler: PartialFailureHandler) -> Dataset:
        """Load Cosmos Dreams synthetic data.

        Args:
            dataset: Current dataset
            failure_handler: Failure handler instance

        Returns:
            Dataset with synthetic data added
        """
        if not self.synthetic_loaders:
            return dataset

        synthetic_datasets = []
        for loader in self.synthetic_loaders:
            synth_ds = failure_handler.execute_with_fallback(
                func=lambda l=loader: l.load(),
                fallback_value=None,
                error_context=f"Cosmos Dreams load from {loader.dreams_path}",
            )
            if synth_ds is not None:
                synthetic_datasets.append(synth_ds)
            else:
                self.metrics.record_error(f"Cosmos Dreams load failed from {loader.dreams_path}")

        if synthetic_datasets:
            return self._add_datasets(dataset, synthetic_datasets, "Cosmos Dreams")
        return dataset

    def _load_omniverse_data(self, dataset: Dataset, failure_handler: PartialFailureHandler) -> Dataset:
        """Load Omniverse USD and Replicator data.

        Args:
            dataset: Current dataset
            failure_handler: Failure handler instance

        Returns:
            Dataset with Omniverse data added
        """
        # Load from configured paths
        if hasattr(self.config, "omniverse_paths") and self.config.omniverse_paths:
            try:
                from pipeline.integrations.omniverse import OmniverseLoader
                import ray.data
                from pathlib import Path

                omniverse_items = []
                for omniverse_path in self.config.omniverse_paths:
                    loader = OmniverseLoader(
                        omniverse_path=omniverse_path,
                        include_metadata=True,
                        include_annotations=True,
                    )
                    
                    # Load USD files
                    usd_files = list(Path(omniverse_path).glob("*.usd")) + list(Path(omniverse_path).glob("*.usda")) + list(Path(omniverse_path).glob("*.usdc"))
                    for usd_file in usd_files:
                        items = loader.load_usd_scene(str(usd_file))
                        omniverse_items.extend(items)
                    
                    # Load Replicator data if available
                    replicator_path = Path(omniverse_path) / "replicator_output"
                    if replicator_path.exists():
                        replicator_items = loader.load_replicator_data(str(replicator_path))
                        omniverse_items.extend(replicator_items)

                if omniverse_items:
                    omniverse_dataset = ray.data.from_items(omniverse_items)
                    dataset = dataset.union(omniverse_dataset)
                    logger.info(f"Loaded {len(omniverse_items)} items from Omniverse")
            except (DataLoadError, IOError, OSError, RuntimeError) as e:
                dataset = failure_handler.execute_with_fallback(
                    func=lambda: dataset,
                    fallback_value=dataset,
                    error_context="Omniverse loading from config paths",
                )

        # Load from added loaders
        if self.omniverse_loaders:
            omniverse_datasets = []
            for loader in self.omniverse_loaders:
                try:
                    from pathlib import Path
                    import ray.data

                    omniverse_items = []
                    # Load USD files
                    usd_files = list(Path(loader.omniverse_path).glob("*.usd")) + list(Path(loader.omniverse_path).glob("*.usda")) + list(Path(loader.omniverse_path).glob("*.usdc"))
                    for usd_file in usd_files:
                        items = loader.load_usd_scene(str(usd_file))
                        omniverse_items.extend(items)
                    
                    # Load Replicator data if available
                    replicator_path = Path(loader.omniverse_path) / "replicator_output"
                    if replicator_path.exists():
                        replicator_items = loader.load_replicator_data(str(replicator_path))
                        omniverse_items.extend(replicator_items)

                    if omniverse_items:
                        omniverse_dataset = ray.data.from_items(omniverse_items)
                        omniverse_datasets.append(omniverse_dataset)
                except (DataLoadError, IOError, OSError, RuntimeError) as e:
                    failure_handler.execute_with_fallback(
                        func=lambda: None,
                        fallback_value=None,
                        error_context=f"Omniverse load from {loader.omniverse_path}",
                    )

            if omniverse_datasets:
                return self._add_datasets(dataset, omniverse_datasets, "Omniverse")

        return dataset

    def _load_synthetic_generators(self, dataset: Dataset, failure_handler: PartialFailureHandler) -> Dataset:
        """Load synthetic generator data.

        Args:
            dataset: Current dataset
            failure_handler: Failure handler instance

        Returns:
            Dataset with generator data added
        """
        if not self.synthetic_generators:
            return dataset

        generator_datasets = []
        for generator in self.synthetic_generators:
            gen_ds = failure_handler.execute_with_fallback(
                func=lambda g=generator: g.generate(),
                fallback_value=None,
                error_context=f"Synthetic generation from {generator.__class__.__name__}",
            )
            if gen_ds is not None:
                generator_datasets.append(gen_ds)
            else:
                self.metrics.record_error(f"Synthetic generation failed from {generator.__class__.__name__}")

        if generator_datasets:
            return self._add_datasets(dataset, generator_datasets, "synthetic generators")
        return dataset

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
        try:
            results = self.metrics.finish()
            results["output_path"] = self.config.output_path
            results["total_stages"] = len(self.stages)
            return results
        except (AttributeError, KeyError, TypeError, MetricsError) as e:
            logger.error(f"Error collecting metrics: {e}", exc_info=True)
            return {
                "output_path": self.config.output_path,
                "total_stages": len(self.stages),
                "error": "Failed to collect metrics",
            }

    def shutdown(self) -> None:
        """Shutdown pipeline and cleanup resources."""
        self.lifecycle.shutdown()

