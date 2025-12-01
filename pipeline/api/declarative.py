"""High-level declarative API for multimodal data pipeline.

Supports both Python and YAML-based configuration for easy pipeline setup.
Provides single-pipeline and multi-pipeline orchestration similar to modern
multimodal ETL tools (Airflow, Dagster, Prefect, NVIDIA NeMo Data Curator),
while tightly integrating with Ray Data and NVIDIA robotics tooling such as
Isaac Lab and Cosmos (GR00T Dreams).
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

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

    @classmethod
    def cpu_demo(
        cls,
        sources: List[Dict[str, Any]],
        output: str,
        **kwargs: Any,
    ) -> "Pipeline":
        """Convenience constructor for CPU-only local demos."""
        return cls(
            sources=sources,
            output=output,
            enable_gpu=False,
            num_gpus=0,
            compute_mode="cpu",
            **kwargs,
        )

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

    def run(self) -> Dict[str, Any]:
        """Run the pipeline.

        Returns:
            Dictionary with pipeline execution results

        Raises:
            RuntimeError: If pipeline execution fails
        """
        if not self.sources:
            raise ValueError("No sources configured for pipeline")

        logger.info(f"Running declarative pipeline with {len(self.sources)} sources")
        try:
            return self.pipeline.run()
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
        except Exception as e:
            logger.error(f"Failed to export YAML config to {output_path}: {e}", exc_info=True)
            raise


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

