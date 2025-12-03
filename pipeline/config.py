"""Configuration management for the multimodal data pipeline."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pipeline.config.defaults import (
    get_default_gpu_analytics_config,
    get_default_output_config,
    get_default_pipeline_config,
    get_default_ray_data_config,
    get_default_resource_config,
    get_default_sensor_config,
    get_default_text_config,
    get_default_video_config,
)
from pipeline.exceptions import ConfigurationError
from pipeline.utils.constants import (
    _DEFAULT_CHECKPOINT_INTERVAL,
    _DEFAULT_COMPUTE_MODE,
    _DEFAULT_DEDUP_METHOD,
    _DEFAULT_GPUS,
    _DEFAULT_PREFETCH_BUFFER_SIZE,
    _DEFAULT_RAY_DASHBOARD_PORT,
    _DEFAULT_SIMILARITY_THRESHOLD,
)


@dataclass
class PipelineConfig:
    """Configuration for multimodal data curation pipeline.

    Attributes:
        input_paths: List of input data paths (S3, local, etc.)
        output_path: Output path for curated data
        enable_gpu_dedup: Enable GPU-accelerated deduplication
        num_gpus: Number of GPUs to use
        num_cpus: Number of CPUs to use (default: auto)
        batch_size: Batch size for processing
        streaming: Enable streaming mode
        checkpoint_interval: Checkpoint every N batches
        dedup_method: Deduplication method ('fuzzy', 'semantic', 'both')
        similarity_threshold: Threshold for semantic similarity
        enable_observability: Enable metrics collection
    """

    input_paths: List[str]
    output_path: str
    enable_gpu_dedup: bool = True
    num_gpus: int = _DEFAULT_GPUS
    num_cpus: Optional[int] = None
    batch_size: Optional[int] = None
    streaming: bool = True
    checkpoint_interval: int = field(default_factory=lambda: _DEFAULT_CHECKPOINT_INTERVAL)
    checkpoint_dir: Optional[str] = None  # Directory for checkpoints (None = disabled)
    dedup_method: str = field(default_factory=lambda: _DEFAULT_DEDUP_METHOD)
    prefetch_buffer_size: int = field(default_factory=lambda: _DEFAULT_PREFETCH_BUFFER_SIZE)
    similarity_threshold: float = _DEFAULT_SIMILARITY_THRESHOLD
    enable_observability: bool = True
    enable_prometheus: bool = True  # Enable Prometheus metrics export
    prometheus_port: int = _DEFAULT_RAY_DASHBOARD_PORT  # Port for Prometheus metrics server
    enable_grafana: bool = True  # Generate Grafana dashboard configs
    compute_mode: str = _DEFAULT_COMPUTE_MODE  # "auto", "cpu", "gpu"
    random_seed: Optional[int] = None  # Random seed for reproducibility
    enable_lineage_tracking: bool = True  # Enable data lineage tracking

    # Video processing config - uses defaults factory for repeatability
    video_config: Dict[str, Any] = field(default_factory=get_default_video_config)

    # Text processing config - uses defaults factory for repeatability
    text_config: Dict[str, Any] = field(default_factory=get_default_text_config)

    # Sensor data config - uses defaults factory for repeatability
    sensor_config: Dict[str, Any] = field(default_factory=get_default_sensor_config)

    # Resource allocation - uses defaults factory for repeatability
    resource_config: Dict[str, Any] = field(default_factory=get_default_resource_config)

    # Ray Data performance configuration - uses defaults factory for repeatability
    ray_data_config: Dict[str, Any] = field(default_factory=get_default_ray_data_config)

    # GPU analytics / RAPIDS integration - uses defaults factory for repeatability
    gpu_analytics_config: Dict[str, Any] = field(default_factory=get_default_gpu_analytics_config)

    # GPU memory pool configuration
    gpu_memory_pool_size: Optional[int] = None  # RMM pool size in bytes (None = auto-detect)
    enable_rmm_pool: bool = True  # Enable RMM memory pool for GPU operations

    # Output configuration - uses defaults factory for repeatability
    output_config: Dict[str, Any] = field(default_factory=get_default_output_config)

    # Omniverse integration configuration
    omniverse_paths: Optional[List[str]] = None  # Paths to Omniverse USD files or Replicator output
    cosmos_paths: Optional[List[str]] = None  # Paths to Cosmos Dreams data (kept for backward compatibility)
    enable_omniverse_export: bool = False  # Export curated data to USD format
    omniverse_export_path: Optional[str] = None  # Output path for USD export

    def validate(self) -> None:
        """Validate configuration parameters.
        
        Raises:
            ConfigurationError: If configuration is invalid
        """
        errors: List[str] = []
        
        # Validate input_paths
        if not isinstance(self.input_paths, list):
            errors.append("input_paths must be a list")
        elif not self.input_paths:
            errors.append("At least one input path must be specified")
        elif not all(isinstance(path, str) and path for path in self.input_paths):
            errors.append("Each input path must be a non-empty string")
        else:
            # Check for duplicates
            normalized_inputs = [path.rstrip("/") for path in self.input_paths]
            if len(set(normalized_inputs)) != len(normalized_inputs):
                errors.append("input_paths contains duplicate entries")

        # Validate output_path
        if not isinstance(self.output_path, str) or not self.output_path:
            errors.append("output_path must be a non-empty string")
        elif isinstance(self.input_paths, list) and self.input_paths:
            # Check output doesn't match input
            normalized_inputs = [path.rstrip("/") for path in self.input_paths]
            if self.output_path.rstrip("/") in normalized_inputs:
                errors.append("output_path must differ from all input paths")

        # Validate GPU/CPU counts
        if self.num_gpus < 0:
            errors.append("num_gpus must be non-negative")
        if self.num_cpus is not None and self.num_cpus < 0:
            errors.append("num_cpus must be non-negative")

        from pipeline.utils.constants import _GPU_BATCH_SIZE, _TRAINING_BATCH_SIZE

        # Validate batch_size
        if self.batch_size is None:
            self.batch_size = _TRAINING_BATCH_SIZE
        elif self.batch_size <= 0:
            errors.append("batch_size must be positive")
        elif self.batch_size > _GPU_BATCH_SIZE * 10:
            errors.append(
                f"batch_size {self.batch_size} is too large, "
                "may cause memory issues. Use smaller batch size."
            )

        # Validate dedup_method
        if self.dedup_method not in ["fuzzy", "semantic", "both"]:
            errors.append("dedup_method must be 'fuzzy', 'semantic', or 'both'")

        # Validate similarity_threshold
        if not 0.0 <= self.similarity_threshold <= 1.0:
            errors.append("similarity_threshold must be between 0.0 and 1.0")
        
        # Validate checkpoint_interval
        if self.checkpoint_interval < 0:
            errors.append("checkpoint_interval must be non-negative")
        
        # Validate prefetch_buffer_size
        if self.prefetch_buffer_size < 0:
            errors.append("prefetch_buffer_size must be non-negative")
        
        # Validate prometheus_port
        if not 1024 <= self.prometheus_port <= 65535:
            errors.append("prometheus_port must be between 1024 and 65535")
        
        # Normalize and validate compute_mode
        self.compute_mode = (self.compute_mode or "auto").lower()
        if self.compute_mode not in {"auto", "cpu", "gpu"}:
            errors.append("compute_mode must be 'auto', 'cpu', or 'gpu'")
        else:
            # Apply compute_mode constraints
            if self.compute_mode == "cpu":
                self.num_gpus = 0
                self.enable_gpu_dedup = False
            elif self.compute_mode == "gpu":
                if self.num_gpus <= 0:
                    errors.append("compute_mode='gpu' requires num_gpus > 0")
                else:
                    self.enable_gpu_dedup = True
            else:  # auto
                if self.num_gpus <= 0:
                    self.compute_mode = "cpu"
                    self.enable_gpu_dedup = False

        # Mirror compute mode into resource config for downstream consumers
        if "compute_mode" not in self.resource_config:
            self.resource_config["compute_mode"] = self.compute_mode

        # GPU analytics config validation
        gpu_cfg = self.gpu_analytics_config or {}
        if not isinstance(gpu_cfg, dict):
            errors.append("gpu_analytics_config must be a dictionary")
        elif gpu_cfg.get("enabled"):
            targets = gpu_cfg.get("target_columns", [])
            if not isinstance(targets, list) or not targets:
                errors.append("gpu_analytics_config.target_columns must be a non-empty list when enabled")
            metrics = gpu_cfg.get("metrics", [])
            if not isinstance(metrics, list) or not metrics:
                errors.append("gpu_analytics_config.metrics must be a non-empty list when enabled")
            if gpu_cfg.get("num_gpus", 1) <= 0:
                errors.append("gpu_analytics_config.num_gpus must be positive")

        # Validate ray_data_config
        if self.ray_data_config:
            rd_config = self.ray_data_config
            if "override_num_blocks" in rd_config and rd_config["override_num_blocks"] is not None:
                if rd_config["override_num_blocks"] <= 0:
                    errors.append("ray_data_config.override_num_blocks must be positive")

            if "read_num_cpus" in rd_config and rd_config["read_num_cpus"] is not None:
                if rd_config["read_num_cpus"] <= 0:
                    errors.append("ray_data_config.read_num_cpus must be positive")

            if "min_block_size" in rd_config and rd_config["min_block_size"] is not None:
                if rd_config["min_block_size"] <= 0:
                    errors.append("ray_data_config.min_block_size must be positive")

            if "max_block_size" in rd_config and rd_config["max_block_size"] is not None:
                if rd_config["max_block_size"] <= 0:
                    errors.append("ray_data_config.max_block_size must be positive")
                elif (
                    "min_block_size" in rd_config
                    and rd_config["min_block_size"] is not None
                    and rd_config["max_block_size"] < rd_config["min_block_size"]
                ):
                    errors.append("ray_data_config.max_block_size must be >= min_block_size")

        # Validate resource config
        self._validate_resource_config(errors)
        
        # Validate processing configs
        self._validate_processing_configs(errors)
        
        # Raise ConfigurationError if any errors found
        if errors:
            raise ConfigurationError(
                "Configuration validation failed",
                error_code="CONFIG_VALIDATION_FAILED",
                details={"errors": errors}
            )

    def _validate_resource_config(self, errors: List[str]) -> None:
        """Validate resource configuration block.
        
        Args:
            errors: List to append validation errors to
        """
        if not isinstance(self.resource_config, dict):
            errors.append("resource_config must be a dictionary")
            return

        min_workers = self.resource_config.get("min_workers", 0)
        max_workers = self.resource_config.get("max_workers")

        if not isinstance(min_workers, int):
            errors.append("resource_config.min_workers must be an integer")
        elif min_workers < 0:
            errors.append("resource_config.min_workers must be non-negative")
        
        if max_workers is not None:
            if not isinstance(max_workers, int):
                errors.append("resource_config.max_workers must be an integer")
            elif max_workers < 0:
                errors.append("resource_config.max_workers must be non-negative")
            elif max_workers < min_workers:
                errors.append("resource_config.max_workers must be >= min_workers")

        enable_autoscaling = self.resource_config.get("enable_autoscaling", True)
        if enable_autoscaling and max_workers is None:
            errors.append("resource_config.max_workers must be set when autoscaling is enabled")

    def _validate_processing_configs(self, errors: List[str]) -> None:
        """Validate per-modality processing configuration dictionaries.
        
        Args:
            errors: List to append validation errors to
        """
        for name, config in (
            ("video_config", self.video_config),
            ("text_config", self.text_config),
            ("sensor_config", self.sensor_config),
        ):
            if not isinstance(config, dict):
                errors.append(f"{name} must be a dictionary")
            elif not config:
                errors.append(f"{name} cannot be empty")
