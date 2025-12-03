"""Per-stage configuration for batch sizing and resource allocation.

Provides automatic tuning and per-stage resource management.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class StageResourceConfig:
    """Resource configuration for a single pipeline stage.
    
    Supports both explicit configuration and automatic tuning.
    """
    
    # Batch sizing
    batch_size: Optional[int] = None  # None = auto-tune
    min_batch_size: int = 1
    max_batch_size: int = 100000
    
    # Resource allocation
    num_gpus: Optional[int] = None  # None = inherit from pipeline or auto-tune
    num_cpus: Optional[int] = None  # None = inherit from pipeline or auto-tune
    memory_mb: Optional[int] = None  # Memory limit in MB
    
    # Automatic tuning
    auto_tune: bool = True  # Enable automatic tuning
    target_throughput: Optional[float] = None  # Target samples/second (None = maximize)
    target_latency_ms: Optional[float] = None  # Target latency in ms (None = no constraint)
    
    # Stage-specific hints for tuning
    stage_type: Optional[str] = None  # "cpu_bound", "gpu_bound", "io_bound", "memory_bound"
    estimated_memory_per_sample_mb: Optional[float] = None  # Memory per sample estimate
    
    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.batch_size is not None and self.batch_size < self.min_batch_size:
            raise ValueError(f"batch_size {self.batch_size} < min_batch_size {self.min_batch_size}")
        if self.batch_size is not None and self.batch_size > self.max_batch_size:
            raise ValueError(f"batch_size {self.batch_size} > max_batch_size {self.max_batch_size}")
        if self.num_gpus is not None and self.num_gpus < 0:
            raise ValueError("num_gpus must be non-negative")
        if self.num_cpus is not None and self.num_cpus < 0:
            raise ValueError("num_cpus must be non-negative")
        if self.memory_mb is not None and self.memory_mb < 0:
            raise ValueError("memory_mb must be non-negative")


@dataclass
class StageConfig:
    """Complete configuration for a pipeline stage.
    
    Combines stage-specific configuration with resource allocation.
    """
    
    stage_name: str
    stage_class: str  # Stage class name or type identifier
    stage_kwargs: Dict[str, Any] = field(default_factory=dict)
    resources: StageResourceConfig = field(default_factory=StageResourceConfig)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "stage_name": self.stage_name,
            "stage_class": self.stage_class,
            "stage_kwargs": self.stage_kwargs,
            "resources": {
                "batch_size": self.resources.batch_size,
                "num_gpus": self.resources.num_gpus,
                "num_cpus": self.resources.num_cpus,
                "memory_mb": self.resources.memory_mb,
                "auto_tune": self.resources.auto_tune,
                "stage_type": self.resources.stage_type,
            },
        }


class StageResourceTuner:
    """Automatic tuner for stage batch sizes and resources.
    
    Analyzes stage characteristics and available resources to optimize
    batch sizes and resource allocation per stage.
    """
    
    def __init__(
        self,
        total_gpus: int = 0,
        total_cpus: Optional[int] = None,
        total_memory_gb: Optional[float] = None,
    ):
        """Initialize resource tuner.
        
        Args:
            total_gpus: Total GPUs available
            total_cpus: Total CPUs available (None = auto-detect)
            total_memory_gb: Total memory in GB (None = auto-detect)
        """
        self.total_gpus = total_gpus
        self.total_cpus = total_cpus
        self.total_memory_gb = total_memory_gb
        
        # Stage type defaults
        self._stage_type_defaults = {
            "video": {"batch_size": 10, "num_gpus": 1, "stage_type": "gpu_bound"},
            "inference": {"batch_size": 32, "num_gpus": 1, "stage_type": "gpu_bound"},
            "profiler": {"batch_size": 1000, "num_gpus": 0, "stage_type": "cpu_bound"},
            "validator": {"batch_size": 1000, "num_gpus": 0, "stage_type": "cpu_bound"},
            "transformer": {"batch_size": 100, "num_gpus": 0, "stage_type": "cpu_bound"},
            "aggregator": {"batch_size": 1000, "num_gpus": 0, "stage_type": "memory_bound"},
        }
    
    def tune_stage(
        self,
        stage_config: StageConfig,
        pipeline_batch_size: Optional[int] = None,
        pipeline_num_gpus: int = 0,
        pipeline_num_cpus: Optional[int] = None,
    ) -> StageResourceConfig:
        """Tune resource configuration for a stage.
        
        Args:
            stage_config: Stage configuration to tune
            pipeline_batch_size: Default batch size from pipeline config
            pipeline_num_gpus: Default GPUs from pipeline config
            pipeline_num_cpus: Default CPUs from pipeline config
            
        Returns:
            Tuned resource configuration
        """
        resources = stage_config.resources
        
        # If auto_tune is disabled and explicit values are set, use them
        if not resources.auto_tune:
            if resources.batch_size is None:
                resources.batch_size = pipeline_batch_size or 100
            if resources.num_gpus is None:
                resources.num_gpus = pipeline_num_gpus
            if resources.num_cpus is None:
                resources.num_cpus = pipeline_num_cpus
            return resources
        
        # Automatic tuning logic
        stage_type = resources.stage_type or self._infer_stage_type(stage_config)
        
        # Get defaults for stage type
        defaults = self._stage_type_defaults.get(stage_type, {})
        
        # Tune batch size
        if resources.batch_size is None:
            resources.batch_size = self._tune_batch_size(
                stage_config,
                defaults.get("batch_size", pipeline_batch_size or 100),
                resources,
            )
        
        # Tune GPU allocation
        if resources.num_gpus is None:
            resources.num_gpus = self._tune_gpu_allocation(
                stage_config,
                defaults.get("num_gpus", 0),
                pipeline_num_gpus,
            )
        
        # Tune CPU allocation
        if resources.num_cpus is None:
            resources.num_cpus = self._tune_cpu_allocation(
                stage_config,
                defaults.get("num_cpus", 1),
                pipeline_num_cpus,
            )
        
        # Tune memory allocation
        if resources.memory_mb is None and resources.estimated_memory_per_sample_mb:
            estimated_memory_mb = (
                resources.batch_size * resources.estimated_memory_per_sample_mb * 1.5  # 1.5x safety margin
            )
            if self.total_memory_gb:
                max_memory_mb = int(self.total_memory_gb * 1024 * 0.8)  # Use 80% of total
                resources.memory_mb = min(estimated_memory_mb, max_memory_mb)
            else:
                resources.memory_mb = int(estimated_memory_mb)
        
        return resources
    
    def _infer_stage_type(self, stage_config: StageConfig) -> str:
        """Infer stage type from configuration.
        
        Args:
            stage_config: Stage configuration
            
        Returns:
            Inferred stage type
        """
        stage_class = stage_config.stage_class.lower()
        stage_name = stage_config.stage_name.lower()
        
        # Check stage class name
        if "video" in stage_class or "video" in stage_name:
            return "video"
        if "inference" in stage_class or "inference" in stage_name:
            return "inference"
        if "profiler" in stage_class or "profiler" in stage_name:
            return "profiler"
        if "validator" in stage_class or "validator" in stage_name:
            return "validator"
        if "transformer" in stage_class or "transformer" in stage_name:
            return "transformer"
        if "aggregator" in stage_class or "aggregator" in stage_name:
            return "aggregator"
        
        # Check stage kwargs for hints
        kwargs = stage_config.stage_kwargs
        if kwargs.get("use_gpu") or kwargs.get("num_gpus", 0) > 0:
            return "gpu_bound"
        if kwargs.get("use_cpu") or "cpu" in stage_name:
            return "cpu_bound"
        
        # Default
        return "cpu_bound"
    
    def _tune_batch_size(
        self,
        stage_config: StageConfig,
        default_batch_size: int,
        resources: StageResourceConfig,
    ) -> int:
        """Tune batch size for a stage.
        
        Args:
            stage_config: Stage configuration
            default_batch_size: Default batch size
            resources: Resource configuration
            
        Returns:
            Tuned batch size
        """
        # Start with default or stage-specific default
        batch_size = default_batch_size
        
        # Apply constraints
        batch_size = max(resources.min_batch_size, min(batch_size, resources.max_batch_size))
        
        # Adjust based on memory estimate if available
        if resources.estimated_memory_per_sample_mb and self.total_memory_gb:
            max_batch_by_memory = int(
                (self.total_memory_gb * 1024 * 0.5) / resources.estimated_memory_per_sample_mb
            )  # Use 50% of memory
            batch_size = min(batch_size, max_batch_by_memory)
        
        return batch_size
    
    def _tune_gpu_allocation(
        self,
        stage_config: StageConfig,
        default_gpus: int,
        pipeline_gpus: int,
    ) -> int:
        """Tune GPU allocation for a stage.
        
        Args:
            stage_config: Stage configuration
            default_gpus: Default GPUs for stage type
            pipeline_gpus: Total GPUs from pipeline config
            
        Returns:
            Tuned GPU count
        """
        # Check if stage needs GPU
        stage_type = self._infer_stage_type(stage_config)
        needs_gpu = stage_type in ("video", "inference", "gpu_bound")
        
        if not needs_gpu:
            return 0
        
        # Use stage default or pipeline default
        gpus = default_gpus or pipeline_gpus
        
        # Don't exceed total available
        if self.total_gpus > 0:
            gpus = min(gpus, self.total_gpus)
        
        return max(0, gpus)
    
    def _tune_cpu_allocation(
        self,
        stage_config: StageConfig,
        default_cpus: int,
        pipeline_cpus: Optional[int],
    ) -> int:
        """Tune CPU allocation for a stage.
        
        Args:
            stage_config: Stage configuration
            default_cpus: Default CPUs for stage type
            pipeline_cpus: Total CPUs from pipeline config
            
        Returns:
            Tuned CPU count
        """
        # Use stage default or pipeline default
        cpus = default_cpus or pipeline_cpus or 1
        
        # Don't exceed total available
        if self.total_cpus:
            cpus = min(cpus, self.total_cpus)
        
        return max(1, cpus)

