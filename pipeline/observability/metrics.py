"""Performance metrics collection for the pipeline."""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class StageMetrics:
    """Metrics for a single pipeline stage."""

    stage_name: str
    start_time: float
    end_time: float = 0.0
    items_processed: int = 0
    items_filtered: int = 0
    errors: List[str] = field(default_factory=list)
    per_modality_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    gpu_utilization: float = 0.0
    memory_usage: float = 0.0
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    resource_metrics: Optional[Dict[str, Any]] = None  # Detailed resource metrics from psutil

    @property
    def duration(self) -> float:
        """Get stage duration in seconds."""
        if self.end_time == 0:
            return time.time() - self.start_time
        return self.end_time - self.start_time

    @property
    def throughput(self) -> float:
        """Get items per second."""
        if self.duration == 0:
            return 0.0
        return self.items_processed / self.duration


class PipelineMetrics:
    """Metrics collection for the entire pipeline.

    Enhanced with per-modality and per-stage metrics for GR00T.
    Integrates with Prometheus for Grafana monitoring.
    Supports MLflow and W&B for experiment tracking.
    """

    def __init__(
        self,
        enabled: bool = True,
        enable_prometheus: bool = True,
        enable_mlflow: bool = False,
        enable_wandb: bool = False,
        mlflow_tracking_uri: Optional[str] = None,
        wandb_project: Optional[str] = None,
    ):
        """Initialize metrics collector.

        Args:
            enabled: Whether to collect metrics
            enable_prometheus: Whether to export to Prometheus
            enable_mlflow: Whether to log metrics to MLflow
            enable_wandb: Whether to log metrics to W&B
            mlflow_tracking_uri: MLflow tracking URI
            wandb_project: W&B project name
        """
        self.enabled = enabled
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self.stages: List[StageMetrics] = []
        self.errors: List[str] = []
        self.total_samples: int = 0
        self.dedup_rate: float = 0.0
        self.avg_gpu_util: float = 0.0
        self.per_modality_counts: Dict[str, int] = {}
        self.per_stage_metrics: Dict[str, Dict[str, Any]] = {}
        
        # Initialize Prometheus exporter if enabled
        self.prometheus_exporter = None
        if enable_prometheus and enabled:
            try:
                from pipeline.observability.prometheus import create_prometheus_exporter
                self.prometheus_exporter = create_prometheus_exporter(
                    enabled=True,
                    start_server=True,
                )
                logger.info("Prometheus metrics exporter initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Prometheus exporter: {e}")

        # Initialize MLflow tracker if enabled
        self.mlflow_tracker = None
        if enable_mlflow and enabled:
            try:
                from pipeline.integrations.mlflow import create_mlflow_tracker
                self.mlflow_tracker = create_mlflow_tracker(
                    tracking_uri=mlflow_tracking_uri,
                    enabled=True,
                )
                logger.info("MLflow tracker initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize MLflow tracker: {e}")

        # Initialize W&B tracker if enabled
        self.wandb_tracker = None
        if enable_wandb and enabled:
            try:
                from pipeline.integrations.wandb import create_wandb_tracker
                self.wandb_tracker = create_wandb_tracker(
                    project=wandb_project or "multimodal-pipeline",
                    enabled=True,
                )
                logger.info("W&B tracker initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize W&B tracker: {e}")

    def start(self) -> None:
        """Start metrics collection."""
        if self.enabled:
            self.start_time = time.time()
            logger.info("Metrics collection started")

    def record_stage(self, stage_name: str) -> None:
        """Record a stage execution.

        Args:
            stage_name: Name of the stage
        """
        if not self.enabled:
            return

        # End previous stage if exists
        if self.stages:
            prev_stage = self.stages[-1]
            prev_stage.end_time = time.time()
            # Store per-stage metrics
            self.per_stage_metrics[prev_stage.stage_name] = {
                "duration": prev_stage.duration,
                "throughput": prev_stage.throughput,
                "items_processed": prev_stage.items_processed,
                "items_filtered": prev_stage.items_filtered,
                "gpu_utilization": prev_stage.gpu_utilization,
                "memory_usage": prev_stage.memory_usage,
                "per_modality": prev_stage.per_modality_metrics,
            }
            
            # Export to Prometheus
            if self.prometheus_exporter:
                self.prometheus_exporter.record_stage_metrics(
                    stage_name=prev_stage.stage_name,
                    duration=prev_stage.duration,
                    items_processed=prev_stage.items_processed,
                    items_filtered=prev_stage.items_filtered,
                    errors=len(prev_stage.errors),
                    cpu_percent=getattr(prev_stage, 'cpu_percent', None),
                    memory_percent=getattr(prev_stage, 'memory_percent', None),
                    memory_used_mb=getattr(prev_stage, 'memory_usage', None),
                )

            # Export to MLflow
            if self.mlflow_tracker:
                try:
                    self.mlflow_tracker.log_metrics({
                        f"stage/{prev_stage.stage_name}/duration": prev_stage.duration,
                        f"stage/{prev_stage.stage_name}/throughput": prev_stage.throughput,
                        f"stage/{prev_stage.stage_name}/items_processed": prev_stage.items_processed,
                        f"stage/{prev_stage.stage_name}/items_filtered": prev_stage.items_filtered,
                        f"stage/{prev_stage.stage_name}/errors": len(prev_stage.errors),
                    })
                except Exception as e:
                    logger.debug(f"Failed to log metrics to MLflow: {e}")

            # Export to W&B
            if self.wandb_tracker:
                try:
                    self.wandb_tracker.log_metrics({
                        f"stage/{prev_stage.stage_name}/duration": prev_stage.duration,
                        f"stage/{prev_stage.stage_name}/throughput": prev_stage.throughput,
                        f"stage/{prev_stage.stage_name}/items_processed": prev_stage.items_processed,
                        f"stage/{prev_stage.stage_name}/items_filtered": prev_stage.items_filtered,
                        f"stage/{prev_stage.stage_name}/errors": len(prev_stage.errors),
                    })
                except Exception as e:
                    logger.debug(f"Failed to log metrics to W&B: {e}")

        # Start new stage
        stage = StageMetrics(stage_name=stage_name, start_time=time.time())
        self.stages.append(stage)
        logger.info(f"Stage started: {stage_name}")

    def record_modality_metrics(
        self, modality: str, items_processed: int = 0, items_filtered: int = 0
    ) -> None:
        """Record metrics for a specific modality.

        Args:
            modality: Modality name (video, sensor, text)
            items_processed: Number of items processed
            items_filtered: Number of items filtered
        """
        if not self.enabled:
            return

        if modality not in self.per_modality_counts:
            self.per_modality_counts[modality] = 0
        self.per_modality_counts[modality] += items_processed

        # Record in current stage
        if self.stages:
            current_stage = self.stages[-1]
            if modality not in current_stage.per_modality_metrics:
                current_stage.per_modality_metrics[modality] = {
                    "items_processed": 0,
                    "items_filtered": 0,
                }
            current_stage.per_modality_metrics[modality]["items_processed"] += items_processed
            current_stage.per_modality_metrics[modality]["items_filtered"] += items_filtered

    def record_gpu_utilization(self, utilization: float, device_id: int = 0) -> None:
        """Record GPU utilization.

        Args:
            utilization: GPU utilization (0.0-1.0)
            device_id: GPU device ID
        """
        if not self.enabled:
            return

        if self.stages:
            self.stages[-1].gpu_utilization = utilization

        # Update average
        if len(self.stages) > 0:
            all_utils = [s.gpu_utilization for s in self.stages if s.gpu_utilization > 0]
            if all_utils:
                self.avg_gpu_util = sum(all_utils) / len(all_utils)

        # Export to Prometheus
        if self.prometheus_exporter:
            try:
                from pipeline.utils.gpu.memory import get_gpu_memory_info
                mem_info = get_gpu_memory_info(device_id=device_id)
                self.prometheus_exporter.record_gpu_metrics(
                    device_id=device_id,
                    utilization=utilization,
                    memory_used=mem_info.get("allocated", 0),
                    memory_total=mem_info.get("total", 0),
                )
            except Exception as e:
                logger.debug(f"Failed to record GPU metrics to Prometheus: {e}")

    def record_error(self, error: str) -> None:
        """Record an error.

        Args:
            error: Error message
        """
        if self.enabled:
            self.errors.append(error)
            if self.stages:
                self.stages[-1].errors.append(error)
            logger.error(f"Error recorded: {error}")

    def finish(self) -> Dict[str, Any]:
        """Finish metrics collection and return results.

        Returns:
            Dictionary of final metrics
        """
        if not self.enabled:
            return {}

        self.end_time = time.time()

        # Finalize last stage
        if self.stages:
            self.stages[-1].end_time = self.end_time

        # Calculate aggregate metrics
        total_duration = self.end_time - self.start_time
        total_processed = sum(stage.items_processed for stage in self.stages)

        # Export final pipeline metrics to Prometheus
        if self.prometheus_exporter:
            self.prometheus_exporter.record_pipeline_metrics(
                total_samples=self.total_samples,
                dedup_rate=self.dedup_rate,
                total_duration=total_duration,
            )

        results = {
            "total_duration": total_duration,
            "total_samples": self.total_samples,
            "total_processed": total_processed,
            "dedup_rate": self.dedup_rate,
            "avg_gpu_util": self.avg_gpu_util,
            "num_stages": len(self.stages),
            "num_errors": len(self.errors),
            "per_modality_counts": self.per_modality_counts,
            "per_stage_metrics": self.per_stage_metrics,
            "prometheus_enabled": self.prometheus_exporter is not None,
            "stages": [
                {
                    "name": stage.stage_name,
                    "duration": stage.duration,
                    "throughput": stage.throughput,
                    "items_processed": stage.items_processed,
                    "items_filtered": stage.items_filtered,
                    "errors": len(stage.errors),
                    "gpu_utilization": stage.gpu_utilization,
                    "memory_usage": stage.memory_usage,
                    "per_modality": stage.per_modality_metrics,
                }
                for stage in self.stages
            ],
        }

        logger.info(f"Metrics collection finished: {total_duration:.2f}s")
        return results

    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current metrics snapshot.

        Returns:
            Dictionary of current metrics
        """
        if not self.enabled:
            return {}

        current_time = time.time()
        elapsed = current_time - self.start_time if self.start_time > 0 else 0.0

        return {
            "elapsed_time": elapsed,
            "current_stage": self.stages[-1].stage_name if self.stages else None,
            "num_stages_completed": len(self.stages),
            "total_samples": self.total_samples,
            "errors": len(self.errors),
        }
