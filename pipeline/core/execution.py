"""Pipeline execution logic.

Handles stage execution, checkpointing, output writing, and error handling.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional


import ray
from ray.data import Dataset

from pipeline.config import PipelineConfig
from pipeline.observability.metrics import PipelineMetrics
from pipeline.observability.resource_monitor import StageResourceMonitor, ResourceMetricsStore
from pipeline.utils.profiling import create_profiler, create_monitor

logger = logging.getLogger(__name__)


class PipelineExecutor:
    """Executes pipeline stages and manages execution flow."""

    def __init__(self, config: PipelineConfig, metrics: PipelineMetrics) -> None:
        """Initialize pipeline executor.

        Args:
            config: Pipeline configuration
            metrics: Pipeline metrics tracker
        """
        self.config = config
        self.metrics = metrics
        
        # Initialize profiling if enabled
        enable_profiling = getattr(config, "enable_profiling", False)
        profile_dir = getattr(config, "profile_dir", "profiles")
        self.profiler = create_profiler(
            output_dir=profile_dir, enable_profiling=enable_profiling
        )
        self.monitor = create_monitor(enabled=enable_profiling)
        
        # Initialize resource metrics store
        enable_resource_monitoring = getattr(config, "enable_resource_monitoring", True)
        metrics_dir = getattr(config, "metrics_dir", None) or "metrics"
        metrics_path = str(Path(metrics_dir) / "resource_metrics.json")
        self.resource_store = ResourceMetricsStore(
            storage_path=metrics_path if enable_resource_monitoring else None,
            max_history_per_stage=100,
            use_compression=True,
        )
        self.enable_resource_monitoring = enable_resource_monitoring

    def execute_stages(
        self,
        dataset: Dataset,
        stages: list[Any],
    ) -> Dataset:
        """Execute all pipeline stages sequentially.

        Args:
            dataset: Input dataset
            stages: List of pipeline stages to execute

        Returns:
            Processed dataset
        """
        checkpoint_manager = self._initialize_checkpointing()

        batch_count = 0
        for i, stage in enumerate(stages):
            stage_name = getattr(stage, 'name', stage.__class__.__name__)
            logger.info(f"Applying stage {i + 1}/{len(stages)}: {stage_name}")

            try:
                # Monitor resource utilization during stage execution
                # Use longer sampling interval for scalability (1.0s default)
                # Automatically detects and monitors distributed Ray workers
                if self.enable_resource_monitoring:
                    sampling_interval = getattr(self.config, "resource_monitoring_sampling_interval", 1.0)
                    max_workers = getattr(self.config, "resource_monitoring_max_workers", 100)
                    use_distributed = getattr(self.config, "resource_monitoring_use_distributed", True)
                    
                    resource_monitor = StageResourceMonitor(
                        stage_name=stage_name,
                        sampling_interval=sampling_interval,
                        enable_gpu_monitoring=self.config.num_gpus > 0,
                        max_workers_to_monitor=max_workers,
                        use_distributed_monitoring=use_distributed,
                    )
                else:
                    resource_monitor = None
                
                # Profile and monitor stage execution
                if resource_monitor:
                    with self.profiler.profile_stage(stage_name):
                        with self.monitor.monitor_stage(stage_name):
                            with resource_monitor.monitor():
                                dataset = self._execute_stage(dataset, stage, i)
                    
                    # Record resource metrics
                    resource_metrics = resource_monitor.get_metrics()
                    self.resource_store.record_stage_metrics(resource_metrics)
                    
                    # Update pipeline metrics with resource data
                    if self.metrics.stages:
                        current_stage = self.metrics.stages[-1]
                        current_stage.gpu_utilization = resource_metrics.avg_gpu_utilization[0] if resource_metrics.avg_gpu_utilization else 0.0
                        current_stage.memory_usage = resource_metrics.avg_memory_used_mb
                        current_stage.cpu_percent = resource_metrics.avg_cpu_percent
                        current_stage.memory_percent = resource_metrics.avg_memory_percent
                        current_stage.resource_metrics = resource_metrics.to_dict()
                    
                    # Export resource metrics to Prometheus
                    if self.metrics.prometheus_exporter:
                        self._export_resource_metrics_to_prometheus(resource_metrics)
                    
                    # Log resource utilization summary
                    logger.info(
                        f"Stage {stage_name} resource utilization: "
                        f"CPU={resource_metrics.avg_cpu_percent:.1f}%, "
                        f"Memory={resource_metrics.avg_memory_percent:.1f}%, "
                        f"GPU={[f'{u:.1f}%' for u in resource_metrics.avg_gpu_utilization] if resource_metrics.avg_gpu_utilization else 'N/A'}"
                    )
                else:
                    # Resource monitoring disabled
                    with self.profiler.profile_stage(stage_name):
                        with self.monitor.monitor_stage(stage_name):
                            dataset = self._execute_stage(dataset, stage, i)
                
                self.metrics.record_stage(stage_name)

                self._perform_periodic_cleanup(i)

                if checkpoint_manager:
                    batch_count += 1
                    if batch_count >= self.config.checkpoint_interval:
                        self._save_checkpoint(checkpoint_manager, dataset, stage, i)
                        batch_count = 0

            except (ValueError, RuntimeError, AttributeError, TypeError, KeyError, IOError, OSError) as e:
                logger.error(f"Stage {stage.__class__.__name__} failed: {e}", exc_info=True)
                self.metrics.record_error(f"Stage {stage.__class__.__name__}: {str(e)}")

                if checkpoint_manager:
                    self._save_checkpoint_on_error(checkpoint_manager, dataset, stage, i, e)

                raise

        # Generate profiling report if profiling was enabled
        if self.profiler.enable_profiling:
            try:
                report_path = self.profiler.generate_report()
                if report_path:
                    logger.info(f"Performance profiling report saved to {report_path}")
                
                # Log performance summary
                summary = self.monitor.get_summary()
                if summary:
                    logger.info(
                        f"Performance summary: total_duration={summary.get('total_duration', 0):.2f}s, "
                        f"max_memory={summary.get('max_memory_mb', 0):.2f}MB"
                    )
            except Exception as e:
                logger.warning(f"Failed to generate profiling report: {e}")
        
        # Save resource metrics for future analysis
        if self.enable_resource_monitoring:
            try:
                self.resource_store.save()
            except Exception as e:
                logger.warning(f"Failed to save resource metrics: {e}")

        return dataset
    
    def _export_resource_metrics_to_prometheus(self, resource_metrics: Any) -> None:
        """Export resource metrics to Prometheus.
        
        Args:
            resource_metrics: StageResourceMetrics instance
        """
        if not self.metrics.prometheus_exporter:
            return
        
        try:
            from pipeline.observability.resource_monitor import StageResourceMetrics
            
            exporter = self.metrics.prometheus_exporter
            
            # Export CPU metrics
            # Note: Prometheus exporter doesn't have CPU metrics yet, but we can add them
            
            # Export GPU metrics per device
            for device_idx, gpu_util in enumerate(resource_metrics.avg_gpu_utilization):
                gpu_mem_used = resource_metrics.avg_gpu_memory_used_mb[device_idx] * 1024 * 1024 if device_idx < len(resource_metrics.avg_gpu_memory_used_mb) else 0
                gpu_mem_total = resource_metrics.gpu_memory_total_mb[device_idx] * 1024 * 1024 if device_idx < len(resource_metrics.gpu_memory_total_mb) else 0
                gpu_temp = resource_metrics.avg_gpu_temperature[device_idx] if device_idx < len(resource_metrics.avg_gpu_temperature) else None
                
                exporter.record_gpu_metrics(
                    device_id=device_idx,
                    utilization=gpu_util / 100.0,  # Convert percentage to 0.0-1.0
                    memory_used=gpu_mem_used,
                    memory_total=gpu_mem_total,
                    temperature=gpu_temp,
                )
        except Exception as e:
            logger.debug(f"Failed to export resource metrics to Prometheus: {e}")

    def _execute_stage(self, dataset: Dataset, stage: Any, stage_index: int) -> Dataset:
        """Execute a single pipeline stage.

        Applies per-stage resource configuration before execution.

        Args:
            dataset: Input dataset
            stage: Pipeline stage to execute
            stage_index: Index of stage in pipeline

        Returns:
            Processed dataset
        """
        # Log stage resource configuration
        stage_name = getattr(stage, 'name', stage.__class__.__name__)
        batch_size = getattr(stage, 'batch_size', 'default')
        num_gpus = getattr(stage, 'num_gpus', 'default')
        num_cpus = getattr(stage, 'num_cpus', 'default')
        
        logger.debug(
            f"Executing stage {stage_name} with "
            f"batch_size={batch_size}, num_gpus={num_gpus}, num_cpus={num_cpus}"
        )
        
        return stage.process(dataset)

    def _perform_periodic_cleanup(self, stage_index: int) -> None:
        """Perform periodic cleanup every N stages.

        Args:
            stage_index: Current stage index
        """
        if stage_index % 3 == 0:
            import gc

            gc.collect()

            if self.config.num_gpus > 0:
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass

            if ray.is_initialized():
                try:
                    from pipeline.utils.ray.monitoring import log_cluster_status

                    log_cluster_status()
                except Exception:
                    pass

    def _initialize_checkpointing(self) -> Optional[Any]:
        """Initialize checkpoint manager if enabled.

        Returns:
            Checkpoint manager instance or None
        """
        if self.config.checkpoint_interval > 0:
            try:
                from pipeline.utils.execution.checkpoint import create_checkpoint_manager

                checkpoint_dir = str(Path(self.config.output_path).parent / "checkpoints")
                checkpoint_manager = create_checkpoint_manager(
                    checkpoint_dir=checkpoint_dir,
                    checkpoint_interval=self.config.checkpoint_interval,
                )
                logger.info(f"Checkpointing enabled: {checkpoint_dir}")
                return checkpoint_manager
            except (ImportError, IOError, OSError, ValueError, RuntimeError) as e:
                logger.warning(f"Failed to initialize checkpointing: {e}")
        return None

    def _save_checkpoint(
        self,
        checkpoint_manager: Any,
        dataset: Dataset,
        stage: Any,
        stage_index: int,
    ) -> None:
        """Save checkpoint after stage execution.

        Args:
            checkpoint_manager: Checkpoint manager instance
            dataset: Current dataset state
            stage: Current stage
            stage_index: Current stage index
        """
        pipeline_state = {
            "stage": stage.__class__.__name__,
            "stage_index": stage_index,
            "total_stages": len(self.config.stages) if hasattr(self.config, "stages") else 0,
            "metrics": self.metrics.get_current_metrics(),
        }
        checkpoint_manager.save_checkpoint(dataset=dataset, pipeline_state=pipeline_state)

    def _save_checkpoint_on_error(
        self,
        checkpoint_manager: Any,
        dataset: Dataset,
        stage: Any,
        stage_index: int,
        error: Exception,
    ) -> None:
        """Save checkpoint on stage error.

        Args:
            checkpoint_manager: Checkpoint manager instance
            dataset: Current dataset state
            stage: Failed stage
            stage_index: Failed stage index
            error: Error that occurred
        """
        try:
            pipeline_state = {
                "stage": stage.__class__.__name__,
                "stage_index": stage_index,
                "error": str(error),
                "metrics": self.metrics.get_current_metrics(),
            }
            checkpoint_manager.save_checkpoint(dataset=dataset, pipeline_state=pipeline_state)
            logger.info("Checkpoint saved before failure")
        except (IOError, OSError, RuntimeError) as checkpoint_error:
            logger.error(f"Failed to save checkpoint: {checkpoint_error}")

    def write_output(self, dataset: Dataset) -> None:
        """Write processed dataset to output path.

        Args:
            dataset: Processed dataset to write
        """
        logger.info(f"Writing curated data to {self.config.output_path}")

        self._validate_output_path()
        self._check_disk_space()

        # Get compression settings from config
        compression = self.config.output_config.get("compression", "snappy") if hasattr(self.config, "output_config") else "snappy"
        num_rows_per_file = self.config.output_config.get("num_rows_per_file", 1000000) if hasattr(self.config, "output_config") else 1000000
        
        # Apply partitioning if configured
        partition_by = self.config.output_config.get("partition_by") if hasattr(self.config, "output_config") else None
        if partition_by:
            self._write_partitioned(dataset, partition_by, compression, num_rows_per_file)
        else:
            dataset.write_parquet(
                self.config.output_path,
                compression=compression,
                num_rows_per_file=num_rows_per_file,
            )

        # Export to Omniverse USD if configured
        if getattr(self.config, "enable_omniverse_export", False):
            omniverse_export_path = getattr(self.config, "omniverse_export_path", None)
            if omniverse_export_path:
                try:
                    from pipeline.integrations.omniverse import OmniverseLoader

                    loader = OmniverseLoader(omniverse_path=omniverse_export_path)
                    loader.export_to_usd(dataset, omniverse_export_path)
                    logger.info(f"Exported curated data to Omniverse USD: {omniverse_export_path}")
                except Exception as e:
                    logger.warning(f"Failed to export to Omniverse USD: {e}")

    def _write_partitioned(
        self,
        dataset: Dataset,
        partition_by: list[str],
        compression: str,
        num_rows_per_file: int,
    ) -> None:
        """Write dataset with partitioning.

        Args:
            dataset: Dataset to write
            partition_by: List of fields to partition by
            compression: Compression algorithm
            num_rows_per_file: Rows per file
        """
        logger.info(f"Writing partitioned data (partition_by={partition_by})")
        
        # Group by partition keys and write separately
        # Note: This is a simplified implementation
        # For production, would use Ray Data's native partitioning or Dask
        
        def add_partition_keys_batch(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
            """Add partition key to batch."""
            result = []
            for item in batch:
                item_copy = dict(item)
                # Create partition key from partition_by fields
                partition_values = []
                for field in partition_by:
                    value = item.get(field) or item.get("metadata", {}).get(field) or "unknown"
                    partition_values.append(str(value))
                item_copy["_partition_key"] = "/".join(partition_values)
                result.append(item_copy)
            return result
        
        partitioned_dataset = dataset.map_batches(
            add_partition_keys_batch,
            batch_size=self.config.batch_size,
            batch_format="pandas",
        )
        
        # Get unique partition keys
        partition_keys = set()
        for batch in partitioned_dataset.iter_batches(batch_size=1000, prefetch_batches=0):
            for item in batch:
                if "_partition_key" in item:
                    partition_keys.add(item["_partition_key"])
            if len(partition_keys) > 100:  # Limit to prevent too many partitions
                break
        
        # Write each partition
        from pathlib import Path
        output_path = Path(self.config.output_path)
        
        for partition_key in partition_keys:
            partition_path = output_path / partition_key
            
            def filter_partition(item: dict[str, Any]) -> bool:
                """Filter items for this partition."""
                return item.get("_partition_key") == partition_key
            
            partition_data = partitioned_dataset.filter(filter_partition)
            
            # Remove partition key before writing
            def remove_partition_key_batch(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
                """Remove partition key from batch."""
                return [
                    {k: v for k, v in item.items() if k != "_partition_key"}
                    for item in batch
                ]
            
            partition_data = partition_data.map_batches(
                remove_partition_key_batch,
                batch_size=self.config.batch_size,
                batch_format="pandas",
            )
            
            partition_data.write_parquet(
                str(partition_path),
                compression=compression,
                num_rows_per_file=num_rows_per_file,
            )
            logger.info(f"Wrote partition {partition_key} to {partition_path}")

    def _validate_output_path(self) -> None:
        """Validate output path is writable."""
        try:
            from pipeline.utils.resource_manager import validate_path

            validate_path(self.config.output_path, check_writable=True)
        except (ImportError, ValueError, OSError) as e:
            logger.error(f"Output path validation failed: {e}")
            raise

    def _check_disk_space(self) -> None:
        """Check available disk space."""
        try:
            from pipeline.utils.resource_manager import check_disk_space
            from pipeline.utils.constants import _DEFAULT_ESTIMATED_OUTPUT_SIZE_BYTES

            estimated_size = _DEFAULT_ESTIMATED_OUTPUT_SIZE_BYTES
            has_space, available = check_disk_space(self.config.output_path, estimated_size)

            if not has_space:
                raise RuntimeError(
                    f"Insufficient disk space for output: "
                    f"required ~{estimated_size}, available {available}"
                )
        except (ImportError, RuntimeError) as e:
            logger.error(f"Disk space check failed: {e}")
            raise

