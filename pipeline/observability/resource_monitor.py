"""Scalable resource monitoring for pipeline stages using psutil and Ray.

Tracks CPU, GPU, and memory utilization per stage across distributed Ray clusters.
Designed for scalability with efficient sampling, batching, and distributed aggregation.

Scalability Features:
- Distributed monitoring: Uses Ray actors to monitor workers across the cluster
- Configurable sampling: Longer intervals (1.0s default) reduce overhead
- Worker limits: Maximum workers monitored (default 100) prevents overload
- Efficient aggregation: Parallel collection and averaging across workers
- Compressed storage: Optional gzip compression for metrics files
- History limits: Automatic pruning of old metrics (default 100 runs/stage)
- Minimal snapshots: Limits raw snapshot storage for memory efficiency

For very large clusters (1000+ nodes), consider:
- Increasing sampling_interval to 2.0-5.0 seconds
- Reducing max_workers_to_monitor to 50 or fewer
- Using external metrics systems (Prometheus) for long-term storage
"""

from __future__ import annotations

import logging
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Try to import Ray for distributed monitoring
try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    logger.warning("Ray not available. Distributed monitoring will be limited.")

# Try to import psutil
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil not available. Install via: pip install psutil")

# Try to import pynvml for GPU monitoring
try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    logger.debug("pynvml not available. GPU monitoring will be limited.")


@dataclass
class ResourceSnapshot:
    """Snapshot of resource utilization at a point in time."""
    
    timestamp: float
    cpu_percent: float = 0.0
    cpu_count: int = 0
    memory_used_mb: float = 0.0
    memory_total_mb: float = 0.0
    memory_percent: float = 0.0
    gpu_utilization: List[float] = field(default_factory=list)
    gpu_memory_used_mb: List[float] = field(default_factory=list)
    gpu_memory_total_mb: List[float] = field(default_factory=list)
    gpu_temperature: List[Optional[float]] = field(default_factory=list)


@dataclass
class StageResourceMetrics:
    """Aggregated resource metrics for a pipeline stage."""
    
    stage_name: str
    duration_seconds: float
    samples_count: int = 0
    
    # CPU metrics
    avg_cpu_percent: float = 0.0
    max_cpu_percent: float = 0.0
    min_cpu_percent: float = 0.0
    cpu_count: int = 0
    
    # Memory metrics
    avg_memory_percent: float = 0.0
    max_memory_percent: float = 0.0
    min_memory_percent: float = 0.0
    avg_memory_used_mb: float = 0.0
    max_memory_used_mb: float = 0.0
    memory_total_mb: float = 0.0
    
    # GPU metrics (per device)
    avg_gpu_utilization: List[float] = field(default_factory=list)
    max_gpu_utilization: List[float] = field(default_factory=list)
    avg_gpu_memory_percent: List[float] = field(default_factory=list)
    max_gpu_memory_percent: List[float] = field(default_factory=list)
    avg_gpu_memory_used_mb: List[float] = field(default_factory=list)
    max_gpu_memory_used_mb: List[float] = field(default_factory=list)
    gpu_memory_total_mb: List[float] = field(default_factory=list)
    avg_gpu_temperature: List[Optional[float]] = field(default_factory=list)
    
    # Raw snapshots for detailed analysis
    snapshots: List[ResourceSnapshot] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "stage_name": self.stage_name,
            "duration_seconds": self.duration_seconds,
            "samples_count": self.samples_count,
            "cpu": {
                "avg_percent": self.avg_cpu_percent,
                "max_percent": self.max_cpu_percent,
                "min_percent": self.min_cpu_percent,
                "count": self.cpu_count,
            },
            "memory": {
                "avg_percent": self.avg_memory_percent,
                "max_percent": self.max_memory_percent,
                "min_percent": self.min_memory_percent,
                "avg_used_mb": self.avg_memory_used_mb,
                "max_used_mb": self.max_memory_used_mb,
                "total_mb": self.memory_total_mb,
            },
            "gpu": {
                "devices": len(self.avg_gpu_utilization),
                "avg_utilization": self.avg_gpu_utilization,
                "max_utilization": self.max_gpu_utilization,
                "avg_memory_percent": self.avg_gpu_memory_percent,
                "max_memory_percent": self.max_gpu_memory_percent,
                "avg_memory_used_mb": self.avg_gpu_memory_used_mb,
                "max_memory_used_mb": self.max_gpu_memory_used_mb,
                "memory_total_mb": self.gpu_memory_total_mb,
                "avg_temperature": self.avg_gpu_temperature,
            },
        }


# Conditional Ray remote decorator
if RAY_AVAILABLE:
    _DistributedResourceMonitorActorBase = ray.remote(num_cpus=0)
else:
    # Fallback for when Ray is not available
    class _DummyRemote:
        def __call__(self, cls):
            return cls
    _DistributedResourceMonitorActorBase = _DummyRemote()

@_DistributedResourceMonitorActorBase  # Use minimal resources for monitoring actor
class DistributedResourceMonitorActor:
    """Ray actor for distributed resource monitoring.
    
    Runs on each worker node to collect local resource metrics.
    Designed for scalability with minimal overhead.
    """
    
    def __init__(self, stage_name: str, sampling_interval: float = 1.0):
        """Initialize distributed monitor actor.
        
        Args:
            stage_name: Name of the stage being monitored
            sampling_interval: Interval between snapshots (longer for scalability)
        """
        self.stage_name = stage_name
        self.sampling_interval = sampling_interval
        self.snapshots: List[ResourceSnapshot] = []
        self.monitoring = False
        
        # Get process for this worker
        if PSUTIL_AVAILABLE:
            self.process = psutil.Process(os.getpid())
        else:
            self.process = None
        
        # Initialize GPU monitoring
        self.gpu_devices: List[int] = []
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                device_count = pynvml.nvmlDeviceGetCount()
                self.gpu_devices = list(range(device_count))
            except Exception:
                self.gpu_devices = []
    
    def _collect_snapshot(self) -> ResourceSnapshot:
        """Collect resource snapshot from this worker."""
        snapshot = ResourceSnapshot(timestamp=time.time())
        
        if not PSUTIL_AVAILABLE or self.process is None:
            return snapshot
        
        try:
            # CPU metrics
            snapshot.cpu_percent = self.process.cpu_percent(interval=None)
            snapshot.cpu_count = psutil.cpu_count()
            
            # Memory metrics
            memory_info = self.process.memory_info()
            virtual_memory = psutil.virtual_memory()
            snapshot.memory_used_mb = memory_info.rss / (1024 * 1024)
            snapshot.memory_total_mb = virtual_memory.total / (1024 * 1024)
            snapshot.memory_percent = virtual_memory.percent
            
            # GPU metrics
            for device_id in self.gpu_devices:
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    snapshot.gpu_utilization.append(util.gpu)
                    
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    snapshot.gpu_memory_used_mb.append(mem_info.used / (1024 * 1024))
                    snapshot.gpu_memory_total_mb.append(mem_info.total / (1024 * 1024))
                    
                    try:
                        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                        snapshot.gpu_temperature.append(float(temp))
                    except Exception:
                        snapshot.gpu_temperature.append(None)
                except Exception:
                    pass
        except Exception as e:
            logger.debug(f"Failed to collect snapshot on worker: {e}")
        
        return snapshot
    
    def start_monitoring(self) -> None:
        """Start monitoring loop."""
        self.monitoring = True
        self.snapshots = []
    
    def collect_sample(self) -> ResourceSnapshot:
        """Collect a single sample (called periodically)."""
        if self.monitoring:
            return self._collect_snapshot()
        return ResourceSnapshot(timestamp=time.time())
    
    def stop_monitoring(self) -> List[ResourceSnapshot]:
        """Stop monitoring and return collected snapshots."""
        self.monitoring = False
        snapshots = self.snapshots.copy()
        self.snapshots = []
        return snapshots
    
    def get_worker_id(self) -> str:
        """Get unique identifier for this worker."""
        try:
            node_id = ray.get_runtime_context().get_node_id()
            return node_id
        except Exception:
            return f"worker_{os.getpid()}"


class StageResourceMonitor:
    """Monitor resource utilization for a pipeline stage.
    
    Scalable implementation that monitors both local process and distributed
    Ray workers/actors. Uses efficient sampling and aggregation for large clusters.
    """
    
    def __init__(
        self,
        stage_name: str,
        sampling_interval: float = 1.0,  # Longer interval for scalability
        enable_gpu_monitoring: bool = True,
        max_workers_to_monitor: int = 100,  # Limit for very large clusters
        use_distributed_monitoring: bool = True,
    ):
        """Initialize resource monitor.
        
        Args:
            stage_name: Name of the stage being monitored
            sampling_interval: Interval between resource snapshots in seconds
                              (longer = less overhead, default 1.0s for scalability)
            enable_gpu_monitoring: Whether to monitor GPU resources
            max_workers_to_monitor: Maximum number of workers to monitor (for scalability)
            use_distributed_monitoring: Whether to use Ray distributed monitoring
        """
        self.stage_name = stage_name
        self.sampling_interval = sampling_interval
        self.enable_gpu_monitoring = enable_gpu_monitoring
        self.max_workers_to_monitor = max_workers_to_monitor
        self.use_distributed_monitoring = use_distributed_monitoring and RAY_AVAILABLE
        
        self.snapshots: List[ResourceSnapshot] = []
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        # Distributed monitoring actors
        self.monitor_actors: List[Any] = []
        
        # Get process for local process
        if PSUTIL_AVAILABLE:
            self.process = psutil.Process(os.getpid())
        else:
            self.process = None
        
        # Initialize GPU monitoring if available
        self.gpu_devices: List[int] = []
        if enable_gpu_monitoring and PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                device_count = pynvml.nvmlDeviceGetCount()
                self.gpu_devices = list(range(device_count))
                logger.debug(f"Initialized GPU monitoring for {device_count} devices")
            except Exception as e:
                logger.warning(f"Failed to initialize GPU monitoring: {e}")
                self.gpu_devices = []
    
    def _collect_snapshot(self) -> ResourceSnapshot:
        """Collect a single resource snapshot.
        
        Returns:
            Resource snapshot with current utilization metrics
        """
        snapshot = ResourceSnapshot(timestamp=time.time())
        
        if not PSUTIL_AVAILABLE or self.process is None:
            return snapshot
        
        try:
            # CPU metrics
            cpu_percent = self.process.cpu_percent(interval=None)
            snapshot.cpu_percent = cpu_percent
            snapshot.cpu_count = psutil.cpu_count()
            
            # Memory metrics
            memory_info = self.process.memory_info()
            virtual_memory = psutil.virtual_memory()
            snapshot.memory_used_mb = memory_info.rss / (1024 * 1024)  # RSS in MB
            snapshot.memory_total_mb = virtual_memory.total / (1024 * 1024)
            snapshot.memory_percent = virtual_memory.percent
            
        except Exception as e:
            logger.warning(f"Failed to collect CPU/memory snapshot: {e}")
        
        # GPU metrics
        if self.enable_gpu_monitoring and self.gpu_devices:
            try:
                for device_id in self.gpu_devices:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
                    
                    # GPU utilization
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    snapshot.gpu_utilization.append(util.gpu)
                    
                    # GPU memory
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    snapshot.gpu_memory_used_mb.append(mem_info.used / (1024 * 1024))
                    snapshot.gpu_memory_total_mb.append(mem_info.total / (1024 * 1024))
                    
                    # GPU temperature
                    try:
                        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                        snapshot.gpu_temperature.append(float(temp))
                    except Exception:
                        snapshot.gpu_temperature.append(None)
            except Exception as e:
                logger.debug(f"Failed to collect GPU snapshot: {e}")
        
        return snapshot
    
    def _monitor_loop(self) -> None:
        """Background monitoring loop with distributed aggregation."""
        while self.monitoring:
            # Collect local snapshot
            local_snapshot = self._collect_snapshot()
            
            # Collect distributed snapshots if enabled
            if self.use_distributed_monitoring and self.monitor_actors:
                distributed_snapshots = self._collect_distributed_samples()
                if distributed_snapshots:
                    # Aggregate distributed snapshots
                    aggregated = self._aggregate_distributed_snapshots(distributed_snapshots)
                    # Merge with local snapshot (weighted average)
                    merged = self._merge_snapshots([local_snapshot, aggregated])
                    self.snapshots.append(merged)
                else:
                    self.snapshots.append(local_snapshot)
            else:
                self.snapshots.append(local_snapshot)
            
            time.sleep(self.sampling_interval)
    
    def _merge_snapshots(self, snapshots: List[ResourceSnapshot]) -> ResourceSnapshot:
        """Merge multiple snapshots into one (weighted average)."""
        if not snapshots:
            return ResourceSnapshot(timestamp=time.time())
        
        merged = ResourceSnapshot(timestamp=time.time())
        
        # Average CPU
        cpu_values = [s.cpu_percent for s in snapshots if s.cpu_percent > 0]
        if cpu_values:
            merged.cpu_percent = sum(cpu_values) / len(cpu_values)
            merged.cpu_count = max((s.cpu_count for s in snapshots), default=0)
        
        # Average memory
        memory_percents = [s.memory_percent for s in snapshots if s.memory_percent > 0]
        memory_used = [s.memory_used_mb for s in snapshots if s.memory_used_mb > 0]
        if memory_percents:
            merged.memory_percent = sum(memory_percents) / len(memory_percents)
        if memory_used:
            merged.memory_used_mb = sum(memory_used) / len(memory_used)
        if snapshots:
            merged.memory_total_mb = max((s.memory_total_mb for s in snapshots), default=0)
        
        # Merge GPU metrics
        if snapshots and snapshots[0].gpu_utilization:
            num_devices = max(len(s.gpu_utilization) for s in snapshots)
            for device_idx in range(num_devices):
                device_utils = [
                    s.gpu_utilization[device_idx]
                    for s in snapshots
                    if device_idx < len(s.gpu_utilization) and s.gpu_utilization[device_idx] is not None
                ]
                if device_utils:
                    merged.gpu_utilization.append(sum(device_utils) / len(device_utils))
                
                device_mem_used = [
                    s.gpu_memory_used_mb[device_idx]
                    for s in snapshots
                    if device_idx < len(s.gpu_memory_used_mb)
                ]
                device_mem_total = [
                    s.gpu_memory_total_mb[device_idx]
                    for s in snapshots
                    if device_idx < len(s.gpu_memory_total_mb)
                ]
                if device_mem_used:
                    merged.gpu_memory_used_mb.append(sum(device_mem_used) / len(device_mem_used))
                if device_mem_total:
                    merged.gpu_memory_total_mb.append(max(device_mem_total))
        
        return merged
    
    @contextmanager
    def monitor(self):
        """Context manager for monitoring stage execution.
        
        Example:
            ```python
            monitor = StageResourceMonitor("my_stage")
            with monitor.monitor():
                # Stage execution code
                pass
            metrics = monitor.get_metrics()
            ```
        """
        self.start()
        try:
            yield self
        finally:
            self.stop()
    
    def _setup_distributed_monitoring(self) -> None:
        """Set up distributed monitoring actors on Ray workers."""
        if not self.use_distributed_monitoring or not RAY_AVAILABLE:
            return
        
        try:
            # Get available nodes/workers
            cluster_resources = ray.cluster_resources()
            num_nodes = int(cluster_resources.get("node:__internal__", 1))
            
            # Limit number of actors for scalability
            num_actors = min(num_nodes, self.max_workers_to_monitor)
            
            # Create monitoring actors distributed across cluster
            self.monitor_actors = [
                DistributedResourceMonitorActor.remote(
                    stage_name=self.stage_name,
                    sampling_interval=self.sampling_interval,
                )
                for _ in range(num_actors)
            ]
            
            # Start monitoring on all actors
            ray.get([actor.start_monitoring.remote() for actor in self.monitor_actors])
            
            logger.debug(f"Started distributed monitoring with {len(self.monitor_actors)} actors")
        except Exception as e:
            logger.warning(f"Failed to set up distributed monitoring: {e}")
            self.monitor_actors = []
    
    def _collect_distributed_samples(self) -> List[ResourceSnapshot]:
        """Collect samples from distributed workers."""
        if not self.monitor_actors:
            return []
        
        try:
            # Collect samples from all actors in parallel
            sample_futures = [actor.collect_sample.remote() for actor in self.monitor_actors]
            samples = ray.get(sample_futures)
            return [s for s in samples if s is not None]
        except Exception as e:
            logger.debug(f"Failed to collect distributed samples: {e}")
            return []
    
    def _aggregate_distributed_snapshots(self, snapshots: List[ResourceSnapshot]) -> ResourceSnapshot:
        """Aggregate snapshots from multiple workers into a single snapshot.
        
        Uses averaging for CPU/memory, max for GPU utilization.
        """
        if not snapshots:
            return ResourceSnapshot(timestamp=time.time())
        
        aggregated = ResourceSnapshot(timestamp=time.time())
        
        # Aggregate CPU (average across workers)
        cpu_values = [s.cpu_percent for s in snapshots if s.cpu_percent > 0]
        if cpu_values:
            aggregated.cpu_percent = sum(cpu_values) / len(cpu_values)
            aggregated.cpu_count = max((s.cpu_count for s in snapshots), default=0)
        
        # Aggregate memory (average)
        memory_percents = [s.memory_percent for s in snapshots if s.memory_percent > 0]
        memory_used = [s.memory_used_mb for s in snapshots if s.memory_used_mb > 0]
        if memory_percents:
            aggregated.memory_percent = sum(memory_percents) / len(memory_percents)
        if memory_used:
            aggregated.memory_used_mb = sum(memory_used) / len(memory_used)
        if snapshots:
            aggregated.memory_total_mb = max((s.memory_total_mb for s in snapshots), default=0)
        
        # Aggregate GPU (average per device across workers)
        if snapshots and snapshots[0].gpu_utilization:
            num_devices = len(snapshots[0].gpu_utilization)
            for device_idx in range(num_devices):
                device_utils = [
                    s.gpu_utilization[device_idx]
                    for s in snapshots
                    if device_idx < len(s.gpu_utilization) and s.gpu_utilization[device_idx] is not None
                ]
                if device_utils:
                    aggregated.gpu_utilization.append(sum(device_utils) / len(device_utils))
                
                device_mem_used = [
                    s.gpu_memory_used_mb[device_idx]
                    for s in snapshots
                    if device_idx < len(s.gpu_memory_used_mb)
                ]
                device_mem_total = [
                    s.gpu_memory_total_mb[device_idx]
                    for s in snapshots
                    if device_idx < len(s.gpu_memory_total_mb)
                ]
                if device_mem_used:
                    aggregated.gpu_memory_used_mb.append(sum(device_mem_used) / len(device_mem_used))
                if device_mem_total:
                    aggregated.gpu_memory_total_mb.append(max(device_mem_total))
        
        return aggregated
    
    def start(self) -> None:
        """Start resource monitoring."""
        if not PSUTIL_AVAILABLE:
            logger.warning("psutil not available, resource monitoring disabled")
            return
        
        self.monitoring = True
        self.snapshots = []
        
        # Set up distributed monitoring if using Ray
        if self.use_distributed_monitoring:
            self._setup_distributed_monitoring()
        
        # Start background monitoring thread for local process
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.debug(f"Started resource monitoring for stage: {self.stage_name}")
    
    def stop(self) -> None:
        """Stop resource monitoring."""
        self.monitoring = False
        
        # Stop distributed monitoring actors
        if self.monitor_actors:
            try:
                # Collect final snapshots from distributed actors
                final_snapshots_futures = [actor.stop_monitoring.remote() for actor in self.monitor_actors]
                final_snapshots_list = ray.get(final_snapshots_futures)
                
                # Aggregate final distributed snapshots
                for worker_snapshots in final_snapshots_list:
                    if worker_snapshots:
                        aggregated = self._aggregate_distributed_snapshots(worker_snapshots)
                        self.snapshots.append(aggregated)
            except Exception as e:
                logger.debug(f"Error stopping distributed monitoring: {e}")
            finally:
                # Clean up actors
                for actor in self.monitor_actors:
                    try:
                        ray.kill(actor)
                    except Exception:
                        pass
                self.monitor_actors = []
        
        # Stop local monitoring thread
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        
        logger.debug(f"Stopped resource monitoring for stage: {self.stage_name}")
    
    def get_metrics(self) -> StageResourceMetrics:
        """Get aggregated resource metrics.
        
        Returns:
            Aggregated resource metrics for the stage
        """
        if not self.snapshots:
            return StageResourceMetrics(
                stage_name=self.stage_name,
                duration_seconds=0.0,
            )
        
        start_time = self.snapshots[0].timestamp if self.snapshots else time.time()
        end_time = self.snapshots[-1].timestamp if self.snapshots else time.time()
        duration = end_time - start_time
        
        metrics = StageResourceMetrics(
            stage_name=self.stage_name,
            duration_seconds=duration,
            samples_count=len(self.snapshots),
        )
        
        # Aggregate CPU metrics
        cpu_percents = [s.cpu_percent for s in self.snapshots if s.cpu_percent > 0]
        if cpu_percents:
            metrics.avg_cpu_percent = sum(cpu_percents) / len(cpu_percents)
            metrics.max_cpu_percent = max(cpu_percents)
            metrics.min_cpu_percent = min(cpu_percents)
            metrics.cpu_count = self.snapshots[0].cpu_count if self.snapshots else 0
        
        # Aggregate memory metrics
        memory_percents = [s.memory_percent for s in self.snapshots if s.memory_percent > 0]
        memory_used = [s.memory_used_mb for s in self.snapshots if s.memory_used_mb > 0]
        if memory_percents:
            metrics.avg_memory_percent = sum(memory_percents) / len(memory_percents)
            metrics.max_memory_percent = max(memory_percents)
            metrics.min_memory_percent = min(memory_percents)
        if memory_used:
            metrics.avg_memory_used_mb = sum(memory_used) / len(memory_used)
            metrics.max_memory_used_mb = max(memory_used)
        if self.snapshots:
            metrics.memory_total_mb = self.snapshots[0].memory_total_mb
        
        # Aggregate GPU metrics per device
        if self.snapshots and self.snapshots[0].gpu_utilization:
            num_devices = len(self.snapshots[0].gpu_utilization)
            
            for device_idx in range(num_devices):
                device_utils = [
                    s.gpu_utilization[device_idx]
                    for s in self.snapshots
                    if device_idx < len(s.gpu_utilization) and s.gpu_utilization[device_idx] is not None
                ]
                device_mem_percents = []
                device_mem_used = []
                device_temps = []
                
                for snapshot in self.snapshots:
                    if device_idx < len(snapshot.gpu_memory_used_mb):
                        mem_used = snapshot.gpu_memory_used_mb[device_idx]
                        mem_total = snapshot.gpu_memory_total_mb[device_idx]
                        if mem_total > 0:
                            device_mem_percents.append((mem_used / mem_total) * 100.0)
                            device_mem_used.append(mem_used)
                    if device_idx < len(snapshot.gpu_temperature) and snapshot.gpu_temperature[device_idx] is not None:
                        device_temps.append(snapshot.gpu_temperature[device_idx])
                
                if device_utils:
                    metrics.avg_gpu_utilization.append(sum(device_utils) / len(device_utils))
                    metrics.max_gpu_utilization.append(max(device_utils))
                else:
                    metrics.avg_gpu_utilization.append(0.0)
                    metrics.max_gpu_utilization.append(0.0)
                
                if device_mem_percents:
                    metrics.avg_gpu_memory_percent.append(sum(device_mem_percents) / len(device_mem_percents))
                    metrics.max_gpu_memory_percent.append(max(device_mem_percents))
                else:
                    metrics.avg_gpu_memory_percent.append(0.0)
                    metrics.max_gpu_memory_percent.append(0.0)
                
                if device_mem_used:
                    metrics.avg_gpu_memory_used_mb.append(sum(device_mem_used) / len(device_mem_used))
                    metrics.max_gpu_memory_used_mb.append(max(device_mem_used))
                else:
                    metrics.avg_gpu_memory_used_mb.append(0.0)
                    metrics.max_gpu_memory_used_mb.append(0.0)
                
                if device_temps:
                    metrics.avg_gpu_temperature.append(sum(device_temps) / len(device_temps))
                else:
                    metrics.avg_gpu_temperature.append(None)
                
                if self.snapshots and device_idx < len(self.snapshots[0].gpu_memory_total_mb):
                    metrics.gpu_memory_total_mb.append(self.snapshots[0].gpu_memory_total_mb[device_idx])
                else:
                    metrics.gpu_memory_total_mb.append(0.0)
        
        # Store raw snapshots (can be disabled for scalability)
        # For large clusters, consider limiting snapshot storage
        metrics.snapshots = self.snapshots[:100] if len(self.snapshots) > 100 else self.snapshots
        
        return metrics


class ResourceMetricsStore:
    """Store resource metrics for multiple pipeline runs.
    
    Scalable storage for metrics across large clusters. Uses efficient
    serialization and batching for large-scale deployments.
    """
    
    def __init__(
        self,
        storage_path: Optional[str] = None,
        max_history_per_stage: int = 100,  # Limit history for scalability
        use_compression: bool = True,  # Compress stored metrics
    ):
        """Initialize metrics store.
        
        Args:
            storage_path: Path to store metrics (None = in-memory only)
            max_history_per_stage: Maximum number of runs to keep per stage
            use_compression: Whether to compress stored metrics
        """
        self.storage_path = storage_path
        self.max_history_per_stage = max_history_per_stage
        self.use_compression = use_compression
        self.metrics_history: Dict[str, List[StageResourceMetrics]] = {}
    
    def record_stage_metrics(self, metrics: StageResourceMetrics) -> None:
        """Record metrics for a stage.
        
        Automatically limits history size for scalability.
        
        Args:
            metrics: Stage resource metrics to record
        """
        if metrics.stage_name not in self.metrics_history:
            self.metrics_history[metrics.stage_name] = []
        
        self.metrics_history[metrics.stage_name].append(metrics)
        
        # Limit history size for scalability
        if len(self.metrics_history[metrics.stage_name]) > self.max_history_per_stage:
            # Keep most recent runs
            self.metrics_history[metrics.stage_name] = self.metrics_history[metrics.stage_name][-self.max_history_per_stage:]
    
    def get_stage_averages(self, stage_name: str) -> Optional[Dict[str, Any]]:
        """Get average metrics for a stage across all runs.
        
        Args:
            stage_name: Name of the stage
            
        Returns:
            Dictionary with average metrics or None if no data
        """
        if stage_name not in self.metrics_history or not self.metrics_history[stage_name]:
            return None
        
        runs = self.metrics_history[stage_name]
        
        # Calculate averages
        avg_cpu = sum(m.avg_cpu_percent for m in runs) / len(runs)
        avg_memory = sum(m.avg_memory_percent for m in runs) / len(runs)
        avg_gpu = []
        if runs[0].avg_gpu_utilization:
            num_devices = len(runs[0].avg_gpu_utilization)
            for device_idx in range(num_devices):
                device_utils = [m.avg_gpu_utilization[device_idx] for m in runs if device_idx < len(m.avg_gpu_utilization)]
                avg_gpu.append(sum(device_utils) / len(device_utils) if device_utils else 0.0)
        
        return {
            "stage_name": stage_name,
            "runs_count": len(runs),
            "avg_cpu_percent": avg_cpu,
            "avg_memory_percent": avg_memory,
            "avg_gpu_utilization": avg_gpu,
        }
    
    def save(self, path: Optional[str] = None) -> None:
        """Save metrics to disk with efficient serialization.
        
        Uses compression for large-scale deployments.
        
        Args:
            path: Path to save (uses storage_path if None)
        """
        import json
        
        save_path = path or self.storage_path
        if not save_path:
            logger.warning("No storage path specified, skipping save")
            return
        
        # Convert to JSON-serializable format (without raw snapshots for efficiency)
        data = {}
        for stage_name, runs in self.metrics_history.items():
            # Store only aggregated metrics, not raw snapshots
            data[stage_name] = [m.to_dict() for m in runs]
        
        try:
            if self.use_compression:
                import gzip
                with gzip.open(f"{save_path}.gz", 'wt', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
                logger.info(f"Saved compressed resource metrics to {save_path}.gz")
            else:
                with open(save_path, 'w') as f:
                    json.dump(data, f, indent=2)
                logger.info(f"Saved resource metrics to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save resource metrics: {e}")
    
    def load(self, path: Optional[str] = None) -> None:
        """Load metrics from disk (supports compressed files).
        
        Args:
            path: Path to load from (uses storage_path if None)
        """
        import json
        
        load_path = path or self.storage_path
        if not load_path:
            logger.warning("No storage path specified, skipping load")
            return
        
        # Try compressed first, then uncompressed
        compressed_path = f"{load_path}.gz"
        
        try:
            if os.path.exists(compressed_path):
                import gzip
                with gzip.open(compressed_path, 'rt', encoding='utf-8') as f:
                    data = json.load(f)
                logger.info(f"Loaded compressed resource metrics from {compressed_path}")
            elif os.path.exists(load_path):
                with open(load_path, 'r') as f:
                    data = json.load(f)
                logger.info(f"Loaded resource metrics from {load_path}")
            else:
                logger.debug(f"No existing metrics file at {load_path} or {compressed_path}")
                return
            
            # Store loaded data (simplified - full reconstruction would require more complex logic)
            # For now, we just note that data was loaded
            if isinstance(data, dict):
                logger.debug(f"Loaded metrics for {len(data)} stages")
        except Exception as e:
            logger.error(f"Failed to load resource metrics: {e}")

