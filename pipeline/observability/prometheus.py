"""Prometheus metrics exporter for pipeline observability.

Uses prometheus_client library (standard Prometheus Python client) instead of
custom implementation. Provides Prometheus-compatible metrics for integration
with Grafana dashboards and Prometheus monitoring.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Use prometheus_client library instead of custom implementation
try:
    from prometheus_client import Counter, Gauge, Histogram, start_http_server
    PROMETHEUS_CLIENT_AVAILABLE = True
except ImportError:
    PROMETHEUS_CLIENT_AVAILABLE = False
    Counter = None
    Gauge = None
    Histogram = None
    start_http_server = None
    logger.warning("prometheus_client not available. Install via: pip install prometheus-client")


class PrometheusMetricsExporter:
    """Export pipeline metrics using prometheus_client library.

    Uses standard prometheus_client library instead of custom implementation.
    Provides Prometheus-compatible metrics for integration with monitoring systems.
    """

    def __init__(
        self,
        enabled: bool = True,
        port: int = 9090,
        endpoint: str = "/metrics",
    ):
        """Initialize Prometheus metrics exporter.

        Args:
            enabled: Whether to enable Prometheus export
            port: Port for metrics HTTP server
            endpoint: Metrics endpoint path (not used with prometheus_client, kept for compatibility)
        """
        self.enabled = enabled
        self.port = port
        self.endpoint = endpoint
        
        if not PROMETHEUS_CLIENT_AVAILABLE:
            logger.warning("prometheus_client not available, metrics export disabled")
            self.enabled = False
            return
        
        # Initialize Prometheus metrics using prometheus_client
        # Stage metrics
        self.stage_duration = Histogram(
            "pipeline_stage_duration_seconds",
            "Stage execution duration in seconds",
            ["stage"],
            buckets=[0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0],
        )
        self.items_processed = Counter(
            "pipeline_items_processed_total",
            "Total items processed",
            ["stage"],
        )
        self.items_filtered = Counter(
            "pipeline_items_filtered_total",
            "Total items filtered",
            ["stage"],
        )
        self.errors_total = Counter(
            "pipeline_errors_total",
            "Total errors",
            ["stage"],
        )
        self.stage_throughput = Gauge(
            "pipeline_stage_throughput_items_per_second",
            "Stage throughput in items per second",
            ["stage"],
        )
        
        # GPU metrics
        self.gpu_utilization = Gauge(
            "pipeline_gpu_utilization",
            "GPU utilization (0.0-1.0)",
            ["device_id"],
        )
        self.gpu_memory_used = Gauge(
            "pipeline_gpu_memory_used_bytes",
            "GPU memory used in bytes",
            ["device_id"],
        )
        self.gpu_memory_total = Gauge(
            "pipeline_gpu_memory_total_bytes",
            "GPU total memory in bytes",
            ["device_id"],
        )
        self.gpu_memory_usage_ratio = Gauge(
            "pipeline_gpu_memory_usage_ratio",
            "GPU memory usage ratio (0.0-1.0)",
            ["device_id"],
        )
        self.gpu_temperature = Gauge(
            "pipeline_gpu_temperature_celsius",
            "GPU temperature in Celsius",
            ["device_id"],
        )
        
        # Pipeline metrics
        self.pipeline_total_samples = Gauge(
            "pipeline_total_samples",
            "Total samples processed",
        )
        self.pipeline_dedup_rate = Gauge(
            "pipeline_dedup_rate",
            "Deduplication rate (0.0-1.0)",
        )
        self.pipeline_duration = Gauge(
            "pipeline_duration_seconds",
            "Total pipeline duration in seconds",
        )
        self.pipeline_throughput = Gauge(
            "pipeline_throughput_samples_per_second",
            "Pipeline throughput in samples per second",
        )

    def record_counter(
        self,
        name: str,
        value: float = 1.0,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Record a counter metric (kept for compatibility, use direct metrics instead).

        Args:
            name: Metric name (will be prefixed with 'pipeline_')
            value: Counter increment value
            labels: Optional labels for the metric
        """
        if not self.enabled:
            return
        logger.warning("record_counter() is deprecated, use direct metric objects instead")

    def record_gauge(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Record a gauge metric (kept for compatibility, use direct metrics instead).

        Args:
            name: Metric name (will be prefixed with 'pipeline_')
            value: Gauge value
            labels: Optional labels for the metric
        """
        if not self.enabled:
            return
        logger.warning("record_gauge() is deprecated, use direct metric objects instead")

    def record_histogram(
        self,
        name: str,
        value: float,
        buckets: Optional[List[float]] = None,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Record a histogram metric (kept for compatibility, use direct metrics instead).

        Args:
            name: Metric name (will be prefixed with 'pipeline_')
            value: Histogram value
            buckets: Histogram buckets (ignored, uses predefined buckets)
            labels: Optional labels for the metric
        """
        if not self.enabled:
            return
        logger.warning("record_histogram() is deprecated, use direct metric objects instead")

    def record_stage_metrics(
        self,
        stage_name: str,
        duration: float,
        items_processed: int,
        items_filtered: int = 0,
        errors: int = 0,
    ) -> None:
        """Record metrics for a pipeline stage.

        Args:
            stage_name: Name of the stage
            duration: Stage duration in seconds
            items_processed: Number of items processed
            items_filtered: Number of items filtered
            errors: Number of errors
        """
        if not self.enabled:
            return
        
        labels = [stage_name]
        
        # Record metrics using prometheus_client
        self.stage_duration.labels(*labels).observe(duration)
        self.items_processed.labels(*labels).inc(items_processed)
        
        if items_filtered > 0:
            self.items_filtered.labels(*labels).inc(items_filtered)
        
        if errors > 0:
            self.errors_total.labels(*labels).inc(errors)
        
        # Throughput gauge
        throughput = items_processed / duration if duration > 0 else 0.0
        self.stage_throughput.labels(*labels).set(throughput)

    def record_gpu_metrics(
        self,
        device_id: int,
        utilization: float,
        memory_used: float,
        memory_total: float,
        temperature: Optional[float] = None,
    ) -> None:
        """Record GPU metrics.

        Args:
            device_id: GPU device ID
            utilization: GPU utilization (0.0-1.0)
            memory_used: Memory used in bytes
            memory_total: Total memory in bytes
            temperature: GPU temperature in Celsius (optional)
        """
        if not self.enabled:
            return
        
        labels = [str(device_id)]
        
        self.gpu_utilization.labels(*labels).set(utilization)
        self.gpu_memory_used.labels(*labels).set(memory_used)
        self.gpu_memory_total.labels(*labels).set(memory_total)
        self.gpu_memory_usage_ratio.labels(*labels).set(
            memory_used / memory_total if memory_total > 0 else 0.0
        )
        
        if temperature is not None:
            self.gpu_temperature.labels(*labels).set(temperature)

    def record_pipeline_metrics(
        self,
        total_samples: int,
        dedup_rate: float,
        total_duration: float,
    ) -> None:
        """Record overall pipeline metrics.

        Args:
            total_samples: Total number of samples processed
            dedup_rate: Deduplication rate (0.0-1.0)
            total_duration: Total pipeline duration in seconds
        """
        if not self.enabled:
            return
        
        self.pipeline_total_samples.set(total_samples)
        self.pipeline_dedup_rate.set(dedup_rate)
        self.pipeline_duration.set(total_duration)
        self.pipeline_throughput.set(
            total_samples / total_duration if total_duration > 0 else 0.0
        )

    def export_metrics(self) -> str:
        """Export metrics in Prometheus format (not needed with prometheus_client).

        prometheus_client handles export automatically via HTTP server.
        This method is kept for compatibility but returns empty string.

        Returns:
            Empty string (prometheus_client handles export)
        """
        # prometheus_client handles export automatically
        return ""

    def start_http_server(self) -> None:
        """Start HTTP server for Prometheus scraping using prometheus_client.

        Uses prometheus_client's built-in HTTP server instead of custom implementation.
        """
        if not self.enabled or not PROMETHEUS_CLIENT_AVAILABLE:
            return
        
        try:
            start_http_server(self.port)
            logger.info(f"Prometheus metrics server started on port {self.port}, endpoint: {self.endpoint}")
        except Exception as e:
            logger.error(f"Failed to start Prometheus metrics server: {e}")


def create_prometheus_exporter(
    enabled: bool = True,
    port: int = 9090,
    start_server: bool = True,
) -> PrometheusMetricsExporter:
    """Create and configure Prometheus metrics exporter.

    Args:
        enabled: Whether to enable Prometheus export
        port: Port for metrics HTTP server
        start_server: Whether to start HTTP server immediately

    Returns:
        PrometheusMetricsExporter instance
    """
    exporter = PrometheusMetricsExporter(enabled=enabled, port=port)
    
    if start_server and enabled:
        exporter.start_http_server()
    
    return exporter
