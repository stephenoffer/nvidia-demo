"""Grafana dashboard configuration and integration.

Provides Grafana dashboard JSON configurations for visualizing
pipeline metrics. Critical for GR00T: Grafana dashboards are essential
for monitoring production pipelines.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class GrafanaDashboardGenerator:
    """Generate Grafana dashboard configurations for pipeline monitoring."""

    def __init__(self, datasource_name: str = "Prometheus"):
        """Initialize Grafana dashboard generator.

        Args:
            datasource_name: Name of Prometheus datasource in Grafana
        """
        self.datasource_name = datasource_name

    def generate_pipeline_dashboard(self) -> Dict[str, Any]:
        """Generate Grafana dashboard JSON for pipeline monitoring.

        Returns:
            Grafana dashboard JSON configuration
        """
        dashboard = {
            "dashboard": {
                "title": "GR00T Data Pipeline Monitoring",
                "tags": ["pipeline", "groot", "ray", "nvidia"],
                "timezone": "browser",
                "schemaVersion": 27,
                "version": 1,
                "refresh": "10s",
                "panels": self._generate_panels(),
            }
        }
        return dashboard

    def _generate_panels(self) -> List[Dict[str, Any]]:
        """Generate dashboard panels.

        Returns:
            List of panel configurations
        """
        panels = []

        # Pipeline throughput panel
        panels.append({
            "id": 1,
            "title": "Pipeline Throughput",
            "type": "graph",
            "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
            "targets": [
                {
                    "expr": "rate(pipeline_pipeline_throughput_samples_per_second[5m])",
                    "legendFormat": "Samples/sec",
                    "refId": "A",
                }
            ],
            "yaxes": [
                {"format": "short", "label": "Samples/sec"},
                {"format": "short"},
            ],
        })

        # Stage duration panel
        panels.append({
            "id": 2,
            "title": "Stage Duration",
            "type": "graph",
            "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
            "targets": [
                {
                    "expr": "histogram_quantile(0.95, pipeline_stage_duration_seconds_bucket)",
                    "legendFormat": "{{stage}} (p95)",
                    "refId": "A",
                },
                {
                    "expr": "histogram_quantile(0.50, pipeline_stage_duration_seconds_bucket)",
                    "legendFormat": "{{stage}} (p50)",
                    "refId": "B",
                },
            ],
            "yaxes": [
                {"format": "s", "label": "Duration"},
                {"format": "short"},
            ],
        })

        # GPU utilization panel
        panels.append({
            "id": 3,
            "title": "GPU Utilization",
            "type": "graph",
            "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8},
            "targets": [
                {
                    "expr": "pipeline_gpu_utilization{device_id=\"0\"}",
                    "legendFormat": "GPU {{device_id}}",
                    "refId": "A",
                }
            ],
            "yaxes": [
                {"format": "percent", "label": "Utilization", "max": 1.0, "min": 0.0},
                {"format": "short"},
            ],
        })

        # GPU memory usage panel
        panels.append({
            "id": 4,
            "title": "GPU Memory Usage",
            "type": "graph",
            "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8},
            "targets": [
                {
                    "expr": "pipeline_gpu_memory_usage_ratio{device_id=\"0\"}",
                    "legendFormat": "GPU {{device_id}}",
                    "refId": "A",
                }
            ],
            "yaxes": [
                {"format": "percent", "label": "Memory Usage", "max": 1.0, "min": 0.0},
                {"format": "short"},
            ],
        })

        # Items processed by stage panel
        panels.append({
            "id": 5,
            "title": "Items Processed by Stage",
            "type": "graph",
            "gridPos": {"h": 8, "w": 12, "x": 0, "y": 16},
            "targets": [
                {
                    "expr": "rate(pipeline_items_processed_total[5m])",
                    "legendFormat": "{{stage}}",
                    "refId": "A",
                }
            ],
            "yaxes": [
                {"format": "short", "label": "Items/sec"},
                {"format": "short"},
            ],
        })

        # Error rate panel
        panels.append({
            "id": 6,
            "title": "Error Rate",
            "type": "graph",
            "gridPos": {"h": 8, "w": 12, "x": 12, "y": 16},
            "targets": [
                {
                    "expr": "rate(pipeline_errors_total[5m])",
                    "legendFormat": "{{stage}}",
                    "refId": "A",
                }
            ],
            "yaxes": [
                {"format": "short", "label": "Errors/sec"},
                {"format": "short"},
            ],
        })

        # Deduplication rate panel
        panels.append({
            "id": 7,
            "title": "Deduplication Rate",
            "type": "stat",
            "gridPos": {"h": 4, "w": 6, "x": 0, "y": 24},
            "targets": [
                {
                    "expr": "pipeline_pipeline_dedup_rate",
                    "refId": "A",
                }
            ],
            "options": {
                "unit": "percent",
                "decimals": 2,
            },
        })

        # Pipeline uptime panel
        panels.append({
            "id": 8,
            "title": "Pipeline Uptime",
            "type": "stat",
            "gridPos": {"h": 4, "w": 6, "x": 6, "y": 24},
            "targets": [
                {
                    "expr": "pipeline_pipeline_uptime_seconds",
                    "refId": "A",
                }
            ],
            "options": {
                "unit": "s",
            },
        })

        # Total samples panel
        panels.append({
            "id": 9,
            "title": "Total Samples Processed",
            "type": "stat",
            "gridPos": {"h": 4, "w": 6, "x": 12, "y": 24},
            "targets": [
                {
                    "expr": "pipeline_pipeline_total_samples",
                    "refId": "A",
                }
            ],
            "options": {
                "unit": "short",
            },
        })

        return panels

    def export_dashboard_json(self, output_path: str) -> None:
        """Export dashboard JSON to file.

        Args:
            output_path: Path to save dashboard JSON
        """
        dashboard = self.generate_pipeline_dashboard()
        
        with open(output_path, "w") as f:
            json.dump(dashboard, f, indent=2)
        
        logger.info(f"Grafana dashboard exported to {output_path}")

    def generate_alert_rules(self) -> List[Dict[str, Any]]:
        """Generate Prometheus alert rules for Grafana.

        Returns:
            List of alert rule configurations
        """
        alerts = [
            {
                "alert": "HighErrorRate",
                "expr": "rate(pipeline_errors_total[5m]) > 0.1",
                "for": "5m",
                "labels": {
                    "severity": "warning",
                },
                "annotations": {
                    "summary": "High error rate detected in pipeline",
                    "description": "Error rate is {{ $value }} errors/sec",
                },
            },
            {
                "alert": "LowThroughput",
                "expr": "pipeline_pipeline_throughput_samples_per_second < 100",
                "for": "10m",
                "labels": {
                    "severity": "warning",
                },
                "annotations": {
                    "summary": "Low pipeline throughput",
                    "description": "Throughput is {{ $value }} samples/sec",
                },
            },
            {
                "alert": "HighGPUUtilization",
                "expr": "pipeline_gpu_utilization > 0.95",
                "for": "5m",
                "labels": {
                    "severity": "warning",
                },
                "annotations": {
                    "summary": "High GPU utilization",
                    "description": "GPU {{ $labels.device_id }} utilization is {{ $value }}",
                },
            },
            {
                "alert": "HighGPUMemoryUsage",
                "expr": "pipeline_gpu_memory_usage_ratio > 0.9",
                "for": "5m",
                "labels": {
                    "severity": "critical",
                },
                "annotations": {
                    "summary": "High GPU memory usage",
                    "description": "GPU {{ $labels.device_id }} memory usage is {{ $value }}",
                },
            },
        ]
        return alerts

    def export_alert_rules(self, output_path: str) -> None:
        """Export alert rules to file.

        Args:
            output_path: Path to save alert rules JSON
        """
        alerts = {
            "groups": [
                {
                    "name": "pipeline_alerts",
                    "rules": self.generate_alert_rules(),
                }
            ]
        }
        
        with open(output_path, "w") as f:
            json.dump(alerts, f, indent=2)
        
        logger.info(f"Prometheus alert rules exported to {output_path}")


def create_grafana_dashboard(
    datasource_name: str = "Prometheus",
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Create Grafana dashboard configuration.

    Args:
        datasource_name: Name of Prometheus datasource
        output_path: Optional path to save dashboard JSON

    Returns:
        Grafana dashboard JSON
    """
    generator = GrafanaDashboardGenerator(datasource_name=datasource_name)
    dashboard = generator.generate_pipeline_dashboard()
    
    if output_path:
        generator.export_dashboard_json(output_path)
    
    return dashboard

