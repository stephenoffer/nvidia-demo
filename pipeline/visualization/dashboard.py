"""Interactive dashboard for visualizing pipeline results and simulation data.

Uses Grafana as the primary visualization tool for production-grade monitoring.
Grafana provides professional dashboards with Prometheus integration.

For local development, falls back to Plotly for interactive web-based visualization.
See: https://grafana.com/, https://prometheus.io/, and https://plotly.com/python/
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


class VisualizationDashboard:
    """Interactive dashboard for data visualization.

    Uses Grafana as the primary visualization backend for production monitoring.
    Falls back to Plotly for interactive web-based visualization.
    """

    def __init__(self, mode: str = "grafana", datasource_name: str = "Prometheus"):
        """Initialize visualization dashboard.

        Args:
            mode: Dashboard mode ('grafana' for production, 'plotly' for web-based, 'local' for plotly)
            datasource_name: Name of Prometheus datasource in Grafana (for grafana mode)
        """
        self.mode = mode
        self.datasource_name = datasource_name
        self._init_backend()

    def _init_backend(self) -> None:
        """Initialize visualization backend."""
        if self.mode == "grafana" or self.mode == "production":
            # Grafana is always available (generates JSON configs)
            self.backend = "grafana"
            logger.info("Using Grafana for visualization (production-grade monitoring)")
        elif self.mode == "plotly" or self.mode == "web" or self.mode == "local":
            # Try Plotly for interactive web-based visualization
            try:
                import plotly  # noqa: F401
                self.backend = "plotly"
                logger.info("Using Plotly for interactive web-based visualization")
            except ImportError:
                logger.warning("Plotly not available, falling back to Grafana")
                self.backend = "grafana"
        else:
            # Default to Grafana
            self.backend = "grafana"

    def create_pipeline_summary(
        self,
        metrics: Dict[str, Any],
        output_path: Optional[str] = None,
    ) -> None:
        """Create a summary visualization of pipeline metrics.

        Args:
            metrics: Pipeline metrics dictionary
            output_path: Optional path to save visualization (for Grafana, saves JSON config; for Plotly, saves HTML)
        """
        if self.backend == "grafana":
            self._create_grafana_summary(metrics, output_path)
        elif self.backend == "plotly":
            self._create_plotly_summary(metrics, output_path)
        else:
            # Fallback to basic text summary
            self._create_basic_summary(metrics, output_path)


    def _create_grafana_summary(
        self,
        metrics: Dict[str, Any],
        output_path: Optional[str],
    ) -> None:
        """Create Grafana dashboard summary.
        
        Generates a Grafana dashboard JSON configuration that can be imported
        into Grafana for professional monitoring and visualization.
        
        Args:
            metrics: Pipeline metrics dictionary
            output_path: Path to save Grafana dashboard JSON
        """
        try:
            from pipeline.observability.grafana import GrafanaDashboardGenerator
            
            generator = GrafanaDashboardGenerator(datasource_name=self.datasource_name)
            
            # Determine output path
            if output_path:
                # If output_path ends with .png, change to .json
                if output_path.endswith('.png'):
                    output_path = output_path.replace('.png', '_grafana.json')
                elif not output_path.endswith('.json'):
                    output_path = str(Path(output_path).parent / "grafana_dashboard.json")
            else:
                output_path = "grafana_dashboard.json"
            
            # Generate and export Grafana dashboard
            generator.export_dashboard_json(output_path)
            
            # Also generate alert rules
            alert_path = str(Path(output_path).parent / "prometheus_alerts.json")
            generator.export_alert_rules(alert_path)
            
            logger.info(f"Grafana dashboard exported to {output_path}")
            logger.info(f"Prometheus alert rules exported to {alert_path}")
            logger.info(
                f"To use this dashboard:\n"
                f"  1. Import {output_path} into Grafana\n"
                f"  2. Ensure Prometheus datasource '{self.datasource_name}' is configured\n"
                f"  3. Import {alert_path} into Prometheus Alertmanager"
            )
            
            # Optionally create a simple HTML summary with Grafana link
            self._create_grafana_html_summary(metrics, output_path)
            
        except ImportError:
            logger.warning("Grafana dashboard generator not available, falling back to Plotly")
            self._create_plotly_summary(metrics, output_path)
        except Exception as e:
            logger.error(f"Error creating Grafana dashboard: {e}", exc_info=True)
            logger.info("Falling back to Plotly visualization")
            self._create_plotly_summary(metrics, output_path)

    def _create_grafana_html_summary(
        self,
        metrics: Dict[str, Any],
        grafana_json_path: str,
    ) -> None:
        """Create a simple HTML summary page with Grafana integration instructions.
        
        Args:
            metrics: Pipeline metrics dictionary
            grafana_json_path: Path to Grafana dashboard JSON
        """
        try:
            html_path = str(Path(grafana_json_path).parent / "pipeline_summary.html")
            
            html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>GR00T Pipeline Summary</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #1f4788; border-bottom: 3px solid #1f4788; padding-bottom: 10px; }}
        .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 30px 0; }}
        .metric-card {{ background: #f8f9fa; padding: 20px; border-radius: 6px; border-left: 4px solid #1f4788; }}
        .metric-label {{ font-size: 14px; color: #666; margin-bottom: 8px; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #1f4788; }}
        .grafana-section {{ background: #e8f4f8; padding: 20px; border-radius: 6px; margin: 30px 0; }}
        .grafana-section h2 {{ color: #1f4788; margin-top: 0; }}
        code {{ background: #f0f0f0; padding: 2px 6px; border-radius: 3px; font-family: monospace; }}
        .instructions {{ background: white; padding: 15px; border-radius: 4px; margin-top: 15px; }}
        .instructions ol {{ margin: 10px 0; padding-left: 25px; }}
        .instructions li {{ margin: 8px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 GR00T Data Pipeline - Execution Summary</h1>
        
        <div class="metrics">
            <div class="metric-card">
                <div class="metric-label">Total Samples</div>
                <div class="metric-value">{metrics.get('total_samples', 0):,}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Deduplication Rate</div>
                <div class="metric-value">{metrics.get('dedup_rate', 0.0) * 100:.2f}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Total Duration</div>
                <div class="metric-value">{metrics.get('total_duration', 0.0):.2f}s</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">GPU Utilization</div>
                <div class="metric-value">{metrics.get('avg_gpu_util', 0.0) * 100:.1f}%</div>
            </div>
        </div>
        
        <div class="grafana-section">
            <h2>📊 Grafana Dashboard</h2>
            <p>For professional monitoring and visualization, use the Grafana dashboard:</p>
            <div class="instructions">
                <ol>
                    <li>Import the Grafana dashboard JSON: <code>{Path(grafana_json_path).name}</code></li>
                    <li>Ensure Prometheus datasource <code>{self.datasource_name}</code> is configured in Grafana</li>
                    <li>Access the dashboard at your Grafana instance URL</li>
                    <li>For real-time monitoring, ensure Prometheus metrics are being exported</li>
                </ol>
                <p><strong>Dashboard File:</strong> <code>{grafana_json_path}</code></p>
            </div>
        </div>
        
        <div style="margin-top: 30px; padding-top: 20px; border-top: 1px solid #ddd; color: #666; font-size: 12px;">
            <p>Generated by NVIDIA GR00T Data Pipeline</p>
            <p>For more information, visit: <a href="https://github.com/nvidia/groot">GR00T Project</a></p>
        </div>
    </div>
</body>
</html>
"""
            
            with open(html_path, "w") as f:
                f.write(html_content)
            
            logger.info(f"HTML summary page created at {html_path}")
            
        except Exception as e:
            logger.debug(f"Could not create HTML summary: {e}")

    def _create_plotly_summary(
        self,
        metrics: Dict[str, Any],
        output_path: Optional[str],
    ) -> None:
        """Create Plotly-based summary (for web dashboards)."""
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            fig = make_subplots(
                rows=2,
                cols=2,
                subplot_titles=("Stage Durations", "Throughput", "Overall Metrics", "Errors"),
                specs=[[{"type": "bar"}, {"type": "bar"}], [{"type": "table"}, {"type": "bar"}]],
            )

            # Stage durations
            if "stages" in metrics:
                stage_names = [s["name"] for s in metrics["stages"]]
                stage_durations = [s["duration"] for s in metrics["stages"]]

                fig.add_trace(
                    go.Bar(x=stage_durations, y=stage_names, orientation="h", name="Duration"),
                    row=1,
                    col=1,
                )

                # Throughput
                stage_throughput = [s["throughput"] for s in metrics["stages"]]
                fig.add_trace(
                    go.Bar(x=stage_names, y=stage_throughput, name="Throughput"), row=1, col=2
                )

            # Overall metrics table
            overall_metrics = {
                "Total Samples": metrics.get("total_samples", 0),
                "Dedup Rate": f"{metrics.get('dedup_rate', 0) * 100:.1f}%",
                "GPU Utilization": f"{metrics.get('avg_gpu_util', 0) * 100:.1f}%",
                "Total Duration": f"{metrics.get('total_duration', 0):.1f}s",
            }

            fig.add_trace(
                go.Table(
                    header={"values": ["Metric", "Value"]},
                    cells={
                        "values": [list(overall_metrics.keys()), list(overall_metrics.values())]
                    },
                ),
                row=2,
                col=1,
            )

            # Errors
            if "num_errors" in metrics:
                fig.add_trace(
                    go.Bar(x=["Errors"], y=[metrics["num_errors"]], name="Errors"), row=2, col=2
                )

            fig.update_layout(
                height=800, showlegend=False, title_text="Pipeline Performance Summary"
            )

            # Determine output path for Plotly HTML
            if output_path:
                # Ensure output path ends with .html
                if not output_path.endswith('.html'):
                    if output_path.endswith('.png') or output_path.endswith('.json'):
                        output_path = output_path.rsplit('.', 1)[0] + '.html'
                    else:
                        output_path = str(Path(output_path).parent / "pipeline_summary_plotly.html")
                
                # Ensure directory exists
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                fig.write_html(output_path)
                logger.info(f"Saved Plotly interactive dashboard to {output_path}")
                logger.info(f"Open {output_path} in a web browser to view the dashboard")
            else:
                fig.show()
        except ImportError:
            logger.warning("Plotly not available, falling back to basic text summary")
            self._create_basic_summary(metrics, output_path)
        except Exception as e:
            logger.error(f"Error creating Plotly dashboard: {e}", exc_info=True)
            logger.info("Falling back to basic text summary")
            self._create_basic_summary(metrics, output_path)

    def _create_basic_summary(
        self,
        metrics: Dict[str, Any],
        output_path: Optional[str],
    ) -> None:
        """Create basic text summary."""
        summary = "\n" + "=" * 60 + "\n"
        summary += "Pipeline Performance Summary\n"
        summary += "=" * 60 + "\n\n"

        summary += f"Total Samples: {metrics.get('total_samples', 0):,}\n"
        summary += f"Total Duration: {metrics.get('total_duration', 0):.2f}s\n"
        summary += f"Deduplication Rate: {metrics.get('dedup_rate', 0) * 100:.2f}%\n"
        summary += f"GPU Utilization: {metrics.get('avg_gpu_util', 0) * 100:.2f}%\n"
        summary += f"Errors: {metrics.get('num_errors', 0)}\n\n"

        if "stages" in metrics:
            summary += "Stage Performance:\n"
            summary += "-" * 60 + "\n"
            for stage in metrics["stages"]:
                summary += f"{stage['name']:30s} {stage['duration']:8.2f}s "
                summary += f"({stage['throughput']:8.0f} items/s)\n"

        summary += "=" * 60 + "\n"

        logger.info(summary)

        if output_path:
            with open(output_path, "w") as f:
                f.write(summary)
            logger.info(f"Saved summary to {output_path}")
