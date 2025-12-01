"""Pipeline visualization management.

Handles visualization generation, dashboard creation, and video generation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import ray
from ray.data import Dataset

from pipeline.visualization.dashboard import VisualizationDashboard
from pipeline.visualization.video_generator import VideoGenerator

logger = logging.getLogger(__name__)


class PipelineVisualizationManager:
    """Manages pipeline visualization features."""

    def __init__(self) -> None:
        """Initialize visualization manager."""
        self.video_generator: Optional[VideoGenerator] = None
        self.dashboard: Optional[VisualizationDashboard] = None

    def enable_visualization(
        self,
        video_resolution: tuple = (1280, 720),
        video_fps: int = 30,
        dashboard_mode: str = "grafana",
        datasource_name: str = "Prometheus",
    ) -> None:
        """Enable visualization features.

        Args:
            video_resolution: Resolution for generated videos
            video_fps: Frames per second for videos
            dashboard_mode: Dashboard mode ('grafana' for production, 'plotly' for interactive web, 'local' for plotly)
            datasource_name: Name of Prometheus datasource in Grafana
        """
        self.video_generator = VideoGenerator(resolution=video_resolution, fps=video_fps)
        self.dashboard = VisualizationDashboard(mode=dashboard_mode, datasource_name=datasource_name)
        logger.info(f"Visualization features enabled with {dashboard_mode} dashboard")

    def generate_visualizations(
        self,
        results: dict[str, Any],
        dataset: Dataset,
        output_path: str,
    ) -> None:
        """Generate visualizations for pipeline results.

        Args:
            results: Pipeline results dictionary
            dataset: Final processed dataset
            output_path: Output path for visualizations
        """
        if not self.dashboard:
            # Default to Grafana for production-grade visualization
            self.dashboard = VisualizationDashboard(mode="grafana")

        # For Grafana, this will create a JSON config file
        # For matplotlib/plotly, this creates an image file
        summary_path = Path(output_path) / "pipeline_summary"
        self.dashboard.create_pipeline_summary(results, str(summary_path))

        if self.video_generator:
            self._generate_videos(dataset, output_path)

    def _generate_videos(self, dataset: Dataset, output_path: str) -> None:
        """Generate sample visualization videos.

        Args:
            dataset: Dataset to visualize
            output_path: Output directory path
        """
        if not self.video_generator:
            return

        video_output_dir = Path(output_path) / "videos"
        logger.info("Generating sample visualization videos")

        try:
            from pipeline.utils.constants import _VISUALIZATION_SAMPLE_SIZE, _MAX_VIDEOS

            sample_items = []
            for batch in dataset.iter_batches(batch_size=_VISUALIZATION_SAMPLE_SIZE, prefetch_batches=0):
                sample_items.extend(batch[: min(_VISUALIZATION_SAMPLE_SIZE, len(batch))])
                if len(sample_items) >= _VISUALIZATION_SAMPLE_SIZE:
                    break

            if sample_items:
                sample_dataset = ray.data.from_items(sample_items)
                self.video_generator.generate_from_dataset(
                    sample_dataset,
                    str(video_output_dir),
                    max_videos=_MAX_VIDEOS,
                )
            else:
                logger.warning("No data available for video generation")
        except (OSError, IOError, ValueError, AttributeError) as e:
            logger.error(f"Error generating visualization videos: {e}", exc_info=True)

