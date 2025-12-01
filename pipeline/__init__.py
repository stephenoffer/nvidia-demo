"""Multimodal data curation pipeline for robotics foundation models.

A production-ready, GPU-accelerated data curation pipeline built on Ray
for processing multimodal datasets (video, text, sensor data) used in
robotics foundation model training.

Example:
    ```python
    from pipeline import MultimodalPipeline, PipelineConfig

    config = PipelineConfig(
        input_paths=["s3://bucket/videos/"],
        output_path="s3://bucket/curated/",
        num_gpus=4,
    )
    pipeline = MultimodalPipeline(config)
    results = pipeline.run()
    ```
"""

from pipeline.config import PipelineConfig
from pipeline.core import MultimodalPipeline

# Import version from __version__.py
try:
    from pipeline.__version__ import __version__
except ImportError:
    __version__ = "0.1.0"  # Fallback version

__all__ = [
    "MultimodalPipeline",
    "PipelineConfig",
    "__version__",
]
