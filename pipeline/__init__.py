"""Multimodal data curation pipeline for robotics foundation models.

A production-ready, GPU-accelerated data curation pipeline built on Ray
for processing multimodal datasets (video, text, sensor data) used in
robotics foundation model training.

Quick Start Examples:
    ```python
    # Simple pipeline
    from pipeline import MultimodalPipeline
    
    pipeline = MultimodalPipeline.create(
        input_paths=["s3://bucket/videos/"],
        output_path="s3://bucket/curated/",
        num_gpus=4,
    )
    results = pipeline.run()
    
    # Declarative API
    from pipeline.api import Pipeline
    
    pipeline = Pipeline.quick_start(
        input_path="s3://bucket/videos/",
        output_path="s3://bucket/curated/",
        num_gpus=4,
    )
    results = pipeline.run()
    
    # With context manager for automatic cleanup
    with MultimodalPipeline.create(...) as pipeline:
        results = pipeline.run()
    ```
"""

from pipeline.config import PipelineConfig
from pipeline.core import MultimodalPipeline

# Import declarative API for convenience
try:
    from pipeline.api.declarative import Pipeline, load_from_yaml
    from pipeline.api.multipipeline import MultiPipelineRunner
    _DECLARATIVE_AVAILABLE = True
except ImportError:
    _DECLARATIVE_AVAILABLE = False

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

# Conditionally add declarative API exports
if _DECLARATIVE_AVAILABLE:
    __all__.extend(["Pipeline", "load_from_yaml", "MultiPipelineRunner"])
