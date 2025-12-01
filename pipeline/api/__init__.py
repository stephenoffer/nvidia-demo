"""High-level declarative API for multimodal data pipeline.

Provides both Python and YAML-based configuration interfaces
for easy pipeline setup and execution.
"""

from pipeline.api.declarative import Pipeline, load_from_yaml
from pipeline.api.multipipeline import MultiPipelineRunner

__all__ = ["Pipeline", "MultiPipelineRunner", "load_from_yaml"]

