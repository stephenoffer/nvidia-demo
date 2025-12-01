"""Core pipeline orchestration module.

Provides the main MultimodalPipeline orchestrator and related execution components.
"""

from pipeline.core.orchestrator import MultimodalPipeline
from pipeline.core.registry import (
    ComponentRegistry,
    stage_registry,
    datasource_registry,
    loader_registry,
    validator_registry,
)
from pipeline.core.factory import (
    create_stage,
    create_validator,
    create_datasource,
    create_loader,
)

__all__ = [
    "MultimodalPipeline",
    "ComponentRegistry",
    "stage_registry",
    "datasource_registry",
    "loader_registry",
    "validator_registry",
    "create_stage",
    "create_validator",
    "create_datasource",
    "create_loader",
]

