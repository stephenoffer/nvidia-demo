"""Integration modules for external systems and tools.

Includes integrations with:
- MLflow: Experiment tracking and model versioning
- Weights & Biases: Experiment tracking and visualization
- OpenLineage: Data lineage tracking
- NVIDIA tools: Isaac Lab, Omniverse, Cosmos Dreams
- Other tools: LanceDB, SQL databases, NeMo
"""

from pipeline.integrations.cosmos import CosmosDreamsLoader
from pipeline.integrations.isaac_lab import IsaacLabLoader
from pipeline.integrations.lancedb import LanceDBStorage
from pipeline.integrations.omniverse import OmniverseLoader, OmniverseReplicatorGenerator
from pipeline.integrations.sql import SQLDataLoader

__all__ = [
    "IsaacLabLoader",
    "CosmosDreamsLoader",
    "LanceDBStorage",
    "OmniverseLoader",
    "OmniverseReplicatorGenerator",
    "SQLDataLoader",
]

# NVIDIA NeMo integration (optional)
try:
    from pipeline.integrations.nemo import NeMoEmbeddingGenerator, NeMoTextProcessor
    __all__.extend(["NeMoEmbeddingGenerator", "NeMoTextProcessor"])
except ImportError:
    pass

# MLOps integrations (optional)
try:
    from pipeline.integrations.mlflow import MLflowTracker, create_mlflow_tracker
    __all__.extend(["MLflowTracker", "create_mlflow_tracker"])
except ImportError:
    pass

try:
    from pipeline.integrations.wandb import WandBTracker, create_wandb_tracker
    __all__.extend(["WandBTracker", "create_wandb_tracker"])
except ImportError:
    pass

try:
    from pipeline.integrations.openlineage import OpenLineageTracker, create_openlineage_tracker
    __all__.extend(["OpenLineageTracker", "create_openlineage_tracker"])
except ImportError:
    pass

try:
    from pipeline.integrations.model_registry import ModelRegistry, create_model_registry
    __all__.extend(["ModelRegistry", "create_model_registry"])
except ImportError:
    pass

try:
    from pipeline.integrations.feature_store import FeatureStore, create_feature_store
    __all__.extend(["FeatureStore", "create_feature_store"])
except ImportError:
    pass
