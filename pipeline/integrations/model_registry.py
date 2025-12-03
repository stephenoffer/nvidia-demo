"""Model registry integration for model versioning and management.

Provides integration with MLflow Model Registry and other model registries
for model versioning, staging, and deployment tracking.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Model registry for versioning and managing models."""

    def __init__(
        self,
        registry_type: str = "mlflow",
        tracking_uri: Optional[str] = None,
        registry_name: Optional[str] = None,
    ):
        """Initialize model registry.

        Args:
            registry_type: Type of registry ("mlflow", "wandb", "custom")
            tracking_uri: Tracking URI for MLflow
            registry_name: Name of model registry
        """
        self.registry_type = registry_type
        self.tracking_uri = tracking_uri
        self.registry_name = registry_name

    def register_model(
        self,
        model_uri: str,
        model_name: str,
        version: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """Register model in registry.

        Args:
            model_uri: URI of model to register
            model_name: Name of model
            version: Model version (None = auto-increment)
            metadata: Additional metadata

        Returns:
            Registered model URI
        """
        if self.registry_type == "mlflow":
            return self._register_mlflow(model_uri, model_name, version, metadata)
        else:
            raise ValueError(f"Unsupported registry type: {self.registry_type}")

    def get_model(
        self,
        model_name: str,
        version: Optional[str] = None,
        stage: Optional[str] = None,
    ) -> str:
        """Get model URI from registry.

        Args:
            model_name: Name of model
            version: Model version (None = latest)
            stage: Model stage ("Staging", "Production", "Archived")

        Returns:
            Model URI
        """
        if self.registry_type == "mlflow":
            return self._get_mlflow_model(model_name, version, stage)
        else:
            raise ValueError(f"Unsupported registry type: {self.registry_type}")

    def transition_model_stage(
        self,
        model_name: str,
        version: str,
        stage: str,
    ) -> None:
        """Transition model to new stage.

        Args:
            model_name: Name of model
            version: Model version
            stage: Target stage ("Staging", "Production", "Archived")
        """
        if self.registry_type == "mlflow":
            self._transition_mlflow_stage(model_name, version, stage)
        else:
            raise ValueError(f"Unsupported registry type: {self.registry_type}")

    def _register_mlflow(
        self,
        model_uri: str,
        model_name: str,
        version: Optional[str],
        metadata: Optional[dict[str, Any]],
    ) -> str:
        """Register model in MLflow Model Registry."""
        import mlflow
        
        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)
        
        registered_model = mlflow.register_model(model_uri, model_name)
        
        if metadata:
            mlflow.set_model_version_tag(
                model_name,
                registered_model.version,
                "metadata",
                str(metadata),
            )
        
        return f"models:/{model_name}/{registered_model.version}"

    def _get_mlflow_model(
        self,
        model_name: str,
        version: Optional[str],
        stage: Optional[str],
    ) -> str:
        """Get model from MLflow Model Registry."""
        import mlflow
        
        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)
        
        if stage:
            return f"models:/{model_name}/{stage}"
        elif version:
            return f"models:/{model_name}/{version}"
        else:
            client = mlflow.tracking.MlflowClient()
            latest = client.get_latest_versions(model_name, stages=["None"])[0]
            return f"models:/{model_name}/{latest.version}"

    def _transition_mlflow_stage(
        self,
        model_name: str,
        version: str,
        stage: str,
    ) -> None:
        """Transition model stage in MLflow."""
        import mlflow
        
        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)
        
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(model_name, version, stage)


def create_model_registry(
    registry_type: str = "mlflow",
    tracking_uri: Optional[str] = None,
    registry_name: Optional[str] = None,
) -> ModelRegistry:
    """Create model registry instance.

    Args:
        registry_type: Type of registry ("mlflow", "wandb", "custom")
        tracking_uri: Tracking URI for MLflow
        registry_name: Name of model registry

    Returns:
        ModelRegistry instance
    """
    return ModelRegistry(
        registry_type=registry_type,
        tracking_uri=tracking_uri,
        registry_name=registry_name,
    )

