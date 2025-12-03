"""Feature store integration for feature management and serving.

Provides integration with feature stores for storing, versioning, and serving
features for both training and inference pipelines.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class FeatureStore:
    """Feature store for managing ML features."""

    def __init__(
        self,
        store_type: str = "feast",
        store_path: Optional[str] = None,
        registry_path: Optional[str] = None,
    ):
        """Initialize feature store.

        Args:
            store_type: Type of feature store ("feast", "tecton", "custom")
            store_path: Path to feature store
            registry_path: Path to feature registry
        """
        self.store_type = store_type
        self.store_path = store_path
        self.registry_path = registry_path

    def register_features(
        self,
        feature_name: str,
        feature_def: dict[str, Any],
        version: Optional[str] = None,
    ) -> str:
        """Register features in feature store.

        Args:
            feature_name: Name of feature group
            feature_def: Feature definition
            version: Feature version

        Returns:
            Feature store URI
        """
        if self.store_type == "feast":
            return self._register_feast_features(feature_name, feature_def, version)
        else:
            raise ValueError(f"Unsupported feature store type: {self.store_type}")

    def get_features(
        self,
        feature_names: list[str],
        entity_keys: dict[str, Any],
        timestamp: Optional[str] = None,
    ) -> dict[str, Any]:
        """Get features from feature store.

        Args:
            feature_names: List of feature names to retrieve
            entity_keys: Entity keys for feature lookup
            timestamp: Timestamp for point-in-time feature retrieval

        Returns:
            Dictionary of feature values
        """
        if self.store_type == "feast":
            return self._get_feast_features(feature_names, entity_keys, timestamp)
        else:
            raise ValueError(f"Unsupported feature store type: {self.store_type}")

    def _register_feast_features(
        self,
        feature_name: str,
        feature_def: dict[str, Any],
        version: Optional[str],
    ) -> str:
        """Register features in Feast feature store."""
        from feast import FeatureStore
        
        store = FeatureStore(repo_path=self.store_path or ".")
        
        from feast import Entity, Feature, FeatureView, ValueType
        from datetime import timedelta
        
        entity = Entity(name=feature_def.get("entity_name", "entity_id"), value_type=ValueType.STRING)
        
        feature_view = FeatureView(
            name=feature_name,
            entities=[entity],
            features=[
                Feature(name=f["name"], dtype=ValueType.from_str(f["dtype"]))
                for f in feature_def.get("features", [])
            ],
            ttl=timedelta(days=feature_def.get("ttl_days", 1)),
        )
        
        store.apply([entity, feature_view])
        return f"{self.store_type}://{feature_name}"

    def _get_feast_features(
        self,
        feature_names: list[str],
        entity_keys: dict[str, Any],
        timestamp: Optional[str],
    ) -> dict[str, Any]:
        """Get features from Feast feature store."""
        from feast import FeatureStore
        
        store = FeatureStore(repo_path=self.store_path or ".")
        
        entity_df = store.get_online_features(
            features=feature_names,
            entity_rows=[entity_keys],
        ).to_dict()
        
        return entity_df


def create_feature_store(
    store_type: str = "feast",
    store_path: Optional[str] = None,
    registry_path: Optional[str] = None,
) -> FeatureStore:
    """Create feature store instance.

    Args:
        store_type: Type of feature store ("feast", "tecton", "custom")
        store_path: Path to feature store
        registry_path: Path to feature registry

    Returns:
        FeatureStore instance
    """
    return FeatureStore(
        store_type=store_type,
        store_path=store_path,
        registry_path=registry_path,
    )

