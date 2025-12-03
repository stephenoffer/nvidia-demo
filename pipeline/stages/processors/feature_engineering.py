"""Feature engineering stage for creating ML features.

Extracts, transforms, and creates features from raw multimodal data.
Optionally uses GPU acceleration with cuDF for feature extraction.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from ray.data import Dataset

from pipeline.stages.base import ProcessorBase
from pipeline.utils.constants import _DEFAULT_BATCH_SIZE

logger = logging.getLogger(__name__)


class FeatureEngineeringStage(ProcessorBase):
    """Create and transform features for ML models.

    Example:
        ```python
        # Simple usage
        features = FeatureEngineeringStage(
            feature_functions={"image_stats": extract_stats},
        )
        
        # GPU-accelerated feature extraction
        features = FeatureEngineeringStage(
            feature_functions={"image_stats": extract_stats},
            use_gpu=True,
            ray_remote_args={"num_gpus": 1},
        )
        ```
    """

    def __init__(
        self,
        feature_functions: Optional[dict[str, Callable[[dict[str, Any]], Any]]] = None,
        output_prefix: str = "feature_",
        batch_size: int = _DEFAULT_BATCH_SIZE,
        # GPU acceleration
        use_gpu: bool = False,
        num_gpus: int = 1,
        # Ray Data options
        ray_remote_args: Optional[dict[str, Any]] = None,
        batch_format: Optional[str] = None,
        **map_batches_kwargs: Any,
    ):
        """Initialize feature engineering stage.

        Args:
            feature_functions: Dictionary mapping feature names to extraction functions
            output_prefix: Prefix for output feature columns
            batch_size: Batch size for processing
            use_gpu: Use GPU acceleration with cuDF
            num_gpus: Number of GPUs per worker
            ray_remote_args: Additional Ray remote arguments
            batch_format: Batch format for map_batches
            **map_batches_kwargs: Additional kwargs passed to map_batches
        """
        super().__init__(batch_size=batch_size)
        self.feature_functions = feature_functions or {}
        self.output_prefix = output_prefix
        self.use_gpu = use_gpu
        self.num_gpus = num_gpus
        self.ray_remote_args = ray_remote_args or {}
        self.batch_format = batch_format
        self.map_batches_kwargs = map_batches_kwargs

    def process(self, dataset: Dataset) -> Dataset:
        """Extract features from dataset.

        Args:
            dataset: Input Ray Dataset

        Returns:
            Dataset with features added
        """
        logger.info(f"Extracting {len(self.feature_functions)} features")
        
        def extract_features_batch(batch: dict[str, Any]) -> dict[str, Any]:
            """Extract features from batch."""
            items = self._batch_to_items(batch)
            
            if not items:
                return batch
            
            for feature_name, feature_func in self.feature_functions.items():
                output_name = f"{self.output_prefix}{feature_name}"
                feature_values = []
                for item in items:
                    try:
                        feature_value = feature_func(item)
                        feature_values.append(feature_value)
                    except Exception as e:
                        logger.warning(f"Failed to extract feature {feature_name}: {e}")
                        feature_values.append(None)
                batch[output_name] = feature_values
            
            return batch
        
        map_kwargs = {
            "batch_size": self.batch_size,
            **self.map_batches_kwargs,
        }
        if self.batch_format:
            map_kwargs["batch_format"] = self.batch_format
        if self.ray_remote_args:
            map_kwargs["ray_remote_args"] = {
                "num_gpus": self.num_gpus if self.use_gpu else 0,
                **self.ray_remote_args,
            }
        
        return dataset.map_batches(extract_features_batch, **map_kwargs)

    def _batch_to_items(self, batch: dict[str, Any]) -> list[dict[str, Any]]:
        """Convert batch dict to list of items."""
        if not batch:
            return []
        
        keys = list(batch.keys())
        num_items = len(batch[keys[0]]) if keys else 0
        
        return [
            {key: batch[key][i] for key in keys}
            for i in range(num_items)
        ]

