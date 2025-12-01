"""Base synthetic data generator for multimodal datasets."""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import ray
from ray.data import Dataset

logger = logging.getLogger(__name__)


class SyntheticDataGenerator:
    """Base class for generating synthetic multimodal datasets.

    Supports generating synthetic data for testing, demos, and
    augmenting real datasets following GR00T's synthetic data strategy.
    """

    def __init__(
        self,
        num_samples: int = 1000,
        seed: Optional[int] = None,
    ):
        """Initialize synthetic data generator.

        Args:
            num_samples: Number of samples to generate
            seed: Random seed for reproducibility
        """
        self.num_samples = num_samples
        self.seed = seed

        if seed is not None:
            np.random.seed(seed)

    def generate(self) -> Dataset:
        """Generate synthetic dataset.

        Returns:
            Ray Dataset containing synthetic data
        """
        raise NotImplementedError("Subclasses must implement generate()")

    def generate_batch(self, batch_size: int) -> List[Dict[str, Any]]:
        """Generate a batch of synthetic samples.

        Args:
            batch_size: Number of samples in batch

        Returns:
            List of synthetic data samples
        """
        raise NotImplementedError("Subclasses must implement generate_batch()")


@ray.remote(num_cpus=1, memory=2 * 1024 * 1024 * 1024)  # 2GB memory limit
class SyntheticDataWorker:
    """Ray worker for parallel synthetic data generation."""

    def __init__(self, generator_class, generator_kwargs: Dict[str, Any]):
        """Initialize synthetic data worker.

        Args:
            generator_class: Generator class to use
            generator_kwargs: Arguments for generator initialization
        """
        self.generator = generator_class(**generator_kwargs)

    def generate_batch(self, batch_size: int) -> List[Dict[str, Any]]:
        """Generate a batch of synthetic data.

        Args:
            batch_size: Number of samples to generate

        Returns:
            List of synthetic samples
        """
        return self.generator.generate_batch(batch_size)
