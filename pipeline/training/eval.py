"""Evaluation pipeline integration for model evaluation.

Provides integration with evaluation pipelines to ensure curated
data can be used for model evaluation and benchmarking.
"""

import logging
from typing import Any, Dict, List

from ray.data import Dataset

from pipeline.utils.constants import _EVAL_BATCH_SIZE

logger = logging.getLogger(__name__)


class EvaluationPipelineIntegration:
    """Integration with evaluation pipelines.

    Prepares curated data for model evaluation workflows, including
    test set creation and evaluation data formatting.
    """

    def __init__(self, test_split: float = 0.1):
        """Initialize evaluation pipeline integration.

        Args:
            test_split: Fraction of data for test set
        """
        self.test_split = test_split

    def create_test_set(
        self,
        dataset: Dataset,
        output_path: str,
    ) -> Dict[str, Any]:
        """Create test set from curated dataset.

        Args:
            dataset: Curated Ray Dataset
            output_path: Output path for test set

        Returns:
            Dictionary with test set metadata
        """
        logger.info("Creating test set for evaluation")

        # For test split, we need to know dataset size
        # Use sampling-based estimation to avoid full materialization
        try:
            # Sample first batch to estimate
            sample_batch = next(dataset.iter_batches(batch_size=1000, prefetch_batches=0), None)
            if sample_batch is not None:
                sample_size = len(sample_batch)
                # Estimate total size from sample (rough estimate)
                # This is an approximation but avoids full materialization
                estimated_total = sample_size * max(10, dataset.num_blocks() if hasattr(dataset, 'num_blocks') else 10)
                test_size = int(estimated_total * self.test_split)
                logger.info(f"Estimated dataset size: {estimated_total}, test size: {test_size}")
            else:
                # Fallback: use fixed test size
                test_size = 1000
                logger.warning("Could not estimate dataset size, using default test size")
        except (StopIteration, AttributeError, Exception) as e:
            logger.warning(f"Could not estimate dataset size: {e}, using default test size")
            test_size = 1000  # Default test size
        
        # Use limit to create test set
        test_dataset = dataset.limit(test_size)

        # Write test set
        test_dataset.write_parquet(output_path)

        # Log test set info without materializing
        logger.info(f"Created test set with approximately {test_size} samples")

        return {
            "test_path": output_path,
            "num_test_samples": test_size,  # Use estimated size instead of count()
        }

    def prepare_eval_batches(
        self,
        dataset: Dataset,
        batch_size: int = _EVAL_BATCH_SIZE,
    ) -> List[Dict[str, Any]]:
        """Prepare evaluation batches.

        Args:
            dataset: Evaluation dataset
            batch_size: Size of evaluation batches

        Returns:
            List of evaluation batches
        """
        logger.info("Preparing evaluation batches")

        batches = []
        for batch in dataset.iter_batches(batch_size=batch_size):
            batches.append(batch)

        logger.info(f"Prepared {len(batches)} evaluation batches")
        return batches
