"""Text processing stage for multimodal pipeline.

Handles text cleaning, filtering, and preprocessing.
Uses Ray Data map_batches for efficient distributed processing.
"""

import logging
import re
from typing import Any, Dict, List, Optional

from ray.data import Dataset

from pipeline.stages.base import ProcessorBase
from pipeline.utils.constants import _DEFAULT_BATCH_SIZE, _DEFAULT_CPUS, _DEFAULT_MIN_LENGTH, _DEFAULT_MAX_LENGTH
from pipeline.utils.data_types import get_data_type, DataType, extract_text

logger = logging.getLogger(__name__)


class TextProcessorActor:
    """Text processor for batch processing.

    Note: Not a Ray actor - used directly in map_batches for efficiency.
    """

    def __init__(
        self,
        min_length: int,
        max_length: int,
        remove_boilerplate: bool,
        language: str,
    ):
        """Initialize text processor.

        Args:
            min_length: Minimum text length
            max_length: Maximum text length
            remove_boilerplate: Whether to remove boilerplate
            language: Language code
        """
        self.min_length = min_length
        self.max_length = max_length
        self.remove_boilerplate = remove_boilerplate
        self.language = language

        # Common boilerplate patterns
        self.boilerplate_patterns = [
            r"cookie policy",
            r"privacy policy",
            r"terms of service",
            r"all rights reserved",
            r"copyright",
            r"subscribe",
            r"newsletter",
        ]

    def process_text(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single text item.

        Args:
            item: Text data item

        Returns:
            Processed text item
        """
        try:
            text = extract_text(item)
            if not text and isinstance(item, str):
                text = item
            if not text:
                return {**item, "processed": False, "error": "No text field"}

            text = self._clean_text(text)

            if len(text) < self.min_length or len(text) > self.max_length:
                return {**item, "processed": False, "reason": "length_filter"}

            if self.remove_boilerplate and self._has_boilerplate(text):
                return {**item, "processed": False, "reason": "boilerplate"}

            return {
                **item,
                "text": text,
                "processed": True,
                "text_length": len(text),
            }

        except (KeyError, TypeError, AttributeError, ValueError) as e:
            logger.error(f"Error processing text: {e}")
            return {**item, "processed": False, "error": str(e)}

    def _clean_text(self, text: str) -> str:
        """Clean text content."""
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^\w\s.,!?;:-]", "", text)
        return text.strip()

    def _has_boilerplate(self, text: str) -> bool:
        """Check if text contains boilerplate.

        Args:
            text: Text to check

        Returns:
            True if boilerplate detected
        """
        text_lower = text.lower()
        return any(pattern in text_lower for pattern in self.boilerplate_patterns)


class TextProcessor(ProcessorBase):
    """Text processing stage for the pipeline.

    Handles tokenization, cleaning, and quality filtering.
    """

    def __init__(
        self,
        min_length: int = _DEFAULT_MIN_LENGTH,
        max_length: int = _DEFAULT_MAX_LENGTH,
        remove_boilerplate: bool = True,
        language: str = "en",
        batch_size: int = _DEFAULT_BATCH_SIZE,
    ):
        """Initialize text processor.

        Args:
            min_length: Minimum text length in characters
            max_length: Maximum text length in characters
            remove_boilerplate: Whether to remove boilerplate text
            language: Language code for processing
            batch_size: Batch size for processing
        """
        super().__init__(batch_size=batch_size)
        self.min_length = min_length
        self.max_length = max_length
        self.remove_boilerplate = remove_boilerplate
        self.language = language

    def process(self, dataset: Dataset) -> Dataset:
        """Process text data in the dataset.

        Uses Ray Data map_batches for efficient batch processing.

        Args:
            dataset: Input Ray Dataset

        Returns:
            Processed Ray Dataset
        """
        logger.info("Processing text data")

        # Use named function instead of lambda for better serialization
        def is_text_type(item: dict[str, Any]) -> bool:
            """Check if item is text type."""
            return get_data_type(item) == DataType.TEXT
        
        text_dataset = dataset.filter(is_text_type)

        def process_batch(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            """Process a batch of text items."""
            processor = TextProcessorActor(
                self.min_length,
                self.max_length,
                self.remove_boilerplate,
                self.language,
            )

            return [processor.process_text(item) for item in batch]

        processed = text_dataset.map_batches(
            process_batch,
            batch_size=self.batch_size,
            batch_format="pandas",  # Specify batch format
            num_cpus=_DEFAULT_CPUS,
        )

        # Use named function instead of lambda
        def is_processed(item: dict[str, Any]) -> bool:
            """Check if item was processed successfully."""
            return item.get("processed", False)
        
        return processed.filter(is_processed)

    def _process_item(self, item: dict[str, Any]) -> Optional[dict[str, Any]]:
        """Process item - not used for text processing.

        Text processing uses TextProcessorActor for batch processing.
        """
        return None
