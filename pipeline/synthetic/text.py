"""Synthetic text data generator for multimodal datasets.

Generates synthetic text data that simulates robot instructions, task
descriptions, or natural language annotations.

Uses Ray Data for distributed dataset creation.
See: https://docs.ray.io/en/latest/data/data.html
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np  # https://numpy.org/
import ray
from ray.data import Dataset

logger = logging.getLogger(__name__)


class SyntheticTextGenerator:
    """Generate synthetic text data.

    Creates synthetic text that simulates robot instructions, task
    descriptions, or annotations for multimodal datasets.
    """

    def __init__(
        self,
        text_type: str = "instruction",
        min_length: int = 10,
        max_length: int = 200,
        seed: Optional[int] = None,
    ):
        """Initialize synthetic text generator.

        Args:
            text_type: Type of text ('instruction', 'description', 'annotation')
            min_length: Minimum text length in characters
            max_length: Maximum text length in characters
            seed: Random seed for reproducibility
        """
        self.text_type = text_type
        self.min_length = min_length
        self.max_length = max_length

        if seed is not None:
            np.random.seed(seed)

        # Templates for different text types
        self._init_templates()

    def _init_templates(self) -> None:
        """Initialize text templates."""
        self.instruction_templates = [
            "Pick up the {object} and place it on the {location}.",
            "Move the {object} to the {location}.",
            "Grasp the {object} with your {hand} hand.",
            "Place the {object} carefully on the {location}.",
            "Pick up the {object} from the {source}.",
        ]

        self.description_templates = [
            "A robot is {action} the {object}.",
            "The robot {action} the {object} on the {location}.",
            "Demonstration of {task} with {object}.",
            "Robot performing {task} task.",
        ]

        self.annotation_templates = [
            "Task: {task}, Object: {object}, Location: {location}",
            "Action: {action}, Target: {object}",
            "Robot {action} {object}",
        ]

        # Vocabulary
        self.objects = [
            "apple",
            "banana",
            "cup",
            "bottle",
            "box",
            "book",
            "pen",
            "plate",
            "bowl",
            "spoon",
            "fork",
            "knife",
        ]

        self.locations = [
            "table",
            "shelf",
            "counter",
            "drawer",
            "cabinet",
            "floor",
            "rack",
            "bin",
            "tray",
            "container",
        ]

        self.actions = [
            "picking up",
            "placing",
            "moving",
            "grasping",
            "manipulating",
            "handling",
            "transferring",
        ]

        self.tasks = [
            "pick and place",
            "object manipulation",
            "grasping",
            "sorting",
            "organizing",
            "cleaning",
        ]

    def generate_text(self, text_id: int) -> Dict[str, Any]:
        """Generate a single synthetic text sample.

        Args:
            text_id: Unique identifier for text

        Returns:
            Dictionary containing text data
        """
        if self.text_type == "instruction":
            text = self._generate_instruction()
        elif self.text_type == "description":
            text = self._generate_description()
        else:  # annotation
            text = self._generate_annotation()

        # Ensure length constraints
        if len(text) < self.min_length:
            text = text + " " + self._generate_filler(self.min_length - len(text))
        if len(text) > self.max_length:
            text = text[: self.max_length]

        return {
            "text_id": text_id,
            "text": text,
            "text_type": self.text_type,
            "length": len(text),
            "data_type": "text",
            "format": "synthetic",
            "source": "synthetic_generator",
        }

    def _generate_instruction(self) -> str:
        """Generate instruction text."""
        template = np.random.choice(self.instruction_templates)

        object_name = np.random.choice(self.objects)
        location = np.random.choice(self.locations)
        hand = np.random.choice(["left", "right"])
        source = np.random.choice(self.locations)

        text = template.format(
            object=object_name,
            location=location,
            hand=hand,
            source=source,
        )

        return text.capitalize()

    def _generate_description(self) -> str:
        """Generate description text."""
        template = np.random.choice(self.description_templates)

        action = np.random.choice(self.actions)
        object_name = np.random.choice(self.objects)
        location = np.random.choice(self.locations)
        task = np.random.choice(self.tasks)

        text = template.format(
            action=action,
            object=object_name,
            location=location,
            task=task,
        )

        return text.capitalize()

    def _generate_annotation(self) -> str:
        """Generate annotation text."""
        template = np.random.choice(self.annotation_templates)

        task = np.random.choice(self.tasks)
        object_name = np.random.choice(self.objects)
        location = np.random.choice(self.locations)
        action = np.random.choice(self.actions)

        text = template.format(
            task=task,
            object=object_name,
            location=location,
            action=action,
        )

        return text

    def _generate_filler(self, length: int) -> str:
        """Generate filler text to reach minimum length.

        Args:
            length: Desired additional length

        Returns:
            Filler text
        """
        filler_words = [
            "carefully",
            "slowly",
            "precisely",
            "gently",
            "safely",
            "efficiently",
            "accurately",
        ]

        words = []
        current_length = 0

        while current_length < length:
            word = np.random.choice(filler_words)
            words.append(word)
            current_length += len(word) + 1

        return " ".join(words)

    def generate_batch(self, batch_size: int, start_id: int = 0) -> List[Dict[str, Any]]:
        """Generate a batch of texts.

        Args:
            batch_size: Number of texts to generate
            start_id: Starting text ID

        Returns:
            List of text dictionaries
        """
        texts = []
        for i in range(batch_size):
            text = self.generate_text(start_id + i)
            texts.append(text)

        return texts


class SyntheticTextDataset:
    """Generate synthetic text dataset using Ray Data."""

    def __init__(
        self,
        num_texts: int = 1000,
        text_type: str = "instruction",
        min_length: int = 10,
        max_length: int = 200,
        num_workers: int = 4,
        seed: Optional[int] = None,
    ):
        """Initialize synthetic text dataset generator.

        Args:
            num_texts: Total number of texts to generate
            text_type: Type of text
            min_length: Minimum text length
            max_length: Maximum text length
            num_workers: Number of parallel workers
            seed: Random seed
        """
        self.num_texts = num_texts
        self.text_type = text_type
        self.min_length = min_length
        self.max_length = max_length
        self.num_workers = num_workers
        self.seed = seed

    def generate(self) -> Dataset:
        """Generate synthetic text dataset.

        Returns:
            Ray Dataset containing synthetic texts
        """
        logger.info(f"Generating {self.num_texts} synthetic texts")

        # Create generator
        generator = SyntheticTextGenerator(
            text_type=self.text_type,
            min_length=self.min_length,
            max_length=self.max_length,
            seed=self.seed,
        )

        # Generate texts in batches
        batch_size = max(1, self.num_texts // self.num_workers)
        batches = []

        for i in range(0, self.num_texts, batch_size):
            current_batch_size = min(batch_size, self.num_texts - i)
            batch = generator.generate_batch(current_batch_size, start_id=i)
            batches.extend(batch)

        # Convert to Ray Dataset using Ray Data's optimized from_items
        # Use parallel generation with Ray Data for better performance
        dataset = ray.data.from_items(batches, parallelism=self.num_workers)
        
        # Log generation info without materializing dataset
        # Use len(batches) instead of dataset.count() to avoid materialization
        logger.info(f"Generated {len(batches)} synthetic text batches")
        
        # Optional: Repartition for optimal block distribution
        if len(batches) > self.num_workers:
            dataset = dataset.repartition(num_blocks=self.num_workers)
            logger.info(f"Repartitioned dataset to {self.num_workers} blocks for optimal parallelism")

        return dataset
