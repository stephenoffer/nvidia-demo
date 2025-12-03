"""Instruction grounding stage for pairing text instructions with robot demonstrations.

Pairs text instructions (natural language or structured commands) with robot
trajectories for instruction-following foundation model training.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Dict


from ray.data import Dataset

from pipeline.stages.base import ProcessorBase
from pipeline.utils.constants import _DEFAULT_BATCH_SIZE

logger = logging.getLogger(__name__)


class InstructionGroundingStage(ProcessorBase):
    """Ground text instructions with robot demonstrations.

    Pairs instructions with trajectories, validates alignment, and scores
    instruction quality.
    """

    def __init__(
        self,
        instruction_field: str = "instruction",
        trajectory_field: str = "trajectory",
        instruction_format: str = "auto",
        validate_alignment: bool = True,
        min_instruction_length: int = 5,
        batch_size: int = _DEFAULT_BATCH_SIZE,
    ):
        """Initialize instruction grounding stage.

        Args:
            instruction_field: Field name containing instructions
            trajectory_field: Field name containing trajectories
            instruction_format: Format of instructions ("auto", "natural_language", "structured")
            validate_alignment: Whether to validate instruction-trajectory alignment
            min_instruction_length: Minimum instruction length in characters
            batch_size: Batch size for processing
        """
        super().__init__(batch_size=batch_size)
        self.instruction_field = instruction_field
        self.trajectory_field = trajectory_field
        self.instruction_format = instruction_format
        self.validate_alignment = validate_alignment
        self.min_instruction_length = min_instruction_length

    def process(self, dataset: Dataset) -> Dataset:
        """Ground instructions with trajectories.

        Args:
            dataset: Input Ray Dataset with instructions and trajectories

        Returns:
            Dataset with grounded instruction-trajectory pairs
        """
        logger.info("Grounding instructions with trajectories")
        return super().process(dataset)

    def _process_item(self, item: dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process item by grounding instruction.

        Args:
            item: Data item

        Returns:
            Grounded item
        """
        return self._ground_item(item)

    def _ground_item(self, item: dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Ground a single instruction-trajectory pair.

        Args:
            item: Data item with instruction and trajectory

        Returns:
            Grounded item or None if invalid
        """
        # Extract instruction
        instruction = self._extract_instruction(item)
        if not instruction:
            return None

        # Validate instruction
        if len(instruction) < self.min_instruction_length:
            logger.debug(f"Instruction too short: {len(instruction)} < {self.min_instruction_length}")
            return None

        # Extract trajectory (can be embedded in item or referenced)
        trajectory = self._extract_trajectory(item)

        # Create grounded pair
        grounded = dict(item)
        grounded["instruction"] = instruction
        grounded["instruction_length"] = len(instruction)
        grounded["instruction_format"] = self._detect_instruction_format(instruction)

        if trajectory:
            grounded["trajectory"] = trajectory
            grounded["has_trajectory"] = True

            # Validate alignment if requested
            if self.validate_alignment:
                alignment_score = self._validate_alignment(instruction, trajectory)
                grounded["instruction_alignment_score"] = alignment_score
                grounded["instruction_aligned"] = alignment_score > 0.5
        else:
            grounded["has_trajectory"] = False
            grounded["instruction_alignment_score"] = 0.0
            grounded["instruction_aligned"] = False

        # Score instruction quality
        quality_score = self._score_instruction_quality(instruction)
        grounded["instruction_quality_score"] = quality_score

        return grounded

    def _extract_instruction(self, item: dict[str, Any]) -> Optional[str]:
        """Extract instruction from item.

        Args:
            item: Data item

        Returns:
            Instruction text or None
        """
        from pipeline.utils.field_extraction import extract_field
        from pipeline.utils.data.data_types import extract_text

        instruction = extract_field(
            item,
            self.instruction_field,
            nested_paths=[["metadata"]],
        )
        if isinstance(instruction, str):
            return instruction

        for field in ["instruction", "text", "command", "task_description", "goal"]:
            value = extract_field(item, field, nested_paths=[["metadata"]])
            if isinstance(value, str):
                return value

        return extract_text(item)

    def _extract_trajectory(self, item: dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract trajectory from item.

        Args:
            item: Data item

        Returns:
            Trajectory dictionary or None
        """
        from pipeline.utils.field_extraction import extract_field
        from pipeline.utils.data.data_types import extract_sensor_data

        trajectory = extract_field(item, self.trajectory_field)
        if isinstance(trajectory, dict):
            return trajectory

        if "observations" in item or "actions" in item:
            return item

        sensor_data = extract_sensor_data(item)
        if sensor_data:
            return sensor_data

        return None

    def _detect_instruction_format(self, instruction: str) -> str:
        """Detect instruction format.

        Args:
            instruction: Instruction text

        Returns:
            Format name ("natural_language" or "structured")
        """
        if self.instruction_format != "auto":
            return self.instruction_format

        # Simple heuristic: structured commands often have specific patterns
        structured_patterns = [
            r"^\w+\([^)]+\)",  # Function call pattern
            r"^\w+:\s*\w+",  # Key-value pattern
            r"^\d+\s+\d+",  # Numeric pattern
        ]

        import re

        for pattern in structured_patterns:
            if re.match(pattern, instruction):
                return "structured"

        return "natural_language"

    def _validate_alignment(
        self, instruction: str, trajectory: dict[str, Any]
    ) -> float:
        """Validate instruction-trajectory alignment.

        Args:
            instruction: Instruction text
            trajectory: Trajectory data

        Returns:
            Alignment score (0.0-1.0)
        """
        # Would use learned models (e.g., CLIP) in production
        # For now, return a basic score based on trajectory completeness
        if not trajectory:
            return 0.0

        # Check trajectory completeness
        has_observations = "observations" in trajectory or "sensor_data" in trajectory
        has_actions = "actions" in trajectory

        if has_observations and has_actions:
            return 0.8  # Good alignment
        elif has_observations or has_actions:
            return 0.5  # Partial alignment
        else:
            return 0.2  # Poor alignment

    def _score_instruction_quality(self, instruction: str) -> float:
        """Score instruction quality.

        Args:
            instruction: Instruction text

        Returns:
            Quality score (0.0-1.0)
        """
        score = 1.0

        # Check length
        if len(instruction) < self.min_instruction_length:
            score *= 0.3
        elif len(instruction) > 1000:
            score *= 0.8  # Very long instructions may be noisy

        # Check for common quality issues
        instruction_lower = instruction.lower()
        quality_issues = [
            "test",
            "placeholder",
            "todo",
            "fixme",
            "xxx",
        ]

        if any(issue in instruction_lower for issue in quality_issues):
            score *= 0.5

        return max(0.0, min(1.0, score))

