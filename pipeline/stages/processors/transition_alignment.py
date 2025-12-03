"""Transition alignment stage for reinforcement learning data.

Creates aligned (s_t, a_t, r_t, s_{t+1}) tuples from trajectory data.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Dict


from ray.data import Dataset

from pipeline.stages.base import ProcessorBase
from pipeline.utils.constants import _DEFAULT_BATCH_SIZE

logger = logging.getLogger(__name__)


class TransitionAlignmentStage(ProcessorBase):
    """Align observations, actions, and rewards into RL transitions.

    Creates (s_t, a_t, r_t, s_{t+1}) tuples from trajectory data,
    ensuring temporal consistency and handling missing transitions.
    """

    def __init__(
        self,
        observation_field: str = "observations",
        action_field: str = "actions",
        reward_field: str = "rewards",
        next_observation_field: Optional[str] = None,
        step_field: str = "step",
        episode_id_field: str = "episode_id",
        validate_consistency: bool = True,
        handle_missing: str = "skip",
        batch_size: int = _DEFAULT_BATCH_SIZE,
    ):
        """Initialize transition alignment stage.

        Args:
            observation_field: Field name containing observations
            action_field: Field name containing actions
            reward_field: Field name containing rewards
            next_observation_field: Field name for next observations (auto-detect if None)
            step_field: Field name containing step number
            episode_id_field: Field name containing episode ID
            validate_consistency: Whether to validate transition consistency
            handle_missing: How to handle missing transitions
            batch_size: Batch size for processing
        """
        super().__init__(batch_size=batch_size)
        self.observation_field = observation_field
        self.action_field = action_field
        self.reward_field = reward_field
        self.next_observation_field = next_observation_field
        self.step_field = step_field
        self.episode_id_field = episode_id_field
        self.validate_consistency = validate_consistency
        self.handle_missing = handle_missing

    def process(self, dataset: Dataset) -> Dataset:
        """Create aligned transitions from trajectory data.

        Args:
            dataset: Input Ray Dataset with trajectory data

        Returns:
            Dataset with aligned transitions
        """
        logger.info("Aligning transitions (s_t, a_t, r_t, s_{t+1})")

        def create_transitions_batch(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
            """Create transitions from a batch of trajectory items."""
            transitions = []

            episodes: dict[str, list[dict[str, Any]]] = {}
            for item in batch:
                episode_id = item.get(self.episode_id_field, "unknown")
                if episode_id not in episodes:
                    episodes[episode_id] = []
                episodes[episode_id].append(item)

            for episode_id, items in episodes.items():
                items_sorted = sorted(items, key=lambda x: x.get(self.step_field, 0))

                for i in range(len(items_sorted) - 1):
                    current = items_sorted[i]
                    next_item = items_sorted[i + 1]

                    transition = self._create_transition(current, next_item)
                    if transition:
                        transitions.append(transition)

            return transitions

        return dataset.map_batches(
            create_transitions_batch,
            batch_size=self.batch_size,
            batch_format="pandas",  # Specify batch format for consistency
        )

    def _process_item(self, item: dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process item - not used for transition alignment.

        Transition alignment requires batch processing to create pairs.
        """
        return None

    def _create_transition(
        self, current: dict[str, Any], next_item: dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Create a single transition from current and next items.

        Args:
            current: Current timestep data
            next_item: Next timestep data

        Returns:
            Transition dictionary or None if invalid
        """
        try:
            # Extract components
            s_t = self._extract_field(current, self.observation_field)
            a_t = self._extract_field(current, self.action_field)
            r_t = self._extract_field(current, self.reward_field)

            # Next observation is from next timestep
            s_t_next = self._extract_field(next_item, self.observation_field)

            # Validate all components present
            if s_t is None or a_t is None or r_t is None or s_t_next is None:
                if self.handle_missing == "skip":
                    return None
                elif self.handle_missing == "zero":
                    s_t = s_t or self._zero_like(s_t_next)
                    a_t = a_t or self._zero_like(a_t) if a_t is not None else None
                    r_t = r_t or 0.0
                    s_t_next = s_t_next or self._zero_like(s_t)
                elif self.handle_missing == "interpolate":
                    # Interpolation would require more context
                    return None

            # Create transition
            transition = {
                "state": s_t,
                "action": a_t,
                "reward": float(r_t) if r_t is not None else 0.0,
                "next_state": s_t_next,
                "done": self._is_done(current, next_item),
                "episode_id": current.get(self.episode_id_field, "unknown"),
                "step": current.get(self.step_field, 0),
                "transition_type": "aligned",
            }

            # Validate consistency if requested
            if self.validate_consistency:
                if not self._validate_transition(transition):
                    logger.warning("Transition validation failed")
                    transition["transition_validation_warning"] = True

            return transition

        except Exception as e:
            logger.warning(f"Failed to create transition: {e}")
            return None

    def _extract_field(self, item: dict[str, Any], field_name: str) -> Any:
        """Extract field from item, handling nested structures.

        Args:
            item: Data item
            field_name: Field name to extract

        Returns:
            Field value or None if not found
        """
        from pipeline.utils.field_extraction import extract_field

        return extract_field(item, field_name, default=None)

    def _zero_like(self, reference: Any) -> Any:
        """Create zero-filled array like reference using GPU acceleration.

        Args:
            reference: Reference array or value

        Returns:
            Zero-filled array with same shape/type
        """
        try:
            import cupy as cp

            if isinstance(reference, cp.ndarray):
                return cp.zeros_like(reference)
        except ImportError:
            pass

        try:
            import numpy as np

            if isinstance(reference, (list, tuple)):
                return [0.0] * len(reference)
            elif hasattr(reference, "shape"):
                return np.zeros_like(reference)
            else:
                return 0.0
        except ImportError:
            if isinstance(reference, (list, tuple)):
                return [0.0] * len(reference)
            else:
                return 0.0

    def _is_done(self, current: dict[str, Any], next_item: dict[str, Any]) -> bool:
        """Determine if transition is terminal (episode done).

        Args:
            current: Current timestep
            next_item: Next timestep

        Returns:
            True if episode is done
        """
        # Check if episode IDs differ
        current_ep = current.get(self.episode_id_field)
        next_ep = next_item.get(self.episode_id_field)
        if current_ep != next_ep:
            return True

        # Check for done flag
        if "done" in current:
            return bool(current["done"])
        if "is_episode_end" in current:
            return bool(current["is_episode_end"])

        return False

    def _validate_transition(self, transition: dict[str, Any]) -> bool:
        """Validate transition consistency.

        Args:
            transition: Transition dictionary

        Returns:
            True if transition is valid
        """
        # Check all required fields present
        required = ["state", "action", "reward", "next_state"]
        if not all(field in transition for field in required):
            return False

        # Check state shapes match
        s_t = transition["state"]
        s_t_next = transition["next_state"]

        if isinstance(s_t, (list, tuple)) and isinstance(s_t_next, (list, tuple)):
            if len(s_t) != len(s_t_next):
                logger.warning("State shape mismatch in transition")
                return False

        # Check reward is numeric
        reward = transition["reward"]
        if not isinstance(reward, (int, float)):
            return False

        return True

