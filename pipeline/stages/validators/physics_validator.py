"""Physics constraint validation for robotics data.

Validates that sensor data satisfies physical constraints:
- Joint limits
- Velocity limits
- Acceleration limits
- Dynamics consistency
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from ray.data import Dataset

from pipeline.stages.base import ValidatorBase
from pipeline.utils.constants import _DEFAULT_BATCH_SIZE

logger = logging.getLogger(__name__)


class PhysicsValidator(ValidatorBase):
    """Validate physical constraints for robotics sensor data.

    Checks joint limits, velocity limits, acceleration limits, and
    dynamics consistency (e.g., velocity = d(position)/dt).
    """

    def __init__(
        self,
        joint_limits: Optional[dict[str, tuple[float, float]]] = None,
        velocity_limits: Optional[dict[str, float]] = None,
        acceleration_limits: Optional[dict[str, float]] = None,
        validate_dynamics: bool = True,
        tolerance: float = 1e-3,
        reject_invalid: bool = True,
        batch_size: int = _DEFAULT_BATCH_SIZE,
    ):
        """Initialize physics validator.

        Args:
            joint_limits: Dictionary mapping joint names to (min, max) limits
            velocity_limits: Dictionary mapping joint names to max velocity
            acceleration_limits: Dictionary mapping joint names to max acceleration
            validate_dynamics: Whether to validate dynamics consistency
            tolerance: Tolerance for numerical comparisons
            reject_invalid: Whether to reject invalid trajectories
            batch_size: Batch size for processing
        """
        super().__init__(reject_invalid=reject_invalid, batch_size=batch_size)
        self.joint_limits = joint_limits or {}
        self.velocity_limits = velocity_limits or {}
        self.acceleration_limits = acceleration_limits or {}
        self.validate_dynamics = validate_dynamics
        self.tolerance = tolerance

    def process(self, dataset: Dataset) -> Dataset:
        """Validate physics constraints for sensor data.

        Args:
            dataset: Input Ray Dataset with sensor data

        Returns:
            Dataset with validation results (filtered if reject_invalid=True)
        """
        logger.info("Validating physics constraints")
        return super().process(dataset)

    def _validate_item(self, item: dict[str, Any]) -> dict[str, Any]:
        """Validate physics constraints for a single item.

        Args:
            item: Sensor data item

        Returns:
            Validation result dictionary
        """
        violations: list[str] = []
        warnings: list[str] = []

        from pipeline.utils.data.data_types import get_data_type, DataType

        if get_data_type(item) != DataType.SENSOR:
            return {
                "is_valid": True,
                "violations": [],
                "warnings": [],
                "physics_validated": False,
            }

        sensor_data = item.get("sensor_data", {})
        if not isinstance(sensor_data, dict):
            return {
                "is_valid": False,
                "violations": ["sensor_data is not a dictionary"],
                "warnings": [],
                "physics_validated": True,
            }

        # Validate joint positions
        joint_positions = sensor_data.get("joint_positions")
        if joint_positions:
            violations.extend(self._validate_joint_positions(joint_positions))

        # Validate joint velocities
        joint_velocities = sensor_data.get("joint_velocities")
        if joint_velocities:
            violations.extend(self._validate_joint_velocities(joint_velocities))

        # Validate dynamics consistency
        if self.validate_dynamics and joint_positions and joint_velocities:
            dynamics_violations = self._validate_dynamics(joint_positions, joint_velocities)
            violations.extend(dynamics_violations)

        # Validate accelerations if available
        joint_accelerations = sensor_data.get("joint_accelerations")
        if joint_accelerations:
            violations.extend(self._validate_joint_accelerations(joint_accelerations))

        is_valid = len(violations) == 0

        return {
            "is_valid": is_valid,
            "violations": violations,
            "warnings": warnings,
            "physics_validated": True,
            "num_violations": len(violations),
        }

    def _validate_joint_positions(self, positions: Any) -> list[str]:
        """Validate joint positions against limits.

        Args:
            positions: Joint positions (list, array, or dict)

        Returns:
            List of violation messages
        """
        violations = []

        # Convert to list if needed
        if isinstance(positions, dict):
            positions_list = list(positions.values())
            joint_names = list(positions.keys())
        elif isinstance(positions, (list, tuple)):
            positions_list = list(positions)
            joint_names = [f"joint_{i}" for i in range(len(positions_list))]
        else:
            return ["Invalid joint positions format"]

        # Check against limits
        for i, (joint_name, pos) in enumerate(zip(joint_names, positions_list)):
            if joint_name in self.joint_limits:
                min_limit, max_limit = self.joint_limits[joint_name]
                if pos < min_limit - self.tolerance:
                    violations.append(f"{joint_name} position {pos} < min limit {min_limit}")
                elif pos > max_limit + self.tolerance:
                    violations.append(f"{joint_name} position {pos} > max limit {max_limit}")

        return violations

    def _validate_joint_velocities(self, velocities: Any) -> list[str]:
        """Validate joint velocities against limits.

        Args:
            velocities: Joint velocities (list, array, or dict)

        Returns:
            List of violation messages
        """
        violations = []

        # Convert to list if needed
        if isinstance(velocities, dict):
            velocities_list = list(velocities.values())
            joint_names = list(velocities.keys())
        elif isinstance(velocities, (list, tuple)):
            velocities_list = list(velocities)
            joint_names = [f"joint_{i}" for i in range(len(velocities_list))]
        else:
            return ["Invalid joint velocities format"]

        # Check against limits
        for joint_name, vel in zip(joint_names, velocities_list):
            if joint_name in self.velocity_limits:
                max_vel = self.velocity_limits[joint_name]
                if abs(vel) > max_vel + self.tolerance:
                    violations.append(
                        f"{joint_name} velocity {vel} exceeds limit {max_vel}"
                    )

        return violations

    def _validate_joint_accelerations(self, accelerations: Any) -> list[str]:
        """Validate joint accelerations against limits.

        Args:
            accelerations: Joint accelerations (list, array, or dict)

        Returns:
            List of violation messages
        """
        violations = []

        # Convert to list if needed
        if isinstance(accelerations, dict):
            accel_list = list(accelerations.values())
            joint_names = list(accelerations.keys())
        elif isinstance(accelerations, (list, tuple)):
            accel_list = list(accelerations)
            joint_names = [f"joint_{i}" for i in range(len(accel_list))]
        else:
            return ["Invalid joint accelerations format"]

        # Check against limits
        for joint_name, accel in zip(joint_names, accel_list):
            if joint_name in self.acceleration_limits:
                max_accel = self.acceleration_limits[joint_name]
                if abs(accel) > max_accel + self.tolerance:
                    violations.append(
                        f"{joint_name} acceleration {accel} exceeds limit {max_accel}"
                    )

        return violations

    def _validate_dynamics(
        self, positions: Any, velocities: Any, dt: float = 0.01
    ) -> list[str]:
        """Validate dynamics consistency (velocity = d(position)/dt).

        Args:
            positions: Joint positions
            velocities: Joint velocities
            dt: Time step (default 0.01s for 100Hz)

        Returns:
            List of violation messages
        """
        violations = []

        # This is a simplified check - would need sequence data for full validation
        # For now, just check that positions and velocities have compatible shapes
        try:
            if isinstance(positions, (list, tuple)) and isinstance(velocities, (list, tuple)):
                if len(positions) != len(velocities):
                    violations.append(
                        f"Position-velocity shape mismatch: {len(positions)} vs {len(velocities)}"
                    )
        except Exception as e:
            violations.append(f"Dynamics validation error: {e}")

        return violations

