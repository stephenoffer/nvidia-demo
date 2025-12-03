"""Domain Randomization Utilities for Sim-to-Real Transfer.

Implements domain randomization techniques for robust sim-to-real transfer,
following NVIDIA's Simulation Principle: if a model masters 1 million realities
with different physical parameters, it will likely zero-shot transfer to reality.

Key Concepts:
- Visual Randomization: Lighting, textures, colors, backgrounds
- Physical Randomization: Friction, mass, damping, gravity
- Geometric Randomization: Object sizes, shapes, positions
- Sensor Randomization: Camera noise, calibration errors
- Dynamics Randomization: Actuator delays, joint limits, compliance
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DomainRandomizationConfig:
    """Configuration for domain randomization parameters."""

    # Visual randomization
    randomize_lighting: bool = True
    lighting_intensity_range: Tuple[float, float] = (0.5, 2.0)
    lighting_color_range: Tuple[float, float] = (0.8, 1.2)
    randomize_textures: bool = True
    texture_scale_range: Tuple[float, float] = (0.5, 2.0)
    randomize_backgrounds: bool = True
    background_color_range: Tuple[float, float] = (0.0, 1.0)

    # Physical randomization
    randomize_friction: bool = True
    friction_range: Tuple[float, float] = (0.1, 2.0)
    randomize_mass: bool = True
    mass_scale_range: Tuple[float, float] = (0.5, 2.0)
    randomize_damping: bool = True
    damping_range: Tuple[float, float] = (0.0, 0.1)
    randomize_gravity: bool = True
    gravity_scale_range: Tuple[float, float] = (0.8, 1.2)

    # Geometric randomization
    randomize_object_sizes: bool = True
    size_scale_range: Tuple[float, float] = (0.8, 1.2)
    randomize_object_positions: bool = True
    position_noise_std: float = 0.05
    randomize_object_orientations: bool = True
    orientation_noise_std: float = 0.1

    # Sensor randomization
    randomize_camera_noise: bool = True
    camera_noise_std: float = 0.01
    randomize_camera_calibration: bool = True
    calibration_error_range: Tuple[float, float] = (-0.05, 0.05)
    randomize_imu_noise: bool = True
    imu_noise_std: float = 0.001

    # Dynamics randomization
    randomize_actuator_delays: bool = True
    actuator_delay_range: Tuple[float, float] = (0.0, 0.02)
    randomize_joint_limits: bool = True
    joint_limit_noise_std: float = 0.05
    randomize_compliance: bool = True
    compliance_range: Tuple[float, float] = (0.0, 0.1)

    # Environment count for randomization
    num_randomized_environments: int = 10000


class DomainRandomizer:
    """Domain randomization for sim-to-real transfer.

    Applies randomization to simulation parameters to improve robustness
    and enable zero-shot transfer to real-world scenarios.
    """

    def __init__(self, config: Optional[DomainRandomizationConfig] = None):
        """Initialize domain randomizer.

        Args:
            config: Domain randomization configuration
        """
        self.config = config or DomainRandomizationConfig()
        self._rng = np.random.RandomState()

    def randomize_visual_params(self) -> Dict[str, Any]:
        """Generate randomized visual parameters.

        Returns:
            Dictionary of randomized visual parameters
        """
        params = {}
        if self.config.randomize_lighting:
            params["lighting_intensity"] = self._rng.uniform(
                *self.config.lighting_intensity_range
            )
            params["lighting_color"] = [
                self._rng.uniform(*self.config.lighting_color_range) for _ in range(3)
            ]
        if self.config.randomize_textures:
            params["texture_scale"] = self._rng.uniform(*self.config.texture_scale_range)
        if self.config.randomize_backgrounds:
            params["background_color"] = [
                self._rng.uniform(*self.config.background_color_range) for _ in range(3)
            ]
        return params

    def randomize_physical_params(self) -> Dict[str, Any]:
        """Generate randomized physical parameters.

        Returns:
            Dictionary of randomized physical parameters
        """
        params = {}
        if self.config.randomize_friction:
            params["friction"] = self._rng.uniform(*self.config.friction_range)
        if self.config.randomize_mass:
            params["mass_scale"] = self._rng.uniform(*self.config.mass_scale_range)
        if self.config.randomize_damping:
            params["damping"] = self._rng.uniform(*self.config.damping_range)
        if self.config.randomize_gravity:
            params["gravity_scale"] = self._rng.uniform(*self.config.gravity_scale_range)
        return params

    def randomize_geometric_params(self) -> Dict[str, Any]:
        """Generate randomized geometric parameters.

        Returns:
            Dictionary of randomized geometric parameters
        """
        params = {}
        if self.config.randomize_object_sizes:
            params["size_scale"] = self._rng.uniform(*self.config.size_scale_range)
        if self.config.randomize_object_positions:
            params["position_noise"] = self._rng.normal(
                0, self.config.position_noise_std, size=3
            ).tolist()
        if self.config.randomize_object_orientations:
            params["orientation_noise"] = self._rng.normal(
                0, self.config.orientation_noise_std, size=3
            ).tolist()
        return params

    def randomize_sensor_params(self) -> Dict[str, Any]:
        """Generate randomized sensor parameters.

        Returns:
            Dictionary of randomized sensor parameters
        """
        params = {}
        if self.config.randomize_camera_noise:
            params["camera_noise"] = self._rng.normal(0, self.config.camera_noise_std)
        if self.config.randomize_camera_calibration:
            params["calibration_error"] = self._rng.uniform(
                *self.config.calibration_error_range
            )
        if self.config.randomize_imu_noise:
            params["imu_noise"] = self._rng.normal(0, self.config.imu_noise_std, size=6).tolist()
        return params

    def randomize_dynamics_params(self) -> Dict[str, Any]:
        """Generate randomized dynamics parameters.

        Returns:
            Dictionary of randomized dynamics parameters
        """
        params = {}
        if self.config.randomize_actuator_delays:
            params["actuator_delay"] = self._rng.uniform(*self.config.actuator_delay_range)
        if self.config.randomize_joint_limits:
            params["joint_limit_noise"] = self._rng.normal(
                0, self.config.joint_limit_noise_std
            )
        if self.config.randomize_compliance:
            params["compliance"] = self._rng.uniform(*self.config.compliance_range)
        return params

    def generate_randomized_environment(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """Generate a complete randomized environment configuration.

        Args:
            seed: Random seed for reproducibility

        Returns:
            Complete randomized environment configuration
        """
        if seed is not None:
            self._rng.seed(seed)

        return {
            "visual": self.randomize_visual_params(),
            "physical": self.randomize_physical_params(),
            "geometric": self.randomize_geometric_params(),
            "sensor": self.randomize_sensor_params(),
            "dynamics": self.randomize_dynamics_params(),
        }

    def generate_randomized_environments(
        self, num_environments: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Generate multiple randomized environment configurations.

        Args:
            num_environments: Number of environments to generate

        Returns:
            List of randomized environment configurations
        """
        num_envs = num_environments or self.config.num_randomized_environments
        return [
            self.generate_randomized_environment(seed=i) for i in range(num_envs)
        ]


def apply_domain_randomization(
    data: Dict[str, Any], config: Optional[DomainRandomizationConfig] = None
) -> Dict[str, Any]:
    """Apply domain randomization to a data sample.

    Args:
        data: Input data sample
        config: Domain randomization configuration

    Returns:
        Randomized data sample
    """
    randomizer = DomainRandomizer(config)
    env_config = randomizer.generate_randomized_environment()

    randomized_data = data.copy()

    if "observations" in randomized_data:
        obs = randomized_data["observations"]
        if "image" in obs and env_config["sensor"].get("camera_noise"):
            noise = np.random.normal(0, env_config["sensor"]["camera_noise"], obs["image"].shape)
            obs["image"] = np.clip(obs["image"] + noise, 0, 1)

    if "actions" in randomized_data and env_config["dynamics"].get("actuator_delay"):
        delay = env_config["dynamics"]["actuator_delay"]
        randomized_data["action_delay"] = delay

    randomized_data["domain_randomization"] = env_config
    return randomized_data

