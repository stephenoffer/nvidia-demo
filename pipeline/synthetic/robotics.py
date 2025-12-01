"""Synthetic robotics trajectory generator.

Generates realistic robot trajectories for testing and demos,
simulating the kind of data collected from Isaac Lab simulations.

Uses Ray Data for distributed dataset creation.
See: https://docs.ray.io/en/latest/data/data.html
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np  # https://numpy.org/
import ray
from ray.data import Dataset

logger = logging.getLogger(__name__)


class RoboticsTrajectoryGenerator:
    """Generate synthetic robotics trajectories.

    Creates realistic robot motion data including joint positions,
    velocities, base poses, and reward signals.
    """

    def __init__(
        self,
        num_joints: int = 12,
        trajectory_length: int = 100,
        num_trajectories: int = 1000,
        robot_type: str = "humanoid",
        include_rewards: bool = True,
        seed: Optional[int] = None,
    ):
        """Initialize robotics trajectory generator.

        Args:
            num_joints: Number of joints in robot
            trajectory_length: Length of each trajectory in steps
            num_trajectories: Number of trajectories to generate
            robot_type: Type of robot (humanoid, quadruped, etc.)
            include_rewards: Whether to include reward signals
            seed: Random seed for reproducibility
        """
        self.num_joints = num_joints
        self.trajectory_length = trajectory_length
        self.num_trajectories = num_trajectories
        self.robot_type = robot_type
        self.include_rewards = include_rewards

        if seed is not None:
            np.random.seed(seed)

    def generate_trajectory(self, trajectory_id: int) -> Dict[str, Any]:
        """Generate a single synthetic trajectory.

        Args:
            trajectory_id: Unique identifier for trajectory

        Returns:
            Dictionary containing trajectory data
        """
        # Generate joint positions using sinusoidal patterns
        # Simulates realistic robot motion
        t = np.linspace(0, 2 * np.pi, self.trajectory_length)

        # Each joint has different frequency and amplitude
        joint_positions = np.zeros((self.trajectory_length, self.num_joints))
        joint_velocities = np.zeros((self.trajectory_length, self.num_joints))

        for i in range(self.num_joints):
            # Vary frequencies and amplitudes for realistic motion
            freq = 0.3 + i * 0.1 + np.random.uniform(-0.05, 0.05)
            amp = 0.5 + i * 0.1 + np.random.uniform(-0.1, 0.1)
            phase = np.random.uniform(0, 2 * np.pi)

            # Position
            joint_positions[:, i] = amp * np.sin(freq * t + phase)

            # Velocity (derivative)
            joint_velocities[:, i] = amp * freq * np.cos(freq * t + phase)

        # Generate base pose (robot movement in space)
        base_pose = self._generate_base_trajectory()

        # Generate observations (simplified)
        observations = self._generate_observations(joint_positions, base_pose)

        # Generate actions (simplified control signals)
        actions = self._generate_actions(joint_positions, joint_velocities)

        trajectory = {
            "trajectory_id": trajectory_id,
            "episode_id": f"synth_{trajectory_id:06d}",
            "robot_type": self.robot_type,
            "joint_positions": joint_positions.tolist(),
            "joint_velocities": joint_velocities.tolist(),
            "base_pose": base_pose.tolist(),
            "observations": observations.tolist(),
            "actions": actions.tolist(),
            "data_type": "sensor",
            "format": "synthetic",
            "source": "synthetic_generator",
        }

        # Add rewards if requested
        if self.include_rewards:
            rewards = self._generate_rewards()
            trajectory["rewards"] = rewards.tolist()

        return trajectory

    def _generate_base_trajectory(self) -> np.ndarray:
        """Generate base pose trajectory.

        Returns:
            Array of base poses [trajectory_length, 3] (x, y, z)
        """
        t = np.linspace(0, 2 * np.pi, self.trajectory_length)

        # Circular or linear trajectory
        pattern = np.random.choice(["circular", "linear", "zigzag"])

        if pattern == "circular":
            radius = np.random.uniform(1.0, 3.0)
            x = radius * np.cos(t)
            y = radius * np.sin(t)
            z = 0.5 + 0.1 * np.sin(2 * t)  # Slight height variation
        elif pattern == "linear":
            x = np.linspace(0, 5, self.trajectory_length)
            y = np.random.uniform(-0.5, 0.5, self.trajectory_length)
            z = 0.5 + 0.05 * np.sin(t)
        else:  # zigzag
            x = np.linspace(0, 5, self.trajectory_length)
            y = 0.5 * np.sin(3 * t)
            z = 0.5

        return np.column_stack([x, y, z])

    def _generate_observations(
        self,
        joint_positions: np.ndarray,
        base_pose: np.ndarray,
    ) -> np.ndarray:
        """Generate observation vector.

        Args:
            joint_positions: Joint position array
            base_pose: Base pose array

        Returns:
            Observation array
        """
        # Concatenate joint positions and base pose
        observations = np.concatenate(
            [
                joint_positions.flatten().reshape(self.trajectory_length, -1),
                base_pose,
            ],
            axis=1,
        )

        # Add noise for realism
        noise = np.random.normal(0, 0.01, observations.shape)
        observations = observations + noise

        return observations

    def _generate_actions(
        self,
        joint_positions: np.ndarray,
        joint_velocities: np.ndarray,
    ) -> np.ndarray:
        """Generate action signals.

        Args:
            joint_positions: Joint positions
            joint_velocities: Joint velocities

        Returns:
            Action array
        """
        # Simple PD controller-like actions
        target_positions = np.random.uniform(-1.0, 1.0, (self.trajectory_length, self.num_joints))

        # Actions are proportional to error + derivative term
        position_error = target_positions - joint_positions
        actions = 0.5 * position_error + 0.3 * joint_velocities

        # Add noise
        noise = np.random.normal(0, 0.05, actions.shape)
        actions = actions + noise

        return actions

    def _generate_rewards(self) -> np.ndarray:
        """Generate reward signal.

        Returns:
            Reward array
        """
        # Generate increasing reward with some noise
        base_reward = np.linspace(0, 1, self.trajectory_length)
        noise = np.random.normal(0, 0.1, self.trajectory_length)
        rewards = base_reward + noise

        # Clip to valid range
        rewards = np.clip(rewards, 0, 1)

        return rewards

    def generate_batch(self, batch_size: int, start_id: int = 0) -> List[Dict[str, Any]]:
        """Generate a batch of trajectories.

        Args:
            batch_size: Number of trajectories to generate
            start_id: Starting trajectory ID

        Returns:
            List of trajectory dictionaries
        """
        trajectories = []
        for i in range(batch_size):
            trajectory = self.generate_trajectory(start_id + i)
            trajectories.append(trajectory)

        return trajectories


class SyntheticRoboticsDataset:
    """Generate synthetic robotics dataset using Ray Data."""

    def __init__(
        self,
        num_trajectories: int = 1000,
        trajectory_length: int = 100,
        num_joints: int = 12,
        robot_type: str = "humanoid",
        num_workers: int = 4,
        seed: Optional[int] = None,
    ):
        """Initialize synthetic robotics dataset generator.

        Args:
            num_trajectories: Total number of trajectories to generate
            trajectory_length: Length of each trajectory
            num_joints: Number of robot joints
            robot_type: Type of robot
            num_workers: Number of parallel workers
            seed: Random seed
        """
        self.num_trajectories = num_trajectories
        self.trajectory_length = trajectory_length
        self.num_joints = num_joints
        self.robot_type = robot_type
        self.num_workers = num_workers
        self.seed = seed

    def generate(self) -> Dataset:
        """Generate synthetic robotics dataset.

        Returns:
            Ray Dataset containing synthetic trajectories
        """
        logger.info(f"Generating {self.num_trajectories} synthetic trajectories")

        # Create generator
        generator = RoboticsTrajectoryGenerator(
            num_joints=self.num_joints,
            trajectory_length=self.trajectory_length,
            num_trajectories=self.num_trajectories,
            robot_type=self.robot_type,
            seed=self.seed,
        )

        # Generate trajectories in batches
        batch_size = max(1, self.num_trajectories // self.num_workers)
        batches = []

        for i in range(0, self.num_trajectories, batch_size):
            current_batch_size = min(batch_size, self.num_trajectories - i)
            batch = generator.generate_batch(current_batch_size, start_id=i)
            batches.extend(batch)

        # Convert to Ray Dataset using Ray Data's optimized from_items
        # Use parallel generation with Ray Data for better performance
        dataset = ray.data.from_items(batches, parallelism=self.num_workers)
        
        # Log generation info without materializing dataset
        # Use len(batches) instead of dataset.count() to avoid materialization
        logger.info(f"Generated {len(batches)} synthetic trajectory batches")
        
        # Optional: Repartition for optimal block distribution
        # This helps with downstream processing efficiency
        if len(batches) > self.num_workers:
            # Repartition to match worker count for optimal parallelism
            dataset = dataset.repartition(num_blocks=self.num_workers)
            logger.info(f"Repartitioned dataset to {self.num_workers} blocks for optimal parallelism")
