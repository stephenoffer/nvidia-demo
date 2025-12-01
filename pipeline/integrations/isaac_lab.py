"""Integration with NVIDIA Isaac Lab for simulation data.

Isaac Lab is NVIDIA's open-source robotics simulation framework.
See: https://github.com/NVIDIA/Isaac-Lab

Uses Ray Data for distributed dataset processing.
See: https://docs.ray.io/en/latest/data/data.html
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import ray
from ray.data import Dataset

from pipeline.exceptions import DataSourceError
from pipeline.utils.decorators import handle_errors, log_execution_time
from pipeline.utils.constants import _DEFAULT_BATCH_SIZE

logger = logging.getLogger(__name__)

# Constants
_MAX_FILES_TO_SCAN = 10000
_MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024 * 1024  # 10GB


class IsaacLabLoader:
    """Loader for Isaac Lab simulation data.

    Isaac Lab generates massive amounts of simulation data for robot training,
    including joint angles, observations, actions, and rewards. This loader
    processes Isaac Lab trajectories and integrates them into the curation pipeline.
    """

    def __init__(
        self,
        simulation_path: Union[str, Path],
        robot_type: str = "humanoid",
        include_observations: bool = True,
        include_actions: bool = True,
        include_rewards: bool = True,
        robot_id_field: str = "robot_id",
        support_multi_robot: bool = True,
        include_task_definitions: bool = True,
        include_environment_metadata: bool = True,
        parse_observation_spaces: bool = True,
        max_files: Optional[int] = None,
        batch_size: Optional[int] = None,
    ):
        """Initialize Isaac Lab loader.

        Args:
            simulation_path: Path to Isaac Lab simulation data directory
            robot_type: Type of robot (humanoid, quadruped, etc.)
            include_observations: Whether to include observation data
            include_actions: Whether to include action data
            include_rewards: Whether to include reward signals
            robot_id_field: Field name for robot ID (for multi-robot support)
            support_multi_robot: Whether to support multi-robot scenarios
            include_task_definitions: Whether to load task definitions
            include_environment_metadata: Whether to load environment metadata
            parse_observation_spaces: Whether to parse structured observation spaces
            max_files: Maximum number of files to process (None = unlimited)
            batch_size: Batch size for processing (None = use default)

        Raises:
            ValueError: If parameters are invalid
            DataSourceError: If simulation_path is invalid
        """
        # Validate and convert path
        if isinstance(simulation_path, str):
            if not simulation_path or not simulation_path.strip():
                raise ValueError("simulation_path cannot be empty")
            self.simulation_path = Path(simulation_path)
        elif isinstance(simulation_path, Path):
            self.simulation_path = simulation_path
        else:
            raise ValueError(f"simulation_path must be str or Path, got {type(simulation_path)}")
        
        # Validate parameters
        if not isinstance(robot_type, str) or not robot_type.strip():
            raise ValueError(f"robot_type must be non-empty str, got {type(robot_type)}")
        
        if not isinstance(robot_id_field, str) or not robot_id_field.strip():
            raise ValueError(f"robot_id_field must be non-empty str, got {type(robot_id_field)}")
        
        if max_files is not None and max_files <= 0:
            raise ValueError(f"max_files must be positive, got {max_files}")
        
        if batch_size is not None and batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        
        self.robot_type = robot_type
        self.include_observations = bool(include_observations)
        self.include_actions = bool(include_actions)
        self.include_rewards = bool(include_rewards)
        self.robot_id_field = robot_id_field
        self.support_multi_robot = bool(support_multi_robot)
        self.include_task_definitions = bool(include_task_definitions)
        self.include_environment_metadata = bool(include_environment_metadata)
        self.parse_observation_spaces = bool(parse_observation_spaces)
        self.max_files = max_files
        self.batch_size = batch_size if batch_size is not None else _DEFAULT_BATCH_SIZE

    @log_execution_time
    @handle_errors(error_class=DataSourceError)
    def load(self) -> Dataset:
        """Load Isaac Lab simulation data.

        Returns:
            Ray Dataset containing simulation trajectories

        Raises:
            DataSourceError: If loading fails
        """
        # Validate path exists
        if not self.simulation_path.exists():
            raise DataSourceError(f"Isaac Lab simulation path does not exist: {self.simulation_path}")

        if not self.simulation_path.is_dir():
            raise DataSourceError(f"Isaac Lab simulation path is not a directory: {self.simulation_path}")

        logger.info(f"Loading Isaac Lab data from {self.simulation_path}")

        # Isaac Lab typically stores data in HDF5 or Parquet format
        # Look for trajectory files
        try:
            trajectory_files = list(self.simulation_path.glob("**/*trajectory*.parquet"))
            trajectory_files.extend(list(self.simulation_path.glob("**/*trajectory*.h5")))
            
            # Apply max_files limit
            if self.max_files is not None:
                trajectory_files = trajectory_files[:self.max_files]
        except (OSError, PermissionError) as e:
            raise DataSourceError(f"Failed to scan Isaac Lab directory: {e}") from e

        if not trajectory_files:
            # Try alternative patterns
            try:
                trajectory_files = list(self.simulation_path.glob("**/*.parquet"))
                if self.max_files is not None:
                    trajectory_files = trajectory_files[:self.max_files]
            except (OSError, PermissionError) as e:
                raise DataSourceError(f"Failed to scan for Parquet files: {e}") from e

        if not trajectory_files:
            logger.warning(f"No trajectory files found in {self.simulation_path}")
            return ray.data.from_items([])

        logger.info(f"Found {len(trajectory_files)} trajectory files")

        # Load Parquet files
        parquet_files = [f for f in trajectory_files if f.suffix == ".parquet"]
        if parquet_files:
            try:
                datasets = []
                for f in parquet_files:
                    # Validate file exists and is readable
                    if not f.exists():
                        logger.warning(f"Parquet file does not exist: {f}")
                        continue
                    
                    if not f.is_file():
                        logger.warning(f"Parquet path is not a file: {f}")
                        continue
                    
                    # Check file size
                    file_size = f.stat().st_size
                    if file_size > _MAX_FILE_SIZE_BYTES:
                        logger.warning(
                            f"Parquet file {f} is {file_size} bytes, "
                            f"exceeds recommended size of {_MAX_FILE_SIZE_BYTES}"
                        )
                    
                    try:
                        dataset = ray.data.read_parquet(str(f))
                        datasets.append(dataset)
                    except Exception as e:
                        logger.error(f"Failed to load Parquet file {f}: {e}")
                        continue
                
                if not datasets:
                    logger.warning("No valid Parquet files loaded")
                    return ray.data.from_items([])
                
                combined = ray.data.union(*datasets) if len(datasets) > 1 else datasets[0]
            except Exception as e:
                raise DataSourceError(f"Failed to load Parquet files: {e}") from e
        else:
            # For HDF5 files, would need custom loader
            logger.warning("HDF5 support not implemented, skipping")
            return ray.data.from_items([])

        # Add metadata and format for pipeline using map_batches
        def format_batch(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            """Format a batch of trajectories.

            Args:
                batch: List of trajectory items

            Returns:
                List of formatted trajectory items
            """
            if not batch:
                return []
            
            formatted_items = []
            for item in batch:
                try:
                    if not isinstance(item, dict):
                        logger.warning(f"Skipping non-dict item: {type(item)}")
                        continue
                    
                    formatted_item = self._format_trajectory(item)
                    formatted_items.append(formatted_item)
                except Exception as e:
                    logger.warning(f"Failed to format trajectory item: {e}", exc_info=True)
                    # Include item with error flag
                    if isinstance(item, dict):
                        item["format_error"] = str(e)
                        formatted_items.append(item)
            
            return formatted_items

        formatted = combined.map_batches(
            format_batch,
            batch_size=self.batch_size,
            batch_format="pandas",  # Specify batch format for consistency
        )

        return formatted

    def _format_trajectory(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Format Isaac Lab trajectory for pipeline processing.

        Args:
            item: Raw trajectory data

        Returns:
            Formatted trajectory item

        Raises:
            ValueError: If item is invalid
        """
        if not isinstance(item, dict):
            raise ValueError(f"Item must be a dictionary, got {type(item)}")
        
        formatted: Dict[str, Any] = {
            "data_type": "sensor",
            "format": "isaac_lab",
            "robot_type": self.robot_type,
            "source": "simulation",
        }

        # Multi-robot support: Extract robot_id
        if self.support_multi_robot:
            robot_id = (
                item.get(self.robot_id_field) or 
                item.get("robot_id") or 
                item.get("agent_id")
            )
            if robot_id is not None:
                formatted["robot_id"] = str(robot_id)
            else:
                formatted["robot_id"] = "default"  # Single robot default

        # Extract sensor data (joint angles, velocities, etc.)
        sensor_data: Dict[str, Any] = {}

        if "observations" in item and self.include_observations:
            obs = item["observations"]
            if obs is not None:
                if self.parse_observation_spaces and isinstance(obs, dict):
                    # Preserve structured observation spaces
                    sensor_data["observations"] = obs
                    try:
                        sensor_data["observation_space"] = self._parse_observation_space(obs)
                    except Exception as e:
                        logger.warning(f"Failed to parse observation space: {e}")
                        sensor_data["observation_space_error"] = str(e)
                else:
                    sensor_data["observations"] = obs

        if "actions" in item and self.include_actions:
            actions = item["actions"]
            if actions is not None:
                sensor_data["actions"] = actions

        if "rewards" in item and self.include_rewards:
            rewards = item["rewards"]
            if rewards is not None:
                sensor_data["rewards"] = rewards

        # Common Isaac Lab fields
        for field in ["joint_positions", "joint_velocities", "base_pose"]:
            if field in item and item[field] is not None:
                sensor_data[field] = item[field]

        if sensor_data:
            formatted["sensor_data"] = sensor_data

        # Add metadata
        for field in ["episode_id", "step"]:
            if field in item and item[field] is not None:
                formatted[field] = item[field]

        # Task definitions support
        if self.include_task_definitions:
            task_id = item.get("task_id") or item.get("task_name")
            if task_id is not None:
                formatted["task_id"] = str(task_id)
                formatted["task_name"] = str(task_id)

        # Environment metadata support
        if self.include_environment_metadata:
            env_metadata: Dict[str, Any] = {}
            metadata_fields = [
                "physics_params", 
                "scene_config", 
                "domain_randomization", 
                "environment_id"
            ]
            for key in metadata_fields:
                if key in item and item[key] is not None:
                    env_metadata[key] = item[key]
            if env_metadata:
                formatted["environment_metadata"] = env_metadata

        return formatted

    def _parse_observation_space(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """Parse structured observation space from Isaac Lab data.

        Args:
            observations: Observation dictionary

        Returns:
            Observation space structure
        """
        if not isinstance(observations, dict):
            return {}
        
        space: Dict[str, Any] = {}
        for key, value in observations.items():
            if value is None:
                space[key] = {"type": "null"}
            elif isinstance(value, (list, tuple)):
                if len(value) > 0:
                    space[key] = {
                        "shape": (len(value),),
                        "dtype": type(value[0]).__name__ if value else "unknown",
                    }
                else:
                    space[key] = {"shape": (0,), "dtype": "unknown"}
            elif isinstance(value, dict):
                space[key] = {
                    "type": "dict",
                    "keys": list(value.keys()),
                }
            else:
                space[key] = {
                    "shape": (),
                    "dtype": type(value).__name__,
                }
        return space

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the Isaac Lab dataset.

        Returns:
            Dictionary of dataset statistics

        Raises:
            DataSourceError: If statistics collection fails
        """
        try:
            dataset = self.load()

            # Estimate count without full materialization
            # For statistics, we can use sampling or estimate from blocks
            try:
                # Try to estimate from dataset metadata if available
                # Fallback: sample and estimate
                sample_batch = next(dataset.iter_batches(batch_size=100, prefetch_batches=0), None)
                if sample_batch is not None:
                    # Estimate: sample size * number of blocks (rough estimate)
                    num_blocks = dataset.num_blocks() if hasattr(dataset, 'num_blocks') else 1
                    estimated_count = len(sample_batch) * max(1, num_blocks)
                    count = estimated_count
                else:
                    count = 0
            except (StopIteration, AttributeError, RuntimeError):
                # Fallback: use num_rows if available
                try:
                    count = dataset.num_rows() if hasattr(dataset, 'num_rows') else 0
                except (AttributeError, RuntimeError):
                    count = 0

            # Sample to get structure (materializes 1 item - acceptable for stats)
            structure: Dict[str, Any] = {}
            try:
                sample_batch = next(dataset.iter_batches(batch_size=1, prefetch_batches=0), None)
                if sample_batch is not None:
                    if hasattr(sample_batch, 'iloc'):
                        sample = sample_batch.iloc[0].to_dict()
                    elif isinstance(sample_batch, list) and len(sample_batch) > 0:
                        sample = sample_batch[0]
                    else:
                        sample = sample_batch
                    
                    if isinstance(sample, dict):
                        structure = sample
            except (StopIteration, AttributeError, RuntimeError):
                pass

            return {
                "num_trajectories": count,
                "robot_type": self.robot_type,
                "data_structure": list(structure.keys()) if structure else [],
            }
        except Exception as e:
            raise DataSourceError(f"Failed to get statistics: {e}") from e
