"""Video generation from simulation trajectories.

Generates visually appealing videos from simulation data for demos
and visualization purposes.

Uses Ray for distributed video generation.
See: https://docs.ray.io/
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import ray
from ray.data import Dataset

from pipeline.visualization.renderer import SimulationRenderer

logger = logging.getLogger(__name__)


# Fractional GPU allocation (0.1) may not work correctly in all Ray versions
# Ray's fractional GPU support is experimental and may cause resource allocation issues
# Use integer GPU allocation (1) or verify fractional GPU support in your Ray version
@ray.remote(num_gpus=1, num_cpus=2, memory=2 * 1024 * 1024 * 1024)  # 2GB memory limit
class VideoGeneratorActor:
    """Ray actor for parallel video generation."""

    def __init__(self, resolution: tuple, fps: int):
        """Initialize video generator actor.

        Args:
            resolution: Video resolution (width, height)
            fps: Frames per second
        """
        self.renderer = SimulationRenderer(
            mode="headless",
            resolution=resolution,
            fps=fps,
        )
        self.resolution = resolution
        self.fps = fps

    def generate_video(
        self,
        trajectory: Dict[str, Any],
        output_path: str,
    ) -> Dict[str, Any]:
        """Generate video from trajectory.

        Args:
            trajectory: Trajectory data
            output_path: Output video path

        Returns:
            Metadata about generated video
        """
        try:
            frames = self.renderer.render_trajectory(
                trajectory,
                output_path=output_path,
                show=False,
            )

            return {
                "success": True,
                "output_path": output_path,
                "num_frames": len(frames),
                "duration": len(frames) / self.fps,
            }
        except Exception as e:
            logger.error(f"Error generating video: {e}")
            return {
                "success": False,
                "error": str(e),
                "output_path": output_path,
            }


class VideoGenerator:
    """Generate videos from simulation trajectories at scale.

    Supports both large-scale batch processing and single trajectory
    rendering for demos.
    """

    def __init__(
        self,
        resolution: tuple = (1280, 720),
        fps: int = 30,
        num_workers: int = 4,
    ):
        """Initialize video generator.

        Args:
            resolution: Video resolution (width, height)
            fps: Frames per second
            num_workers: Number of parallel workers for batch processing
        """
        self.resolution = resolution
        self.fps = fps
        self.num_workers = num_workers

    def generate_from_dataset(
        self,
        dataset: Dataset,
        output_dir: str,
        max_videos: Optional[int] = None,
    ) -> Dataset:
        """Generate videos from a Ray Dataset of trajectories.

        Args:
            dataset: Ray Dataset containing trajectories
            output_dir: Output directory for videos
            max_videos: Maximum number of videos to generate (None for all)

        Returns:
            Dataset with video metadata
        """
        logger.info(f"Generating videos from dataset (max: {max_videos})")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Create video generator actors with proper resource limits
        actors = [
            VideoGeneratorActor.remote(self.resolution, self.fps) for _ in range(self.num_workers)
        ]

        # Process trajectories in batches with async processing
        video_metadata = []
        trajectory_count = 0

        from pipeline.utils.constants import _DEFAULT_BATCH_SIZE

        for batch in dataset.iter_batches(batch_size=_DEFAULT_BATCH_SIZE):
            futures = []

            for item in batch:
                if max_videos and trajectory_count >= max_videos:
                    break

                # Generate output path
                video_path = output_path / f"trajectory_{trajectory_count:06d}.mp4"

                # Assign to actor (round-robin)
                actor = actors[trajectory_count % len(actors)]
                future = actor.generate_video.remote(item, str(video_path))
                futures.append((future, trajectory_count))

                trajectory_count += 1

            # Use ray.wait() for async processing of video generation
            if futures:
                # Extract futures and indices separately for batch processing
                future_list = [fut for fut, _ in futures]
                indices = [idx for _, idx in futures]
                
                # Process asynchronously as videos complete
                remaining = future_list
                completed_indices = []
                completed_results = []
                
                while remaining:
                    ready, remaining = ray.wait(
                        remaining, 
                        num_returns=min(3, len(remaining)),  # Process up to 3 at a time
                        timeout=300.0,  # 5 minute timeout per video
                    )
                    if ready:
                        try:
                            batch_results = ray.get(ready)
                            completed_results.extend(batch_results)
                            # Track which indices completed
                            for i, fut in enumerate(future_list):
                                if fut in ready:
                                    completed_indices.append(indices[i])
                        except Exception as e:
                            logger.error(f"Error generating video: {e}", exc_info=True)
                            # Continue with remaining videos
                    else:
                        # Timeout - get remaining synchronously
                        if remaining:
                            try:
                                remaining_results = ray.get(remaining, timeout=60.0)
                                completed_results.extend(remaining_results)
                                # Add remaining indices
                                for i, fut in enumerate(future_list):
                                    if fut in remaining:
                                        completed_indices.append(indices[i])
                            except Exception as e:
                                logger.error(f"Error getting remaining video results: {e}", exc_info=True)
                        break
                
                # Match results with indices
                for result, idx in zip(completed_results, completed_indices):
                    video_metadata.append(
                        {
                            **result,
                            "trajectory_id": idx,
                        }
                    )
        
        # Cleanup actors - use Ray's built-in cleanup
        for actor in actors:
            try:
                ray.kill(actor, no_restart=True)
            except (ValueError, ray.exceptions.RayActorError):
                pass

        logger.info(f"Generated {len(video_metadata)} videos")

        # Return dataset with video metadata
        return ray.data.from_items(video_metadata)

    def generate_single(
        self,
        trajectory: Dict[str, Any],
        output_path: str,
        show: bool = False,
    ) -> Dict[str, Any]:
        """Generate a single video for demo purposes.

        Args:
            trajectory: Trajectory data
            output_path: Output video path
            show: Whether to display video (for demos)

        Returns:
            Metadata about generated video
        """
        renderer = SimulationRenderer(
            mode="interactive" if show else "headless",
            resolution=self.resolution,
            fps=self.fps,
            enable_gui=show,
        )

        frames = renderer.render_trajectory(
            trajectory,
            output_path=output_path,
            show=show,
        )

        return {
            "success": True,
            "output_path": output_path,
            "num_frames": len(frames),
            "duration": len(frames) / self.fps,
        }
