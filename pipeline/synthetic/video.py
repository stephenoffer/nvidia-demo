"""Synthetic video generator for multimodal datasets.

Generates synthetic video sequences that simulate robot egocentric views
or task demonstrations, useful for testing video processing pipelines.

Uses OpenCV for video processing and Ray Data for distributed datasets.
See: https://opencv.org/ and https://docs.ray.io/en/latest/data/data.html
"""

import logging
from typing import Any, Dict, List, Optional

import cv2  # https://opencv.org/
import numpy as np  # https://numpy.org/
import ray
from ray.data import Dataset

logger = logging.getLogger(__name__)


class SyntheticVideoGenerator:
    """Generate synthetic video sequences.

    Creates synthetic videos that simulate robot camera views or
    task demonstrations, useful for testing and demos.
    """

    def __init__(
        self,
        resolution: tuple = (224, 224),
        num_frames: int = 30,
        fps: int = 30,
        video_type: str = "egocentric",
        seed: Optional[int] = None,
    ):
        """Initialize synthetic video generator.

        Args:
            resolution: Video resolution (width, height)
            num_frames: Number of frames per video
            fps: Frames per second
            video_type: Type of video ('egocentric', 'task_demo', 'simulation')
            seed: Random seed for reproducibility
        """
        self.resolution = resolution
        self.num_frames = num_frames
        self.fps = fps
        self.video_type = video_type

        if seed is not None:
            np.random.seed(seed)

    def generate_video(self, video_id: int) -> Dict[str, Any]:
        """Generate a single synthetic video.

        Args:
            video_id: Unique identifier for video

        Returns:
            Dictionary containing video data and metadata
        """
        frames = []

        for frame_idx in range(self.num_frames):
            frame = self._generate_frame(frame_idx, video_id)
            frames.append(frame)

        # Convert frames to video bytes (simplified - in production use proper encoding)
        video_data = {
            "video_id": video_id,
            "frames": frames,
            "num_frames": self.num_frames,
            "fps": self.fps,
            "resolution": self.resolution,
            "video_type": self.video_type,
            "data_type": "video",
            "format": "synthetic",
            "source": "synthetic_generator",
        }

        return video_data

    def _generate_frame(self, frame_idx: int, video_id: int) -> np.ndarray:
        """Generate a single frame.

        Args:
            frame_idx: Frame index
            video_id: Video identifier

        Returns:
            Frame as numpy array
        """
        frame = np.zeros((self.resolution[1], self.resolution[0], 3), dtype=np.uint8)

        if self.video_type == "egocentric":
            # Simulate egocentric robot view
            frame = self._generate_egocentric_frame(frame_idx, video_id)
        elif self.video_type == "task_demo":
            # Simulate task demonstration
            frame = self._generate_task_demo_frame(frame_idx, video_id)
        else:  # simulation
            # Simulate simulation rendering
            frame = self._generate_simulation_frame(frame_idx, video_id)

        return frame

    def _generate_egocentric_frame(self, frame_idx: int, video_id: int) -> np.ndarray:
        """Generate egocentric robot view frame.

        Args:
            frame_idx: Frame index
            video_id: Video identifier

        Returns:
            Frame array
        """
        frame = np.zeros((self.resolution[1], self.resolution[0], 3), dtype=np.uint8)

        # Simulate moving objects/background
        t = frame_idx / self.num_frames

        # Background gradient
        for y in range(self.resolution[1]):
            intensity = int(50 + 30 * np.sin(t * 2 * np.pi + y / 50))
            frame[y, :] = [intensity, intensity, intensity]

        # Add moving objects (simulate robot manipulating objects)
        num_objects = 3
        for i in range(num_objects):
            obj_x = int(self.resolution[0] * (0.2 + i * 0.3 + 0.1 * np.sin(t * 2 * np.pi + i)))
            obj_y = int(self.resolution[1] * (0.3 + 0.2 * np.cos(t * 2 * np.pi + i)))
            obj_size = 20 + int(10 * np.sin(t * 4 * np.pi + i))

            # Draw object
            color = [
                int(100 + 155 * np.sin(i)),
                int(100 + 155 * np.cos(i)),
                int(100 + 155 * np.sin(i + 1)),
            ]
            cv2.circle(frame, (obj_x, obj_y), obj_size, color, -1)

        # Add noise for realism
        noise = np.random.normal(0, 5, frame.shape).astype(np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        return frame

    def _generate_task_demo_frame(self, frame_idx: int, video_id: int) -> np.ndarray:
        """Generate task demonstration frame.

        Args:
            frame_idx: Frame index
            video_id: Video identifier

        Returns:
            Frame array
        """
        frame = np.zeros((self.resolution[1], self.resolution[0], 3), dtype=np.uint8)

        # Simulate task demonstration (e.g., picking up objects)
        t = frame_idx / self.num_frames

        # Background
        frame[:, :] = [240, 240, 240]

        # Table surface
        table_y = int(self.resolution[1] * 0.7)
        cv2.rectangle(
            frame, (0, table_y), (self.resolution[0], self.resolution[1]), (200, 200, 200), -1
        )

        # Objects on table
        num_objects = 2
        for i in range(num_objects):
            obj_x = int(self.resolution[0] * (0.3 + i * 0.4))
            obj_y = table_y - 30

            # Object moves up when being picked
            if t > 0.3 + i * 0.2:
                lift_amount = min(100, int(200 * (t - 0.3 - i * 0.2)))
                obj_y -= lift_amount

            # Draw object
            cv2.circle(frame, (obj_x, obj_y), 25, (100, 150, 200), -1)
            cv2.circle(frame, (obj_x, obj_y), 25, (50, 100, 150), 2)

        return frame

    def _generate_simulation_frame(self, frame_idx: int, video_id: int) -> np.ndarray:
        """Generate simulation-style frame.

        Args:
            frame_idx: Frame index
            video_id: Video identifier

        Returns:
            Frame array
        """
        frame = np.zeros((self.resolution[1], self.resolution[0], 3), dtype=np.uint8)

        # Simulate simulation rendering (simpler, cleaner look)
        t = frame_idx / self.num_frames

        # Background
        frame[:, :] = [50, 50, 50]

        # Add geometric shapes (simulate robot/environment)
        center_x, center_y = self.resolution[0] // 2, self.resolution[1] // 2

        # Rotating object
        angle = t * 2 * np.pi
        radius = 40
        obj_x = int(center_x + radius * np.cos(angle))
        obj_y = int(center_y + radius * np.sin(angle))

        cv2.circle(frame, (obj_x, obj_y), 20, (200, 100, 50), -1)

        # Grid pattern (simulate simulation environment)
        grid_spacing = 30
        for x in range(0, self.resolution[0], grid_spacing):
            cv2.line(frame, (x, 0), (x, self.resolution[1]), (80, 80, 80), 1)
        for y in range(0, self.resolution[1], grid_spacing):
            cv2.line(frame, (0, y), (self.resolution[0], y), (80, 80, 80), 1)

        return frame

    def generate_batch(self, batch_size: int, start_id: int = 0) -> List[Dict[str, Any]]:
        """Generate a batch of videos.

        Args:
            batch_size: Number of videos to generate
            start_id: Starting video ID

        Returns:
            List of video dictionaries
        """
        videos = []
        for i in range(batch_size):
            video = self.generate_video(start_id + i)
            videos.append(video)

        return videos


class SyntheticVideoDataset:
    """Generate synthetic video dataset using Ray Data."""

    def __init__(
        self,
        num_videos: int = 100,
        resolution: tuple = (224, 224),
        num_frames: int = 30,
        fps: int = 30,
        video_type: str = "egocentric",
        num_workers: int = 4,
        seed: Optional[int] = None,
    ):
        """Initialize synthetic video dataset generator.

        Args:
            num_videos: Total number of videos to generate
            resolution: Video resolution
            num_frames: Frames per video
            fps: Frames per second
            video_type: Type of video
            num_workers: Number of parallel workers
            seed: Random seed
        """
        self.num_videos = num_videos
        self.resolution = resolution
        self.num_frames = num_frames
        self.fps = fps
        self.video_type = video_type
        self.num_workers = num_workers
        self.seed = seed

    def generate(self) -> Dataset:
        """Generate synthetic video dataset.

        Returns:
            Ray Dataset containing synthetic videos
        """
        logger.info(f"Generating {self.num_videos} synthetic videos")

        # Create generator
        generator = SyntheticVideoGenerator(
            resolution=self.resolution,
            num_frames=self.num_frames,
            fps=self.fps,
            video_type=self.video_type,
            seed=self.seed,
        )

        # Generate videos in batches
        batch_size = max(1, self.num_videos // self.num_workers)
        batches = []

        for i in range(0, self.num_videos, batch_size):
            current_batch_size = min(batch_size, self.num_videos - i)
            batch = generator.generate_batch(current_batch_size, start_id=i)
            batches.extend(batch)

        # Convert to Ray Dataset using Ray Data's optimized from_items
        # Use parallel generation with Ray Data for better performance
        dataset = ray.data.from_items(batches, parallelism=self.num_workers)
        
        # Log generation info without materializing dataset
        # Use len(batches) instead of dataset.count() to avoid materialization
        logger.info(f"Generated {len(batches)} synthetic video batches")
        
        # Optional: Repartition for optimal block distribution
        if len(batches) > self.num_workers:
            dataset = dataset.repartition(num_blocks=self.num_workers)
            logger.info(f"Repartitioned dataset to {self.num_workers} blocks for optimal parallelism")

        return dataset
