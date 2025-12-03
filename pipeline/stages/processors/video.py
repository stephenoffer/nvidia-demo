"""Video processing stage for multimodal pipeline.

Uses OpenCV for video processing and Ray Data map_batches for efficient
distributed execution. For GPU-intensive video processing, uses actor pools.
"""

import logging
import os
import tempfile
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
import ray
from ray.data import Dataset
from ray.data import ActorPoolStrategy

from pipeline.stages.base import ProcessorBase
from pipeline.utils.constants import (
    _VIDEO_BATCH_SIZE,
    _NORMALIZATION_DIVISOR,
    _DEFAULT_CPUS_PER_GPU,
)
from pipeline.utils.data.data_types import get_data_type, DataType

logger = logging.getLogger(__name__)

# Fractional GPU allocation may not work correctly
# Changed to integer allocation for reliability
_ACTOR_GPUS = 1
_ACTOR_CPUS = 2


# Fractional GPU allocation (0.25) may not work correctly in all Ray versions
# Ray's fractional GPU support is experimental and may cause resource allocation issues
# Use integer GPU allocation (1) or verify fractional GPU support in your Ray version
@ray.remote(num_gpus=1, num_cpus=2, memory=4 * 1024 * 1024 * 1024)  # 4GB memory limit
class VideoProcessorActor:
    """Ray actor for video processing."""

    def __init__(self, extract_frames: bool, frame_rate: int, resolution: tuple):
        """Initialize video processor actor.

        Args:
            extract_frames: Whether to extract frames
            frame_rate: Target frame rate
            resolution: Target resolution (width, height)
        """
        self.extract_frames = extract_frames
        self.frame_rate = frame_rate
        self.resolution = resolution

    def process_video(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single video item.

        Args:
            item: Video data item

        Returns:
            Processed video item
        """
        video_path = None
        temp_file_created = False
        try:
            if "path" in item:
                video_path = item["path"]
            elif "bytes" in item:
                temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
                temp_file.write(item["bytes"])
                temp_file.close()
                video_path = temp_file.name
                temp_file_created = True
            else:
                return item

            result = {
                **item,
                "processed": True,
            }

            if self.extract_frames:
                frames = self._extract_frames(video_path)
                result["frames"] = frames
                result["num_frames"] = len(frames)

            return result

        except (OSError, IOError, cv2.error, ValueError) as e:
            logger.error(f"Error processing video: {e}", exc_info=True)
            return {**item, "processed": False, "error": str(e)}
        finally:
            if temp_file_created and video_path and os.path.exists(video_path):
                try:
                    os.unlink(video_path)
                except OSError:
                    pass

    def _extract_frames(self, video_path: str) -> list:
        """Extract frames from video.

        Args:
            video_path: Path to video file

        Returns:
            List of extracted frames
        """
        cap = None
        try:
            cap = cv2.VideoCapture(video_path)
            frames = []
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_interval = max(1, int(fps / self.frame_rate))
            frame_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % frame_interval == 0:
                    if self.resolution:
                        frame = cv2.resize(frame, self.resolution)

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Use GPU-accelerated normalization when available
                    try:
                        import cupy as cp
                        frame_cp = cp.asarray(frame_rgb)
                        frame_normalized = (frame_cp.astype(cp.float32) / float(_NORMALIZATION_DIVISOR)).get()
                        frames.append(frame_normalized.tolist())
                    except ImportError:
                        # Fallback to NumPy
                        frame_normalized = frame_rgb.astype(np.float32) / float(_NORMALIZATION_DIVISOR)
                        frames.append(frame_normalized.tolist())

                frame_count += 1

            return frames
        finally:
            if cap is not None:
                cap.release()


class VideoProcessor(ProcessorBase):
    """Video processing stage for the pipeline.

    Handles video decoding, frame extraction, and temporal segmentation.
    """

    def __init__(
        self,
        extract_frames: bool = True,
        frame_rate: int = 30,
        resolution: tuple = (224, 224),
        max_duration: float = 60.0,
        batch_size: int = _VIDEO_BATCH_SIZE,
        config: Any = None,  # PipelineConfig for accessing num_gpus
    ):
        """Initialize video processor.

        Args:
            extract_frames: Whether to extract frames
            frame_rate: Target frame rate for extraction
            resolution: Target resolution (width, height)
            max_duration: Maximum video duration in seconds
            batch_size: Batch size for processing
        """
        super().__init__(batch_size=batch_size)
        self.extract_frames = extract_frames
        self.frame_rate = frame_rate
        self.resolution = resolution
        self.max_duration = max_duration
        self.config = config  # Store config for actor pool sizing

    def process(self, dataset: Dataset) -> Dataset:
        """Process video data in the dataset.

        Uses Ray Data map_batches with actor pool for GPU-accelerated processing.

        Args:
            dataset: Input Ray Dataset

        Returns:
            Processed Ray Dataset
        """
        logger.info("Processing video data")

        # Use named function instead of lambda for better serialization
        def is_video_type(item: dict[str, Any]) -> bool:
            """Check if item is video type."""
            return get_data_type(item) == DataType.VIDEO
        
        video_dataset = dataset.filter(is_video_type)

        if not self.extract_frames:
            return video_dataset

        # Use ActorPoolStrategy for efficient actor reuse
        # Creates a pool of actors that are reused across batches
        # Calculate pool size based on available GPUs
        # Calculate pool size based on integer GPU allocation
        # Each actor uses 1 GPU, so pool size = num_gpus
        if self.config and hasattr(self.config, "num_gpus") and self.config.num_gpus > 0:
            pool_size = min(4, max(1, self.config.num_gpus))  # Each actor uses 1 GPU
            # Use ActorPoolStrategy for GPU processing
            processed = video_dataset.map_batches(
                VideoBatchProcessor(),  # Use module-level callable class
                batch_size=self.batch_size,
                batch_format="pandas",
                compute=ActorPoolStrategy(size=pool_size),
                ray_remote_args={
                    "num_gpus": _ACTOR_GPUS,
                    "num_cpus": _ACTOR_CPUS,
                    "memory": 4 * 1024 * 1024 * 1024,  # 4GB memory limit
                },
            )
        else:
            # CPU-only mode: Use default compute strategy (tasks, not actors)
            # ActorPoolStrategy requires GPU actors, so fall back to regular tasks
            logger.info("CPU-only mode: Using task-based processing instead of actor pool")
            processed = video_dataset.map_batches(
                self._process_batch_cpu,  # Use regular function for CPU
                batch_size=self.batch_size,
                batch_format="pandas",
                # No compute strategy = uses default task-based processing
            )

        return processed

    def _process_batch_cpu(self, batch: pd.DataFrame) -> pd.DataFrame:
        """Process video batch in CPU-only mode (without actors).
        
        Args:
            batch: Batch of video items as pandas DataFrame
            
        Returns:
            Processed batch
        """
        import cv2
        import numpy as np
        
        results = []
        for _, row in batch.iterrows():
            item = row.to_dict()
            video_path = item.get("path")
            
            if not video_path:
                results.append(item)
                continue
            
            try:
                result = {**item, "processed": True}
                
                if self.extract_frames:
                    # Extract frames using OpenCV (CPU)
                    cap = cv2.VideoCapture(video_path)
                    frames = []
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frame_interval = max(1, int(fps / self.frame_rate))
                    frame_count = 0
                    
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        if frame_count % frame_interval == 0:
                            if self.resolution:
                                frame = cv2.resize(frame, self.resolution)
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            frame_normalized = frame_rgb.astype(np.float32) / float(_NORMALIZATION_DIVISOR)
                            frames.append(frame_normalized.tolist())
                        
                        frame_count += 1
                    
                    cap.release()
                    result["frames"] = frames
                    result["num_frames"] = len(frames)
                
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing video {video_path}: {e}")
                results.append({**item, "processed": False, "error": str(e)})
        
        return pd.DataFrame(results)

    def _process_item(self, item: dict[str, Any]) -> Optional[dict[str, Any]]:
        """Process item - not used for video processing.

        Video processing requires actor-based batch processing.
        """
        return None
