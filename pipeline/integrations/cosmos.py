"""Integration with NVIDIA Cosmos for video world models (GR00T Dreams).

Cosmos is NVIDIA's video foundation model framework used for GR00T Dreams.
Uses Ray Data for distributed dataset processing.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import ray
from ray.data import Dataset

from pipeline.exceptions import DataSourceError
from pipeline.utils.decorators import handle_errors, log_execution_time
from pipeline.utils.constants import _SYNTHETIC_BATCH_SIZE

logger = logging.getLogger(__name__)

# Constants
_MAX_FILES_TO_SCAN = 10000
_MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024 * 1024  # 10GB
_DEFAULT_QUALITY_SCORE = 0.5


class CosmosDreamsLoader:
    """Loader for Cosmos video world model outputs (GR00T Dreams)."""

    SUPPORTED_EXTENSIONS = (".mp4", ".avi", ".mov")

    def __init__(
        self,
        dreams_path: Union[str, Path],
        model_name: str = "groot-dreams-v1",
        include_metadata: bool = True,
        batch_size: Optional[int] = None,
        support_video_sequences: bool = True,
        compute_quality_scores: bool = False,
        max_files: Optional[int] = None,
    ):
        """Initialize Cosmos Dreams loader.

        Args:
            dreams_path: Path or URI prefix to Cosmos Dreams output assets
            model_name: Name of the video world model used
            include_metadata: Whether to include generation metadata (JSON)
            batch_size: Batch size for Ray Data map_batches operations
            support_video_sequences: Whether to support video sequence ordering
            compute_quality_scores: Whether to compute quality scores (FVD, IS)
            max_files: Maximum number of files to process (None = unlimited)

        Raises:
            ValueError: If parameters are invalid
        """
        # Validate and convert path
        if isinstance(dreams_path, str):
            if not dreams_path or not dreams_path.strip():
                raise ValueError("dreams_path cannot be empty")
            self._dreams_path_str = dreams_path
            self._base_path = Path(dreams_path) if self._is_local_path(dreams_path) else None
        elif isinstance(dreams_path, Path):
            self._dreams_path_str = str(dreams_path)
            self._base_path = dreams_path if self._is_local_path(str(dreams_path)) else None
        else:
            raise ValueError(f"dreams_path must be str or Path, got {type(dreams_path)}")
        
        # Validate parameters
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError(f"model_name must be non-empty str, got {type(model_name)}")
        
        if batch_size is not None and batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        
        if max_files is not None and max_files <= 0:
            raise ValueError(f"max_files must be positive, got {max_files}")
        
        self.model_name = model_name
        self.include_metadata = bool(include_metadata)
        self.batch_size = batch_size if batch_size is not None else _SYNTHETIC_BATCH_SIZE
        self.support_video_sequences = bool(support_video_sequences)
        self.compute_quality_scores = bool(compute_quality_scores)
        self.max_files = max_files

    @log_execution_time
    @handle_errors(error_class=DataSourceError)
    def load(self) -> Dataset:
        """Load Cosmos Dreams synthetic video data.

        Returns:
            Ray Dataset containing video data

        Raises:
            DataSourceError: If loading fails
        """
        logger.info("Loading Cosmos Dreams data from %s", self._dreams_path_str)
        
        # Validate path exists (for local paths)
        if self._base_path and not self._base_path.exists():
            raise DataSourceError(f"Cosmos Dreams path does not exist: {self._base_path}")
        
        dataset = self._build_dataset()

        if dataset is None:
            logger.warning("Cosmos Dreams loader produced no dataset")
            return ray.data.from_items([])

        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

        return dataset.map_batches(
            self._format_batch,
            batch_size=self.batch_size,
            batch_format="pandas",  # Specify batch format for consistency
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Collect lightweight statistics for observability dashboards.

        Returns:
            Dictionary of statistics

        Raises:
            DataSourceError: If statistics collection fails
        """
        try:
            dataset = self.load()
            # Avoid materialization - use num_rows if available, otherwise skip count
            try:
                num_videos = dataset.num_rows() if hasattr(dataset, 'num_rows') else None
            except (AttributeError, RuntimeError):
                num_videos = None
            
            # Sample first item without materializing entire dataset
            sample = None
            try:
                for batch in dataset.iter_batches(batch_size=1, prefetch_batches=0):
                    if batch:
                        if hasattr(batch, 'iloc'):
                            sample = batch.iloc[0].to_dict()
                        elif isinstance(batch, list) and len(batch) > 0:
                            sample = batch[0]
                        else:
                            sample = batch
                        break
            except (StopIteration, RuntimeError, AttributeError):
                sample = None
            
            has_metadata = bool(sample and isinstance(sample, dict) and "dream_metadata" in sample)

            return {
                "num_videos": num_videos,
                "model_name": self.model_name,
                "has_metadata": has_metadata,
            }
        except Exception as e:
            raise DataSourceError(f"Failed to get statistics: {e}") from e

    def _build_dataset(self) -> Optional[Dataset]:
        """Build dataset from Cosmos Dreams files.

        Returns:
            Ray Dataset or None if no files found
        """
        if self._base_path:
            files = self._discover_local_video_files()
            if not files:
                return None
            return self._read_local_files(files)

        return self._read_remote_prefix()

    def _discover_local_video_files(self) -> List[Path]:
        """Discover local video files.

        Returns:
            List of video file paths

        Raises:
            DataSourceError: If file discovery fails
        """
        if not self._base_path:
            return []
        
        if not self._base_path.exists():
            raise DataSourceError(f"Cosmos Dreams path does not exist: {self._base_path}")
        
        if not self._base_path.is_dir():
            raise DataSourceError(f"Cosmos Dreams path is not a directory: {self._base_path}")
        
        video_files: List[Path] = []
        try:
            for suffix in self.SUPPORTED_EXTENSIONS:
                if self._base_path:
                    found_files = list(self._base_path.glob(f"**/*{suffix}"))
                    video_files.extend(found_files)
            
            # Apply max_files limit
            if self.max_files is not None:
                video_files = video_files[:self.max_files]
            
            return sorted(video_files)
        except (OSError, PermissionError) as e:
            raise DataSourceError(f"Failed to discover video files: {e}") from e

    def _read_local_files(self, files: Iterable[Path]) -> Dataset:
        """Read local video files.

        Args:
            files: Iterable of file paths

        Returns:
            Ray Dataset

        Raises:
            DataSourceError: If reading fails
        """
        if not files:
            return ray.data.from_items([])
        
        paths = []
        for path in files:
            if not isinstance(path, Path):
                raise ValueError(f"Expected Path, got {type(path)}")
            
            if not path.exists():
                logger.warning(f"Video file does not exist: {path}")
                continue
            
            if not path.is_file():
                logger.warning(f"Path is not a file: {path}")
                continue
            
            # Check file size
            try:
                file_size = path.stat().st_size
                if file_size > _MAX_FILE_SIZE_BYTES:
                    logger.warning(
                        f"Video file {path} is {file_size} bytes, "
                        f"exceeds recommended size of {_MAX_FILE_SIZE_BYTES}"
                    )
            except OSError as e:
                logger.warning(f"Failed to get file size for {path}: {e}")
            
            paths.append(str(path))
        
        if not paths:
            return ray.data.from_items([])
        
        try:
            return ray.data.read_binary_files(
                paths=paths,
                include_paths=True,
                file_extensions=list(self.SUPPORTED_EXTENSIONS),
            )
        except Exception as e:
            raise DataSourceError(f"Failed to read local video files: {e}") from e

    def _read_remote_prefix(self) -> Optional[Dataset]:
        """Read remote video files from URI prefix.

        Returns:
            Ray Dataset or None if no files found

        Raises:
            DataSourceError: If reading fails
        """
        try:
            return ray.data.read_binary_files(
                paths=self._dreams_path_str,
                include_paths=True,
                file_extensions=list(self.SUPPORTED_EXTENSIONS),
            )
        except FileNotFoundError:
            logger.warning("No Cosmos Dreams assets found at %s", self._dreams_path_str)
            return None
        except Exception as e:
            raise DataSourceError(f"Failed to read remote video files: {e}") from e

    def _format_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format a batch of Cosmos dream items.

        Args:
            batch: List of raw items

        Returns:
            List of formatted items
        """
        if not batch:
            return []
        
        formatted_items = []
        for item in batch:
            try:
                if not isinstance(item, dict):
                    logger.warning(f"Skipping non-dict item: {type(item)}")
                    continue
                
                formatted_item = self._format_dream(item)
                formatted_items.append(formatted_item)
            except Exception as e:
                logger.warning(f"Failed to format dream item: {e}", exc_info=True)
                # Include item with error flag
                if isinstance(item, dict):
                    item["format_error"] = str(e)
                    formatted_items.append(item)
        
        return formatted_items

    def _format_dream(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Format Cosmos dream item with enhanced metadata and sequence support.

        Args:
            item: Raw dream item

        Returns:
            Formatted dream item

        Raises:
            ValueError: If item is invalid
        """
        if not isinstance(item, dict):
            raise ValueError(f"Item must be a dictionary, got {type(item)}")
        
        formatted = {
            "data_type": "video",
            "format": "cosmos_dreams",
            "source": "synthetic",
            "model_name": self.model_name,
        }

        if "bytes" in item:
            formatted["bytes"] = item["bytes"]

        path = item.get("path")
        if path:
            if not isinstance(path, str):
                logger.warning(f"Invalid path type: {type(path)}")
                path = str(path)
            
            formatted["path"] = path
            metadata = self._load_metadata_for_path(path)
            if metadata:
                formatted["dream_metadata"] = metadata
                formatted.update(self._extract_metadata_fields(metadata))
                
                # Support video sequences
                if self.support_video_sequences:
                    sequence_id = metadata.get("sequence_id") or metadata.get("dream_sequence_id")
                    if sequence_id:
                        formatted["video_sequence_id"] = str(sequence_id)
                        formatted["is_sequence_start"] = metadata.get("sequence_index", 0) == 0
                    
                    # Preserve sequence ordering
                    if "video_sequence_index" in metadata:
                        formatted["sequence_index"] = metadata["video_sequence_index"]
                
                # Compute quality scores if requested
                if self.compute_quality_scores:
                    quality_scores = self._compute_quality_scores(item, metadata)
                    if quality_scores:
                        formatted["cosmos_quality_scores"] = quality_scores
                        formatted["cosmos_quality_score"] = quality_scores.get("overall", _DEFAULT_QUALITY_SCORE)

        return formatted
    
    def _compute_quality_scores(
        self, item: Dict[str, Any], metadata: Dict[str, Any]
    ) -> Dict[str, float]:
        """Compute quality scores for Cosmos-generated videos.
        
        Placeholder for FVD (Fréchet Video Distance) and IS (Inception Score) metrics.
        In production, would use learned models or reference datasets.
        
        Args:
            item: Video item
            metadata: Cosmos metadata
            
        Returns:
            Dictionary with quality scores
        """
        if not isinstance(metadata, dict):
            return {"overall": _DEFAULT_QUALITY_SCORE}
        
        scores: Dict[str, float] = {}
        
        # Would compute FVD and IS in production
        # For now, use metadata-based heuristics
        
        # Check video properties
        if "resolution" in metadata:
            resolution = metadata["resolution"]
            if isinstance(resolution, (list, tuple)) and len(resolution) >= 2:
                width, height = resolution[0], resolution[1]
                if width >= 224 and height >= 224:
                    scores["resolution_score"] = 1.0
                elif width >= 128 and height >= 128:
                    scores["resolution_score"] = 0.7
                else:
                    scores["resolution_score"] = 0.3
            else:
                scores["resolution_score"] = _DEFAULT_QUALITY_SCORE
        
        # Check frame count
        frame_count = metadata.get("frame_count") or metadata.get("num_frames", 0)
        if isinstance(frame_count, (int, float)):
            if frame_count >= 30:
                scores["frame_count_score"] = 1.0
            elif frame_count >= 10:
                scores["frame_count_score"] = 0.7
            else:
                scores["frame_count_score"] = 0.3
        else:
            scores["frame_count_score"] = _DEFAULT_QUALITY_SCORE
        
        # Overall score (weighted average)
        if scores:
            scores["overall"] = sum(scores.values()) / len(scores)
        else:
            scores["overall"] = _DEFAULT_QUALITY_SCORE
        
        return scores

    def _load_metadata_for_path(self, path: str) -> Optional[Dict[str, Any]]:
        """Load metadata JSON file for a video path.

        Args:
            path: Video file path

        Returns:
            Metadata dictionary or None
        """
        if not self.include_metadata or not self._base_path:
            return None
        
        if not path or not isinstance(path, str):
            return None
        
        try:
            metadata_path = Path(path).with_suffix(".json")
            if not metadata_path.exists():
                return None
            
            if not metadata_path.is_file():
                logger.warning(f"Metadata path is not a file: {metadata_path}")
                return None
            
            # Check file size
            try:
                file_size = metadata_path.stat().st_size
                if file_size > 10 * 1024 * 1024:  # 10MB limit for JSON
                    logger.warning(f"Metadata file {metadata_path} is {file_size} bytes, too large")
                    return None
            except OSError:
                pass
            
            with open(metadata_path, encoding="utf-8") as file_obj:
                return json.load(file_obj)
        except (OSError, IOError, json.JSONDecodeError) as e:
            logger.warning(f"Failed to load metadata from {metadata_path}: {e}")
            return None

    @staticmethod
    def _extract_metadata_fields(metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Extract all Cosmos metadata fields for GR00T compatibility.
        
        Includes generation parameters, prompts, quality metrics, and sequence information.
        
        Args:
            metadata: Metadata dictionary

        Returns:
            Extracted metadata fields
        """
        if not isinstance(metadata, dict):
            return {}
        
        result: Dict[str, Any] = {}
        
        # Core metadata fields
        for key in (
            "prompt",
            "initial_frame",
            "generation_params",
            "model_version",
            "timestamp",
            "sequence_id",
            "frame_count",
            "fps",
            "resolution",
            "duration",
            "seed",
            "temperature",
            "top_p",
            "num_frames",
            "video_sequence_index",
            "dream_sequence_id",
        ):
            if key in metadata:
                result[key] = metadata[key]
        
        # Nested generation parameters
        if "generation_params" in metadata and isinstance(metadata["generation_params"], dict):
            gen_params = metadata["generation_params"]
            for key in (
                "num_inference_steps",
                "guidance_scale",
                "noise_schedule",
                "motion_scale",
                "camera_motion",
            ):
                if key in gen_params:
                    result[f"gen_{key}"] = gen_params[key]
        
        # Quality metrics if available
        if "quality_metrics" in metadata:
            result["quality_metrics"] = metadata["quality_metrics"]
        
        return result

    @staticmethod
    def _is_local_path(path: str) -> bool:
        """Check if path is local (not a URI).

        Args:
            path: Path string

        Returns:
            True if local path, False if URI
        """
        if not path or not isinstance(path, str):
            return False
        return "://" not in path
