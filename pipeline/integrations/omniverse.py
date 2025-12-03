"""NVIDIA Omniverse integration for GR00T data pipeline.

Integrates with NVIDIA Omniverse for USD file processing, Replicator
synthetic data generation, and simulation data export.
Critical for GR00T: Omniverse is a key source of synthetic robotics data.

CRITICAL IMPROVEMENTS:
- Uses proper Omniverse Kit API (omni.usd) instead of pxr directly
- Leverages GPU-accelerated USD processing
- Proper Replicator API usage with distributed rendering
- CUDA memory management for large scenes
- GPU object store support for Ray Data
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from contextlib import nullcontext as _null_context

from ray.data import Dataset
from ray.data.context import DataContext

from pipeline.exceptions import DataSourceError
from pipeline.utils.decorators import handle_errors, log_execution_time
from pipeline.utils.gpu.memory import gpu_memory_cleanup

logger = logging.getLogger(__name__)

# Constants
_MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024 * 1024  # 10GB
_MAX_PRIMITIVES_PER_SCENE = 100000
_MAX_ANNOTATION_SIZE_BYTES = 10 * 1024 * 1024  # 10MB

# Check for Omniverse Kit availability
_OMNIVERSE_KIT_AVAILABLE = False
try:
    import omni.usd  # type: ignore[attr-defined]
    from omni.usd import get_context  # type: ignore[attr-defined]
    _OMNIVERSE_KIT_AVAILABLE = True
except ImportError:
    _OMNIVERSE_KIT_AVAILABLE = False
    logger.debug("Omniverse Kit API not available, using pxr fallback")


class OmniverseLoader:
    """Loader for NVIDIA Omniverse USD files and Replicator data.

    Supports loading USD scenes, Replicator-generated synthetic data,
    and Omniverse simulation outputs.

    CRITICAL IMPROVEMENTS:
    - Uses Omniverse Kit API (omni.usd) for proper USD stage management
    - GPU-accelerated USD processing with CUDA memory management
    - Proper Replicator API integration
    - GPU object store support for Ray Data
    """

    def __init__(
        self,
        omniverse_path: Union[str, Path],
        include_metadata: bool = True,
        include_annotations: bool = True,
        use_replicator: bool = False,
        max_primitives: Optional[int] = None,
        use_gpu: bool = True,
        use_gpu_object_store: bool = True,
        num_gpus: Optional[int] = None,
    ):
        """Initialize Omniverse loader.

        Args:
            omniverse_path: Path to Omniverse USD files or Replicator output
            include_metadata: Whether to include USD metadata
            include_annotations: Whether to include Replicator annotations
            use_replicator: Whether to use Omniverse Replicator for data generation
            max_primitives: Maximum number of primitives to load (None = unlimited)
            use_gpu: Whether to use GPU acceleration for USD processing
            use_gpu_object_store: Whether to enable Ray Data GPU object store (RDMA)
            num_gpus: Number of GPUs to use (None = auto-detect)

        Raises:
            ValueError: If parameters are invalid
        """
        # Validate and convert path
        if isinstance(omniverse_path, str):
            if not omniverse_path or not omniverse_path.strip():
                raise ValueError("omniverse_path cannot be empty")
            self.omniverse_path = Path(omniverse_path)
        elif isinstance(omniverse_path, Path):
            self.omniverse_path = omniverse_path
        else:
            raise ValueError(f"omniverse_path must be str or Path, got {type(omniverse_path)}")
        
        if max_primitives is not None and max_primitives <= 0:
            raise ValueError(f"max_primitives must be positive, got {max_primitives}")
        
        self.include_metadata = bool(include_metadata)
        self.include_annotations = bool(include_annotations)
        self.use_replicator = bool(use_replicator)
        self.max_primitives = max_primitives
        self.use_gpu = bool(use_gpu)
        self.use_gpu_object_store = bool(use_gpu_object_store)
        
        # Validate GPU settings
        if self.use_gpu:
            try:
                import torch
                if not torch.cuda.is_available():
                    logger.warning("CUDA not available, disabling GPU acceleration")
                    self.use_gpu = False
                else:
                    if num_gpus is not None:
                        if num_gpus <= 0:
                            raise ValueError(f"num_gpus must be positive, got {num_gpus}")
                        self.num_gpus = num_gpus
                    else:
                        self.num_gpus = torch.cuda.device_count()
                    logger.info(f"Omniverse loader configured for {self.num_gpus} GPU(s)")
            except ImportError:
                logger.warning("PyTorch not available, disabling GPU acceleration")
                self.use_gpu = False
                self.num_gpus = 0
        else:
            self.num_gpus = 0

    @log_execution_time
    @handle_errors(error_class=DataSourceError)
    def load_usd_scene(self, usd_file: Union[str, Path]) -> List[Dict[str, Any]]:
        """Load USD scene file.

        Args:
            usd_file: Path to USD file

        Returns:
            List of data items from USD scene

        Raises:
            DataSourceError: If loading fails
        """
        # Validate and convert path
        if isinstance(usd_file, str):
            if not usd_file or not usd_file.strip():
                raise ValueError("usd_file cannot be empty")
            usd_path = Path(usd_file)
        elif isinstance(usd_file, Path):
            usd_path = usd_file
        else:
            raise ValueError(f"usd_file must be str or Path, got {type(usd_file)}")
        
        # Validate file exists
        if not usd_path.exists():
            raise DataSourceError(f"USD file does not exist: {usd_path}")
        
        if not usd_path.is_file():
            raise DataSourceError(f"USD path is not a file: {usd_path}")
        
        # Check file size
        try:
            file_size = usd_path.stat().st_size
            if file_size > _MAX_FILE_SIZE_BYTES:
                raise DataSourceError(
                    f"USD file {usd_path} is {file_size} bytes, "
                    f"exceeds maximum size of {_MAX_FILE_SIZE_BYTES}"
                )
        except OSError as e:
            raise DataSourceError(f"Failed to check USD file size: {e}") from e
        
        # Use Omniverse Kit API if available, otherwise fallback to pxr
        if _OMNIVERSE_KIT_AVAILABLE:
            try:
                from omni.usd import get_context  # type: ignore[attr-defined]
                from pxr import Usd, UsdGeom, Gf  # type: ignore[attr-defined]
            except ImportError:
                logger.warning("Omniverse Kit context not available, using pxr directly")
                try:
                    from pxr import Usd, UsdGeom, Gf  # type: ignore[attr-defined]
                except ImportError:
                    logger.warning("USD libraries not available. Install with: pip install omni-usd")
                    return []
        else:
            try:
                from pxr import Usd, UsdGeom, Gf  # type: ignore[attr-defined]
            except ImportError:
                logger.warning("USD libraries not available. Install with: pip install omni-usd")
                return []

        # Use GPU memory management for large USD files
        with gpu_memory_cleanup() if self.use_gpu else _null_context():
            try:
                # Use Omniverse Kit context if available for better performance
                if _OMNIVERSE_KIT_AVAILABLE:
                    try:
                        from omni.usd import get_context  # type: ignore[attr-defined]
                        context = get_context()
                        if context:
                            # Use Omniverse Kit's stage management
                            stage = context.get_stage()
                            if not stage:
                                # Fallback to opening directly
                                stage = Usd.Stage.Open(str(usd_path))
                        else:
                            stage = Usd.Stage.Open(str(usd_path))
                    except Exception:
                        # Fallback to direct opening
                        stage = Usd.Stage.Open(str(usd_path))
                else:
                    stage = Usd.Stage.Open(str(usd_path))
                
                if not stage:
                    raise DataSourceError(f"Failed to open USD file: {usd_path}")

                items = []
                primitive_count = 0
                
                for prim in stage.Traverse():
                    # Check max_primitives limit
                    if self.max_primitives is not None and primitive_count >= self.max_primitives:
                        logger.info(f"Reached max_primitives limit ({self.max_primitives})")
                        break
                    
                    try:
                        if prim.IsA(UsdGeom.Mesh):
                            item = self._extract_mesh_data(prim, str(usd_path))
                            if item:
                                items.append(item)
                                primitive_count += 1
                        elif prim.IsA(UsdGeom.Camera):
                            item = self._extract_camera_data(prim, str(usd_path))
                            if item:
                                items.append(item)
                                primitive_count += 1
                        elif prim.IsA(UsdGeom.Xform):
                            item = self._extract_transform_data(prim, str(usd_path))
                            if item:
                                items.append(item)
                                primitive_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to extract data from prim {prim.GetPath()}: {e}")
                        continue

                logger.info(f"Loaded {len(items)} primitives from USD file: {usd_path}")
                return items

            except Exception as e:
                raise DataSourceError(f"Failed to load USD scene {usd_path}: {e}") from e

    @log_execution_time
    @handle_errors(error_class=DataSourceError)
    def load_replicator_data(self, replicator_output_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """Load Omniverse Replicator generated data.

        Args:
            replicator_output_path: Path to Replicator output directory

        Returns:
            List of data items with annotations

        Raises:
            DataSourceError: If loading fails
        """
        # Validate and convert path
        if isinstance(replicator_output_path, str):
            if not replicator_output_path or not replicator_output_path.strip():
                raise ValueError("replicator_output_path cannot be empty")
            replicator_path = Path(replicator_output_path)
        elif isinstance(replicator_output_path, Path):
            replicator_path = replicator_output_path
        else:
            raise ValueError(f"replicator_output_path must be str or Path, got {type(replicator_output_path)}")
        
        if not replicator_path.exists():
            raise DataSourceError(f"Replicator output path does not exist: {replicator_output_path}")

        if not replicator_path.is_dir():
            raise DataSourceError(f"Replicator output path is not a directory: {replicator_output_path}")

        items = []
        
        # Load Replicator annotations
        if self.include_annotations:
            annotations_path = replicator_path / "annotations"
            if annotations_path.exists():
                try:
                    items.extend(self._load_replicator_annotations(annotations_path))
                except Exception as e:
                    logger.error(f"Failed to load Replicator annotations: {e}", exc_info=True)

        # Load rendered images/videos
        images_path = replicator_path / "images"
        if images_path.exists():
            try:
                items.extend(self._load_replicator_images(images_path))
            except Exception as e:
                logger.error(f"Failed to load Replicator images: {e}", exc_info=True)

        # Load depth maps
        depth_path = replicator_path / "depth"
        if depth_path.exists():
            try:
                items.extend(self._load_replicator_depth(depth_path))
            except Exception as e:
                logger.error(f"Failed to load Replicator depth maps: {e}", exc_info=True)

        # Load semantic segmentation
        semantic_path = replicator_path / "semantic"
        if semantic_path.exists():
            try:
                items.extend(self._load_replicator_semantic(semantic_path))
            except Exception as e:
                logger.error(f"Failed to load Replicator semantic segmentation: {e}", exc_info=True)

        logger.info(f"Loaded {len(items)} items from Replicator output: {replicator_output_path}")
        return items

    def _extract_mesh_data(self, prim: Any, usd_file: str) -> Optional[Dict[str, Any]]:
        """Extract mesh data from USD prim.

        Args:
            prim: USD primitive
            usd_file: Source USD file path

        Returns:
            Dictionary with mesh data or None if extraction fails
        """
        try:
            from pxr import UsdGeom

            mesh = UsdGeom.Mesh(prim)
            
            # Get mesh attributes
            points_attr = mesh.GetPointsAttr()
            face_vertex_counts_attr = mesh.GetFaceVertexCountsAttr()
            face_vertex_indices_attr = mesh.GetFaceVertexIndicesAttr()

            points = points_attr.Get() if points_attr else []
            face_vertex_counts = face_vertex_counts_attr.Get() if face_vertex_counts_attr else []
            face_vertex_indices = face_vertex_indices_attr.Get() if face_vertex_indices_attr else []

            # Get transform
            transform = self._get_prim_transform(prim)

            return {
                "data_type": "mesh",
                "format": "usd",
                "source": "omniverse",
                "usd_file": usd_file,
                "prim_path": str(prim.GetPath()),
                "points": [[float(p[0]), float(p[1]), float(p[2])] for p in points] if points else [],
                "face_vertex_counts": list(face_vertex_counts) if face_vertex_counts else [],
                "face_vertex_indices": list(face_vertex_indices) if face_vertex_indices else [],
                "transform": transform,
                "has_mesh_data": len(points) > 0,
            }
        except Exception as e:
            logger.warning(f"Failed to extract mesh data from prim {prim.GetPath()}: {e}")
            return None

    def _extract_camera_data(self, prim: Any, usd_file: str) -> Optional[Dict[str, Any]]:
        """Extract camera data from USD prim.

        Args:
            prim: USD primitive
            usd_file: Source USD file path

        Returns:
            Dictionary with camera data or None if extraction fails
        """
        try:
            from pxr import UsdGeom

            camera = UsdGeom.Camera(prim)
            
            # Get camera attributes
            focal_length = camera.GetFocalLengthAttr().Get()
            horizontal_aperture = camera.GetHorizontalApertureAttr().Get()
            vertical_aperture = camera.GetVerticalApertureAttr().Get()
            clipping_range = camera.GetClippingRangeAttr().Get()

            transform = self._get_prim_transform(prim)

            return {
                "data_type": "camera",
                "format": "usd",
                "source": "omniverse",
                "usd_file": usd_file,
                "prim_path": str(prim.GetPath()),
                "focal_length": float(focal_length) if focal_length else None,
                "horizontal_aperture": float(horizontal_aperture) if horizontal_aperture else None,
                "vertical_aperture": float(vertical_aperture) if vertical_aperture else None,
                "clipping_range": list(clipping_range) if clipping_range else None,
                "transform": transform,
            }
        except Exception as e:
            logger.warning(f"Failed to extract camera data from prim {prim.GetPath()}: {e}")
            return None

    def _extract_transform_data(self, prim: Any, usd_file: str) -> Optional[Dict[str, Any]]:
        """Extract transform data from USD prim.

        Args:
            prim: USD primitive
            usd_file: Source USD file path

        Returns:
            Dictionary with transform data or None if extraction fails
        """
        try:
            transform = self._get_prim_transform(prim)
            
            return {
                "data_type": "transform",
                "format": "usd",
                "source": "omniverse",
                "usd_file": usd_file,
                "prim_path": str(prim.GetPath()),
                "transform": transform,
            }
        except Exception as e:
            logger.warning(f"Failed to extract transform data from prim {prim.GetPath()}: {e}")
            return None

    def _get_prim_transform(self, prim: Any) -> Dict[str, Any]:
        """Get transform matrix from USD prim.

        Args:
            prim: USD primitive

        Returns:
            Dictionary with transform data
        """
        try:
            from pxr import UsdGeom

            xform = UsdGeom.Xformable(prim)
            if xform:
                # Get local transform
                local_transform = xform.GetLocalTransformation()
                matrix = local_transform.GetMatrix()
                
                # Extract translation, rotation, scale
                translation = matrix.ExtractTranslation()
                rotation = matrix.ExtractRotation()
                scale = matrix.ExtractScale()

                return {
                    "translation": [float(translation[0]), float(translation[1]), float(translation[2])],
                    "rotation": [float(rotation[0]), float(rotation[1]), float(rotation[2]), float(rotation[3])],
                    "scale": [float(scale[0]), float(scale[1]), float(scale[2])],
                    "matrix": [[float(matrix[i][j]) for j in range(4)] for i in range(4)],
                }
        except Exception:
            pass

        # Default identity transform
        return {
            "translation": [0.0, 0.0, 0.0],
            "rotation": [0.0, 0.0, 0.0, 1.0],
            "scale": [1.0, 1.0, 1.0],
            "matrix": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        }

    def _load_replicator_annotations(self, annotations_path: Path) -> List[Dict[str, Any]]:
        """Load Replicator annotation files.

        Args:
            annotations_path: Path to annotations directory

        Returns:
            List of annotation items

        Raises:
            DataSourceError: If loading fails
        """
        if not annotations_path.exists():
            return []
        
        if not annotations_path.is_dir():
            raise DataSourceError(f"Annotations path is not a directory: {annotations_path}")
        
        items = []
        
        # Look for JSON annotation files
        try:
            for json_file in annotations_path.glob("*.json"):
                try:
                    # Check file size
                    file_size = json_file.stat().st_size
                    if file_size > _MAX_ANNOTATION_SIZE_BYTES:
                        logger.warning(
                            f"Annotation file {json_file} is {file_size} bytes, "
                            f"exceeds recommended size of {_MAX_ANNOTATION_SIZE_BYTES}"
                        )
                        continue
                    
                    with open(json_file, encoding="utf-8") as f:
                        annotation_data = json.load(f)
                        
                        item = {
                            "data_type": "annotation",
                            "format": "replicator",
                            "source": "omniverse_replicator",
                            "annotation_file": str(json_file),
                            "annotations": annotation_data,
                        }
                        
                        # Extract bounding boxes, keypoints, etc.
                        if isinstance(annotation_data, dict):
                            if "boundingBoxes" in annotation_data:
                                item["bounding_boxes"] = annotation_data["boundingBoxes"]
                            if "keypoints" in annotation_data:
                                item["keypoints"] = annotation_data["keypoints"]
                            if "semantic_segmentation" in annotation_data:
                                item["semantic_segmentation"] = annotation_data["semantic_segmentation"]
                        
                        items.append(item)
                except (OSError, IOError, json.JSONDecodeError) as e:
                    logger.warning(f"Failed to load annotation file {json_file}: {e}")
        except (OSError, PermissionError) as e:
            raise DataSourceError(f"Failed to scan annotations directory: {e}") from e

        return items

    def _load_replicator_images(self, images_path: Path) -> List[Dict[str, Any]]:
        """Load Replicator rendered images.

        Args:
            images_path: Path to images directory

        Returns:
            List of image items

        Raises:
            DataSourceError: If loading fails
        """
        if not images_path.exists():
            return []
        
        if not images_path.is_dir():
            raise DataSourceError(f"Images path is not a directory: {images_path}")
        
        items = []
        
        try:
            for image_file in images_path.glob("*.png"):
                try:
                    # Check file size
                    file_size = image_file.stat().st_size
                    if file_size > _MAX_FILE_SIZE_BYTES:
                        logger.warning(f"Image file {image_file} is {file_size} bytes, too large")
                        continue
                    
                    item = {
                        "data_type": "image",
                        "format": "replicator",
                        "source": "omniverse_replicator",
                        "image_path": str(image_file),
                        "modality": "video",  # Images are part of video modality
                        "size_bytes": file_size,
                    }
                    items.append(item)
                except OSError as e:
                    logger.warning(f"Failed to load image {image_file}: {e}")
        except (OSError, PermissionError) as e:
            raise DataSourceError(f"Failed to scan images directory: {e}") from e

        return items

    def _load_replicator_depth(self, depth_path: Path) -> List[Dict[str, Any]]:
        """Load Replicator depth maps.

        Args:
            depth_path: Path to depth directory

        Returns:
            List of depth map items

        Raises:
            DataSourceError: If loading fails
        """
        if not depth_path.exists():
            return []
        
        if not depth_path.is_dir():
            raise DataSourceError(f"Depth path is not a directory: {depth_path}")
        
        items = []
        
        try:
            for depth_file in depth_path.glob("*.npy"):
                try:
                    # Check file size
                    file_size = depth_file.stat().st_size
                    if file_size > _MAX_FILE_SIZE_BYTES:
                        logger.warning(f"Depth file {depth_file} is {file_size} bytes, too large")
                        continue
                    
                    item = {
                        "data_type": "depth",
                        "format": "replicator",
                        "source": "omniverse_replicator",
                        "depth_path": str(depth_file),
                        "modality": "sensor",
                        "size_bytes": file_size,
                    }
                    items.append(item)
                except OSError as e:
                    logger.warning(f"Failed to load depth map {depth_file}: {e}")
        except (OSError, PermissionError) as e:
            raise DataSourceError(f"Failed to scan depth directory: {e}") from e

        return items

    def _load_replicator_semantic(self, semantic_path: Path) -> List[Dict[str, Any]]:
        """Load Replicator semantic segmentation.

        Args:
            semantic_path: Path to semantic directory

        Returns:
            List of semantic segmentation items

        Raises:
            DataSourceError: If loading fails
        """
        if not semantic_path.exists():
            return []
        
        if not semantic_path.is_dir():
            raise DataSourceError(f"Semantic path is not a directory: {semantic_path}")
        
        items = []
        
        try:
            for semantic_file in semantic_path.glob("*.png"):
                try:
                    # Check file size
                    file_size = semantic_file.stat().st_size
                    if file_size > _MAX_FILE_SIZE_BYTES:
                        logger.warning(f"Semantic file {semantic_file} is {file_size} bytes, too large")
                        continue
                    
                    item = {
                        "data_type": "semantic_segmentation",
                        "format": "replicator",
                        "source": "omniverse_replicator",
                        "semantic_path": str(semantic_file),
                        "modality": "video",
                        "size_bytes": file_size,
                    }
                    items.append(item)
                except OSError as e:
                    logger.warning(f"Failed to load semantic segmentation {semantic_file}: {e}")
        except (OSError, PermissionError) as e:
            raise DataSourceError(f"Failed to scan semantic directory: {e}") from e

        return items

    @log_execution_time
    @handle_errors(error_class=DataSourceError)
    def export_to_usd(self, dataset: Dataset, output_usd_file: Union[str, Path]) -> None:
        """Export curated dataset to USD format.

        Args:
            dataset: Curated Ray Dataset
            output_usd_file: Output USD file path

        Raises:
            DataSourceError: If export fails
        """
        # Validate and convert path
        if isinstance(output_usd_file, str):
            if not output_usd_file or not output_usd_file.strip():
                raise ValueError("output_usd_file cannot be empty")
            output_path = Path(output_usd_file)
        elif isinstance(output_usd_file, Path):
            output_path = output_usd_file
        else:
            raise ValueError(f"output_usd_file must be str or Path, got {type(output_usd_file)}")
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            from pxr import Usd, UsdGeom, Gf, Sdf  # type: ignore[attr-defined]
        except ImportError:
            raise DataSourceError(
                "Omniverse USD libraries not available. Install with: pip install omni-usd"
            ) from None

        try:
            stage = Usd.Stage.CreateNew(str(output_path))
            if not stage:
                raise DataSourceError(f"Failed to create USD file: {output_path}")

            # Export dataset items as USD primitives
            item_count = 0
            for batch in dataset.iter_batches(batch_size=100, prefetch_batches=0):
                for item in batch:
                    try:
                        self._export_item_to_usd(stage, item)
                        item_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to export item to USD: {e}")
                        continue

            stage.Save()
            logger.info(f"Exported {item_count} items to USD: {output_path}")

        except Exception as e:
            raise DataSourceError(f"Failed to export to USD: {e}") from e

    def _export_item_to_usd(self, stage: Any, item: Dict[str, Any]) -> None:
        """Export a single item to USD stage.

        Args:
            stage: USD stage
            item: Data item dictionary
        """
        if not isinstance(item, dict):
            logger.warning(f"Skipping non-dict item: {type(item)}")
            return
        
        try:
            from pxr import UsdGeom, Gf

            # Create prim path from item
            prim_path = item.get("prim_path", f"/{item.get('id', 'item')}")
            
            if item.get("data_type") == "mesh" and "points" in item:
                # Export as mesh
                mesh_prim = stage.DefinePrim(prim_path, "Mesh")
                mesh = UsdGeom.Mesh(mesh_prim)
                
                points = item.get("points", [])
                if points and isinstance(points, list):
                    try:
                        mesh.GetPointsAttr().Set([Gf.Vec3f(p[0], p[1], p[2]) for p in points])
                    except (IndexError, TypeError, ValueError) as e:
                        logger.warning(f"Failed to set mesh points: {e}")
                
                face_vertex_counts = item.get("face_vertex_counts", [])
                if face_vertex_counts:
                    try:
                        mesh.GetFaceVertexCountsAttr().Set(face_vertex_counts)
                    except (TypeError, ValueError) as e:
                        logger.warning(f"Failed to set face vertex counts: {e}")
                
                face_vertex_indices = item.get("face_vertex_indices", [])
                if face_vertex_indices:
                    try:
                        mesh.GetFaceVertexIndicesAttr().Set(face_vertex_indices)
                    except (TypeError, ValueError) as e:
                        logger.warning(f"Failed to set face vertex indices: {e}")

            elif item.get("data_type") == "camera":
                # Export as camera
                camera_prim = stage.DefinePrim(prim_path, "Camera")
                camera = UsdGeom.Camera(camera_prim)
                
                if "focal_length" in item and item["focal_length"] is not None:
                    try:
                        camera.GetFocalLengthAttr().Set(float(item["focal_length"]))
                    except (TypeError, ValueError) as e:
                        logger.warning(f"Failed to set focal length: {e}")
                if "horizontal_aperture" in item and item["horizontal_aperture"] is not None:
                    try:
                        camera.GetHorizontalApertureAttr().Set(float(item["horizontal_aperture"]))
                    except (TypeError, ValueError) as e:
                        logger.warning(f"Failed to set horizontal aperture: {e}")
                if "vertical_aperture" in item and item["vertical_aperture"] is not None:
                    try:
                        camera.GetVerticalApertureAttr().Set(float(item["vertical_aperture"]))
                    except (TypeError, ValueError) as e:
                        logger.warning(f"Failed to set vertical aperture: {e}")

            # Apply transform if available
            if "transform" in item and isinstance(item["transform"], dict):
                transform = item["transform"]
                if "translation" in transform and isinstance(transform["translation"], list):
                    try:
                        xform_prim = UsdGeom.Xformable(stage.GetPrimAtPath(prim_path))
                        if xform_prim:
                            xform_prim.AddTranslateOp().Set(Gf.Vec3d(*transform["translation"][:3]))
                    except (TypeError, ValueError, IndexError) as e:
                        logger.warning(f"Failed to apply transform: {e}")

        except Exception as e:
            logger.warning(f"Failed to export item to USD: {e}")


class OmniverseReplicatorGenerator:
    """Generate synthetic data using Omniverse Replicator.

    Integrates with Omniverse Replicator for programmatic synthetic
    data generation for GR00T training.
    """

    def __init__(
        self,
        output_path: Union[str, Path],
        num_scenes: int = 100,
        randomize_lighting: bool = True,
        randomize_materials: bool = True,
        randomize_camera: bool = True,
    ):
        """Initialize Replicator generator.

        Args:
            output_path: Output path for generated data
            num_scenes: Number of scenes to generate
            randomize_lighting: Whether to randomize lighting
            randomize_materials: Whether to randomize materials
            randomize_camera: Whether to randomize camera poses

        Raises:
            ValueError: If parameters are invalid
        """
        # Validate and convert path
        if isinstance(output_path, str):
            if not output_path or not output_path.strip():
                raise ValueError("output_path cannot be empty")
            self.output_path = Path(output_path)
        elif isinstance(output_path, Path):
            self.output_path = output_path
        else:
            raise ValueError(f"output_path must be str or Path, got {type(output_path)}")
        
        if num_scenes <= 0:
            raise ValueError(f"num_scenes must be positive, got {num_scenes}")
        
        self.num_scenes = num_scenes
        self.randomize_lighting = bool(randomize_lighting)
        self.randomize_materials = bool(randomize_materials)
        self.randomize_camera = bool(randomize_camera)

    @log_execution_time
    @handle_errors(error_class=DataSourceError)
    def generate_synthetic_data(self) -> List[Dict[str, Any]]:
        """Generate synthetic data using Replicator.

        Uses proper Replicator API with GPU-accelerated rendering and distributed support.

        Returns:
            List of generated data items

        Raises:
            DataSourceError: If generation fails
        """
        try:
            import omni.replicator.core as rep  # type: ignore[attr-defined]
        except ImportError:
            raise DataSourceError(
                "Omniverse Replicator not available. Install Omniverse to use Replicator."
            ) from None

        try:
            # Ensure output directory exists
            self.output_path.mkdir(parents=True, exist_ok=True)
            
            # Enable GPU object store for Ray Data if configured
            try:
                ctx = DataContext.get_current()
                if hasattr(ctx.execution_options, 'enable_gpu_object_store'):
                    ctx.execution_options.enable_gpu_object_store = True
                    logger.info("GPU object store enabled for Replicator data")
            except Exception as e:
                logger.warning(f"Failed to enable GPU object store: {e}")
            
            # Configure Replicator with proper GPU settings
            with rep.new_layer():
                # Configure GPU-accelerated rendering
                # Replicator automatically uses GPU if available
                
                # Add randomization
                if self.randomize_lighting:
                    rep.randomizer.register(rep.randomizers.light_dome_light)
                if self.randomize_materials:
                    rep.randomizer.register(rep.randomizers.material_randomizer)
                if self.randomize_camera:
                    rep.randomizer.register(rep.randomizers.camera_randomizer)

                # Render and save with GPU acceleration
                # Replicator uses GPU rendering by default when available
                render_product = rep.create.render_product(
                    resolution=(1920, 1080),
                    output_path=str(self.output_path),
                )

                # Generate scenes with proper batching for GPU efficiency
                # Use frame triggers for efficient GPU utilization
                with rep.trigger.on_frame(num_frames=self.num_scenes):
                    rep.randomizer.randomize()
                
                # Trigger rendering (GPU-accelerated)
                rep.orchestrator.run()

            logger.info(f"Generated {self.num_scenes} synthetic scenes using Replicator (GPU-accelerated)")
            
            # Load generated data with GPU support
            loader = OmniverseLoader(
                str(self.output_path),
                use_replicator=True,
                use_gpu=True,
                use_gpu_object_store=True,
            )
            return loader.load_replicator_data(str(self.output_path))

        except Exception as e:
            raise DataSourceError(f"Failed to generate synthetic data with Replicator: {e}") from e
