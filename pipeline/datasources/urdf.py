"""Custom datasource for URDF/SDF robot description files.

URDF (Unified Robot Description Format) and SDF (Simulation Description Format)
are XML-based formats for describing robot models and their properties.
Follows Ray Data FileBasedDatasource API.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET  # https://docs.python.org/3/library/xml.etree.elementtree.html
from typing import TYPE_CHECKING, Any, Iterator, Union
from xml.etree.ElementTree import Element

from ray.data._internal.arrow_block import ArrowBlockBuilder
from ray.data.block import Block

from pipeline.datasources.base import FileBasedDatasource
from pipeline.exceptions import DataSourceError
from pipeline.utils.decorators import handle_errors, log_execution_time

if TYPE_CHECKING:
    import pyarrow

logger = logging.getLogger(__name__)

# Constants
_MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10MB


class URDFDatasource(FileBasedDatasource):
    """Custom datasource for reading URDF/SDF robot description files.

    Supports URDF (Unified Robot Description Format) and SDF
    (Simulation Description Format) XML files commonly used in ROS.
    """

    def __init__(
        self,
        paths: Union[str, list[str]],
        parse_geometry: bool = True,
        **kwargs: Any,
    ):
        """Initialize URDF/SDF datasource.

        Args:
            paths: URDF/SDF file path(s) or directory path(s)
            parse_geometry: Whether to parse geometry information
            **kwargs: Additional FileBasedDatasource options
        """
        super().__init__(paths=paths, **kwargs)
        self.parse_geometry = bool(parse_geometry)

    @log_execution_time
    @handle_errors(error_class=DataSourceError)
    def _read_stream(
        self, f: "pyarrow.NativeFile", path: str
    ) -> Iterator[Block]:
        """Read URDF/SDF file and yield robot description blocks.

        Args:
            f: pyarrow.NativeFile handle for the file (supports S3, HDFS, etc.)
            path: Path to URDF/SDF file

        Yields:
            Block objects (pyarrow.Table) with robot description data

        Raises:
            DataSourceError: If reading fails

        Note:
            ET.parse() requires a file path. For cloud storage, we read from
            the file handle and parse from memory.
        """
        self._validate_file_handle(f, path)
        
        try:
            # ET.parse() requires file path, but we can parse from file handle
            # Read content from file handle to support S3/HDFS
            content_bytes = f.readall()
            
            if not content_bytes:
                raise DataSourceError(f"Empty URDF/SDF file: {path}")
            
            # Check file size
            if len(content_bytes) > _MAX_FILE_SIZE_BYTES:
                raise DataSourceError(
                    f"URDF/SDF file {path} is {len(content_bytes)} bytes, "
                    f"exceeds maximum size of {_MAX_FILE_SIZE_BYTES}"
                )
            
            # Decode with error handling
            try:
                content = content_bytes.decode("utf-8")
            except UnicodeDecodeError as e:
                raise DataSourceError(f"URDF/SDF file {path} is not valid UTF-8: {e}") from e
            
            try:
                root = ET.fromstring(content)
            except ET.ParseError as e:
                raise DataSourceError(f"Invalid XML in URDF/SDF file {path}: {e}") from e
            
            if root is None:
                raise DataSourceError(f"Failed to parse XML root from {path}")

            # Determine format (URDF or SDF)
            is_sdf = root.tag == "sdf"
            format_type = "sdf" if is_sdf else "urdf"

            # Parse robot model - return clean data without metadata wrapping
            robot_data: dict[str, Any] = {
                "format": format_type,  # Keep format as it's domain-specific
            }
            
            # Extract name safely
            name_elem = root.find("name")
            robot_data["name"] = name_elem.text if name_elem is not None else None

            # Parse links (URDF) or models (SDF)
            if is_sdf:
                model = root.find("model")
                if model is not None:
                    robot_data["model_name"] = model.get("name")
                    robot_data["links"] = self._parse_sdf_links(model)
                    robot_data["joints"] = self._parse_sdf_joints(model)
                else:
                    robot_data["links"] = []
                    robot_data["joints"] = []
            else:
                robot_data["links"] = self._parse_urdf_links(root)
                robot_data["joints"] = self._parse_urdf_joints(root)

            builder = ArrowBlockBuilder()
            builder.add(robot_data)
            yield builder.build()

        except ET.ParseError as e:
            logger.error(f"XML parse error in URDF/SDF file {path}: {e}", exc_info=True)
            # Re-raise exception instead of wrapping in block
            raise DataSourceError(f"XML parse error in URDF/SDF file {path}: {e}") from e
        except Exception as e:
            logger.error(f"Error reading URDF/SDF file {path}: {e}", exc_info=True)
            # Re-raise exception instead of wrapping in block
            raise DataSourceError(f"Failed to read URDF/SDF file {path}: {e}") from e

    def _parse_urdf_links(self, root: Element) -> list[dict]:
        """Parse URDF link elements.

        Args:
            root: Root XML element

        Returns:
            List of link dictionaries
        """
        links = []
        for link in root.findall("link"):
            if link is None:
                continue
            
            link_data: dict[str, Any] = {"name": link.get("name")}
            if self.parse_geometry:
                visual = link.find("visual/geometry")
                if visual is not None:
                    try:
                        link_data["geometry"] = self._parse_geometry(visual)
                    except Exception as e:
                        logger.warning(f"Failed to parse geometry for link {link.get('name')}: {e}")
                        link_data["geometry_error"] = str(e)
            links.append(link_data)
        return links

    def _parse_urdf_joints(self, root: Element) -> list[dict]:
        """Parse URDF joint elements.

        Args:
            root: Root XML element

        Returns:
            List of joint dictionaries
        """
        joints = []
        for joint in root.findall("joint"):
            if joint is None:
                continue
            
            parent_elem = joint.find("parent")
            child_elem = joint.find("child")
            joint_data: dict[str, Any] = {
                "name": joint.get("name"),
                "type": joint.get("type"),
                "parent": parent_elem.get("link") if parent_elem is not None else None,
                "child": child_elem.get("link") if child_elem is not None else None,
            }
            joints.append(joint_data)
        return joints

    def _parse_sdf_links(self, model: Element) -> list[dict]:
        """Parse SDF link elements.

        Args:
            model: Model XML element

        Returns:
            List of link dictionaries
        """
        links = []
        for link in model.findall("link"):
            if link is None:
                continue
            
            link_data: dict[str, Any] = {"name": link.get("name")}
            if self.parse_geometry:
                visual = link.find("visual/geometry")
                if visual is not None:
                    try:
                        link_data["geometry"] = self._parse_geometry(visual)
                    except Exception as e:
                        logger.warning(f"Failed to parse geometry for link {link.get('name')}: {e}")
                        link_data["geometry_error"] = str(e)
            links.append(link_data)
        return links

    def _parse_sdf_joints(self, model: Element) -> list[dict]:
        """Parse SDF joint elements.

        Args:
            model: Model XML element

        Returns:
            List of joint dictionaries
        """
        joints = []
        for joint in model.findall("joint"):
            if joint is None:
                continue
            
            type_elem = joint.find("type")
            parent_elem = joint.find("parent")
            child_elem = joint.find("child")
            joint_data: dict[str, Any] = {
                "name": joint.get("name"),
                "type": type_elem.text if type_elem is not None else None,
                "parent": parent_elem.text if parent_elem is not None else None,
                "child": child_elem.text if child_elem is not None else None,
            }
            joints.append(joint_data)
        return joints

    def _parse_geometry(self, geometry_elem: Element) -> dict:
        """Parse geometry element.

        Args:
            geometry_elem: Geometry XML element

        Returns:
            Geometry dictionary
        """
        if geometry_elem is None:
            return {}
        
        geometry: dict[str, Any] = {}
        for child in geometry_elem:
            if child is None:
                continue
            
            if child.tag == "box":
                size = child.find("size")
                if size is not None and size.text:
                    geometry["type"] = "box"
                    try:
                        geometry["size"] = [float(x) for x in size.text.split()]
                    except ValueError as e:
                        logger.warning(f"Failed to parse box size: {e}")
                        geometry["size"] = size.text.split()
            elif child.tag == "cylinder":
                geometry["type"] = "cylinder"
                radius = child.find("radius")
                length = child.find("length")
                if radius is not None and radius.text:
                    try:
                        geometry["radius"] = float(radius.text)
                    except ValueError:
                        geometry["radius"] = radius.text
                if length is not None and length.text:
                    try:
                        geometry["length"] = float(length.text)
                    except ValueError:
                        geometry["length"] = length.text
            elif child.tag == "sphere":
                geometry["type"] = "sphere"
                radius = child.find("radius")
                if radius is not None and radius.text:
                    try:
                        geometry["radius"] = float(radius.text)
                    except ValueError:
                        geometry["radius"] = radius.text
            elif child.tag == "mesh":
                geometry["type"] = "mesh"
                uri = child.find("uri")
                if uri is not None and uri.text:
                    geometry["uri"] = uri.text
        return geometry


def test() -> None:
    """Test URDF datasource with example data."""
    from pathlib import Path
    
    # Initialize Ray if not already initialized
    try:
        import ray
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
    except Exception:
        pass
    
    # Test data directory
    test_data_dir = Path(__file__).parent.parent.parent / "examples" / "data"
    test_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Create test URDF file
    test_file = test_data_dir / "test_robot.urdf"
    test_urdf_content = """<?xml version="1.0"?>
<robot name="test_robot">
  <link name="base_link">
    <visual>
      <geometry>
        <box size="1 1 1"/>
      </geometry>
    </visual>
  </link>
  <link name="link1">
    <visual>
      <geometry>
        <cylinder radius="0.1" length="0.5"/>
      </geometry>
    </visual>
  </link>
  <joint name="joint1" type="revolute">
    <parent link="base_link"/>
    <child link="link1"/>
  </joint>
</robot>
"""
    
    with open(test_file, "w") as f:
        f.write(test_urdf_content)
    
    logger.info(f"Testing URDFDatasource with {test_file}")
    
    try:
        # Test URDF datasource
        datasource = URDFDatasource(paths=str(test_file))
        dataset = ray.data.read_datasource(datasource)
        
        # Verify dataset
        count = 0
        for batch in dataset.iter_batches(batch_size=10, prefetch_batches=0):
            count += len(batch) if hasattr(batch, '__len__') else 1
        
        logger.info(f"URDFDatasource test passed: loaded {count} items")
        
        # Cleanup
        if test_file.exists():
            test_file.unlink()
    except Exception as e:
        logger.error(f"URDFDatasource test failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    test()
