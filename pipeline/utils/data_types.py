"""Data type and format definitions.

Provides enums and utilities for consistent data type handling across the pipeline.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional, Dict



class DataType(str, Enum):
    """Data type enumeration."""

    VIDEO = "video"
    TEXT = "text"
    SENSOR = "sensor"
    BINARY = "binary"
    UNKNOWN = "unknown"


class DataFormat(str, Enum):
    """Data format enumeration."""

    VIDEO = "video"
    TEXT = "text"
    SENSOR = "sensor"
    BINARY = "binary"
    PARQUET = "parquet"
    JSON = "json"
    JSONL = "jsonl"
    CSV = "csv"
    HDF5 = "hdf5"
    MCAP = "mcap"
    ROSBAG = "rosbag"
    ROS2BAG = "ros2bag"
    PROTOBUF = "protobuf"
    MSGPACK = "msgpack"
    YAML = "yaml"
    POINTCLOUD = "pointcloud"
    URDF = "urdf"
    SDF = "sdf"
    VELODYNE = "velodyne"
    ARCHIVE = "archive"
    CALIBRATION = "calibration"
    COSMOS_DREAMS = "cosmos_dreams"
    ISAAC_LAB = "isaac_lab"
    ISAAC_SIM = "isaac_sim"
    USD = "usd"
    GR00T = "groot"
    UNKNOWN = "unknown"


class Modality(str, Enum):
    """Modality enumeration for multimodal data."""

    VIDEO = "video"
    TEXT = "text"
    SENSOR = "sensor"
    AUDIO = "audio"


def get_data_type(item: dict[str, Any]) -> DataType:
    """Extract data type from item.

    Args:
        item: Data item

    Returns:
        DataType enum value
    """
    data_type = item.get("data_type")
    if data_type:
        try:
            return DataType(data_type.lower())
        except ValueError:
            pass

    format_val = item.get("format")
    if format_val:
        try:
            format_enum = DataFormat(format_val.lower())
            if format_enum in (DataFormat.VIDEO, DataFormat.COSMOS_DREAMS):
                return DataType.VIDEO
            elif format_enum in (DataFormat.TEXT, DataFormat.JSON, DataFormat.JSONL):
                return DataType.TEXT
            elif format_enum in (DataFormat.SENSOR, DataFormat.ROSBAG, DataFormat.MCAP):
                return DataType.SENSOR
            elif format_enum == DataFormat.BINARY:
                return DataType.BINARY
        except ValueError:
            pass

    if "frames" in item or "video" in item:
        return DataType.VIDEO
    if "text" in item or "content" in item:
        return DataType.TEXT
    if "sensor_data" in item or "observations" in item or "actions" in item:
        return DataType.SENSOR

    return DataType.UNKNOWN


def get_format(item: dict[str, Any]) -> DataFormat:
    """Extract format from item.

    Args:
        item: Data item

    Returns:
        DataFormat enum value
    """
    format_val = item.get("format")
    if format_val:
        try:
            return DataFormat(format_val.lower())
        except ValueError:
            pass

    data_type = item.get("data_type")
    if data_type:
        try:
            dt = DataType(data_type.lower())
            if dt == DataType.VIDEO:
                return DataFormat.VIDEO
            elif dt == DataType.TEXT:
                return DataFormat.TEXT
            elif dt == DataType.SENSOR:
                return DataFormat.SENSOR
            elif dt == DataType.BINARY:
                return DataFormat.BINARY
        except ValueError:
            pass

    return DataFormat.UNKNOWN


def extract_text(item: dict[str, Any]) -> Optional[str]:
    """Extract text content from item.

    Args:
        item: Data item

    Returns:
        Text content or None
    """
    return item.get("text") or item.get("content")


def extract_sensor_data(item: dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract sensor data from item.

    Args:
        item: Data item

    Returns:
        Sensor data dictionary or None
    """
    sensor_data = item.get("sensor_data")
    if isinstance(sensor_data, dict):
        return sensor_data
    if "observations" in item or "actions" in item:
        return {
            "observations": item.get("observations"),
            "actions": item.get("actions"),
        }
    return None


def detect_modalities(item: dict[str, Any]) -> list[Modality]:
    """Detect available modalities in item.

    Args:
        item: Data item

    Returns:
        List of detected modalities
    """
    modalities = []
    data_type = get_data_type(item)

    if data_type == DataType.VIDEO or "frames" in item or "video" in item:
        modalities.append(Modality.VIDEO)

    if data_type == DataType.SENSOR or "sensor_data" in item or "observations" in item:
        modalities.append(Modality.SENSOR)

    if data_type == DataType.TEXT or "text" in item or "content" in item:
        modalities.append(Modality.TEXT)

    return modalities

