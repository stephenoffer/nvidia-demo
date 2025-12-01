"""Field extraction utilities for data items.

Provides common patterns for extracting fields from nested data structures.
"""

from __future__ import annotations

from typing import Any


def extract_field(
    item: dict[str, Any],
    field_name: str,
    default: Any = None,
    nested_paths: list[list[str]] | None = None,
) -> Any:
    """Extract field from item with support for nested paths.

    Args:
        item: Data item dictionary
        field_name: Field name to extract
        default: Default value if not found
        nested_paths: List of nested paths to check (e.g., [["sensor_data", "observations"]])

    Returns:
        Field value or default
    """
    if field_name in item:
        return item[field_name]

    if nested_paths:
        for path in nested_paths:
            value = item
            for key in path:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    break
            else:
                return value

    if "." in field_name:
        parts = field_name.split(".")
        value = item
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default
        return value

    if "sensor_data" in item and isinstance(item["sensor_data"], dict):
        if field_name in item["sensor_data"]:
            return item["sensor_data"][field_name]

    if "metadata" in item and isinstance(item["metadata"], dict):
        if field_name in item["metadata"]:
            return item["metadata"][field_name]

    return default


def extract_nested_field(
    item: dict[str, Any],
    path: list[str],
    default: Any = None,
) -> Any:
    """Extract nested field using path list.

    Args:
        item: Data item dictionary
        path: List of keys representing nested path
        default: Default value if not found

    Returns:
        Field value or default
    """
    value = item
    for key in path:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    return value


def has_field(item: dict[str, Any], field_name: str) -> bool:
    """Check if field exists in item.

    Args:
        item: Data item dictionary
        field_name: Field name to check

    Returns:
        True if field exists
    """
    return extract_field(item, field_name) is not None

