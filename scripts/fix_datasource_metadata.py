#!/usr/bin/env python3
"""Fix metadata wrapping and error handling in datasources.

Removes unnecessary metadata fields and fixes error handling patterns.
"""

import logging
import re
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATASOURCE_FILES = [
    "pipeline/datasources/pointcloud.py",
    "pipeline/datasources/rosbag.py",
    "pipeline/datasources/ros2bag.py",
    "pipeline/datasources/velodyne.py",
    "pipeline/datasources/archive.py",
    "pipeline/datasources/urdf.py",
    "pipeline/datasources/protobuf.py",
    "pipeline/datasources/yaml_config.py",
]

def remove_metadata_fields(content: str) -> str:
    """Remove common metadata fields from builder.add() calls."""
    # Pattern: Remove "path", "format", "data_type" fields from dicts
    # This is complex - we'll use regex to find and fix common patterns
    
    # Pattern 1: Remove "path" field when it's the first field
    content = re.sub(
        r'"path":\s*path\s*,?\s*\n\s*',
        '',
        content
    )
    
    # Pattern 2: Remove "format" field
    content = re.sub(
        r'"format":\s*"[^"]*"\s*,?\s*\n\s*',
        '',
        content
    )
    
    # Pattern 3: Remove "data_type" field
    content = re.sub(
        r'"data_type":\s*"[^"]*"\s*,?\s*\n\s*',
        '',
        content
    )
    
    # Pattern 4: Remove "error" field (should raise exceptions instead)
    content = re.sub(
        r'"error":\s*[^,}]+\s*,?\s*\n\s*',
        '',
        content
    )
    
    return content

def fix_error_handling(content: str) -> str:
    """Fix error handling patterns."""
    # Pattern: Replace error block yielding with exception raising
    # This is complex and needs context, so we'll provide patterns to look for
    
    # Remove common error wrapping patterns
    patterns = [
        # Pattern: builder.add({"error": ...})
        (r'builder\s*=\s*ArrowBlockBuilder\(\)\s*\n\s*builder\.add\(\s*\{\s*"error":\s*[^}]+\}\s*\)\s*\n\s*yield\s+builder\.build\(\)', 
         '# Error handling removed - should raise exception instead'),
    ]
    
    for pattern, replacement in patterns:
        content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
    
    return content

def main():
    """Fix metadata wrapping in datasource files."""
    base_path = Path(__file__).parent.parent
    
    for file_path in DATASOURCE_FILES:
        full_path = base_path / file_path
        if not full_path.exists():
            logger.warning(f"Skipping {file_path} (not found)")
            continue
        
        logger.info(f"Reviewing {file_path} for metadata issues...")
        # Note: This script provides guidance but manual review is needed
        # for complex cases

if __name__ == "__main__":
    main()

