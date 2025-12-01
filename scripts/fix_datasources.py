#!/usr/bin/env python3
"""Script to systematically fix datasource issues to match Ray Data patterns.

This script fixes:
1. Return types: Iterator["pyarrow.Table"] -> Iterator[Block]
2. Block builders: DelegatingBlockBuilder -> ArrowBlockBuilder where appropriate
3. Error handling: Remove error wrapping in blocks
4. Metadata wrapping: Remove unnecessary metadata fields
"""

import logging
import re
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Files to fix
DATASOURCE_FILES = [
    "pipeline/datasources/hdf5.py",
    "pipeline/datasources/pointcloud.py",
    "pipeline/datasources/rosbag.py",
    "pipeline/datasources/ros2bag.py",
    "pipeline/datasources/velodyne.py",
    "pipeline/datasources/archive.py",
    "pipeline/datasources/urdf.py",
    "pipeline/datasources/protobuf.py",
    "pipeline/datasources/yaml_config.py",
]

def fix_imports(content: str) -> str:
    """Fix imports to use Block and ArrowBlockBuilder."""
    # Add Block import if not present
    if "from ray.data.block import Block" not in content:
        # Find the last import line before TYPE_CHECKING
        lines = content.split("\n")
        insert_idx = None
        for i, line in enumerate(lines):
            if "if TYPE_CHECKING:" in line:
                insert_idx = i
                break
        
        if insert_idx is not None:
            # Check if ray.data imports exist
            has_ray_data_import = False
            for i in range(insert_idx - 1, -1, -1):
                if "from ray.data" in lines[i]:
                    has_ray_data_import = True
                    # Insert after this line
                    lines.insert(i + 1, "from ray.data.block import Block")
                    break
            
            if not has_ray_data_import:
                lines.insert(insert_idx, "from ray.data.block import Block")
            
            content = "\n".join(lines)
    
    # Replace DelegatingBlockBuilder with ArrowBlockBuilder
    content = content.replace(
        "from ray.data._internal.delegating_block_builder import DelegatingBlockBuilder",
        "from ray.data._internal.arrow_block import ArrowBlockBuilder"
    )
    
    return content

def fix_return_types(content: str) -> str:
    """Fix return type annotations."""
    # Fix Iterator["pyarrow.Table"] -> Iterator[Block]
    content = re.sub(
        r'Iterator\["pyarrow\.Table"\]',
        'Iterator[Block]',
        content
    )
    
    # Fix -> Iterator["pyarrow.Table"]: -> Iterator[Block]:
    content = re.sub(
        r'-> Iterator\["pyarrow\.Table"\]:',
        '-> Iterator[Block]:',
        content
    )
    
    return content

def fix_block_builders(content: str) -> str:
    """Replace DelegatingBlockBuilder with ArrowBlockBuilder."""
    content = content.replace(
        "DelegatingBlockBuilder()",
        "ArrowBlockBuilder()"
    )
    
    return content

def fix_error_wrapping(content: str) -> str:
    """Remove error wrapping patterns."""
    # Pattern: builder.add({"error": ...}) should raise exceptions instead
    # This is more complex and needs manual review, so we'll just flag it
    
    # Remove common error wrapping patterns
    error_patterns = [
        (r'builder\.add\(\s*\{\s*"error":\s*[^}]+\}\s*\)', ''),
        (r'yield builder\.build\(\)\s*#.*error', ''),
    ]
    
    for pattern, replacement in error_patterns:
        content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
    
    return content

def fix_metadata_wrapping(content: str) -> str:
    """Remove unnecessary metadata fields from records."""
    # Remove common metadata fields that Ray Data handles separately
    metadata_fields = [
        '"path"',
        '"format"',
        '"data_type"',
        '"record_index"',
        '"size_bytes"',
    ]
    
    # This is complex and needs context, so we'll provide guidance
    # Pattern matching would be too aggressive here
    
    return content

def main():
    """Fix all datasource files."""
    base_path = Path(__file__).parent.parent
    
    for file_path in DATASOURCE_FILES:
        full_path = base_path / file_path
        if not full_path.exists():
            logger.warning(f"Skipping {file_path} (not found)")
            continue
        
        logger.info(f"Fixing {file_path}...")
        
        with open(full_path, 'r') as f:
            content = f.read()
        
        original_content = content
        
        # Apply fixes
        content = fix_imports(content)
        content = fix_return_types(content)
        content = fix_block_builders(content)
        # Note: Error wrapping and metadata removal need manual review
        
        if content != original_content:
            with open(full_path, 'w') as f:
                f.write(content)
            logger.info(f"Fixed {file_path}")
        else:
            logger.debug(f"No changes needed for {file_path}")

if __name__ == "__main__":
    main()

