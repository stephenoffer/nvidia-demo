#!/usr/bin/env python3
"""Script to help refactor dataframe.py into smaller modules.

This script analyzes the current dataframe.py and helps extract methods
into appropriate modules based on their functionality.
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple

def analyze_dataframe_file(file_path: str) -> Dict[str, List[Tuple[int, str, str]]]:
    """Analyze dataframe.py and categorize methods.
    
    Returns:
        Dictionary mapping category to list of (line_num, method_name, signature)
    """
    categories = {
        'io': [],
        'transformations': [],
        'aggregations': [],
        'joins': [],
        'operations': [],
        'grouped': [],
        'windowed': [],
    }
    
    io_keywords = ['from_', 'to_', 'write_', 'read']
    transformation_keywords = ['filter', 'map', 'select', 'flat_map', 'with_column', 'rename']
    aggregation_keywords = ['groupby', 'agg', 'aggregate']
    join_keywords = ['join', 'union']
    operation_keywords = ['sort', 'limit', 'distinct', 'drop', 'fill', 'sample', 'cache', 
                         'persist', 'repartition', 'coalesce', 'checkpoint', 'collect', 
                         'take', 'count', 'show', 'head', 'tail']
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    current_class = None
    for i, line in enumerate(lines, 1):
        # Detect class definitions
        class_match = re.match(r'^class (\w+)', line.strip())
        if class_match:
            current_class = class_match.group(1)
            continue
        
        # Detect method definitions
        method_match = re.match(r'^\s+def (\w+)\(', line.strip())
        if method_match and current_class:
            method_name = method_match.group(1)
            if method_name.startswith('_'):
                continue
            
            # Categorize based on method name
            if any(kw in method_name for kw in io_keywords):
                categories['io'].append((i, method_name, current_class))
            elif any(kw in method_name for kw in transformation_keywords):
                categories['transformations'].append((i, method_name, current_class))
            elif any(kw in method_name for kw in aggregation_keywords):
                categories['aggregations'].append((i, method_name, current_class))
            elif any(kw in method_name for kw in join_keywords):
                categories['joins'].append((i, method_name, current_class))
            elif any(kw in method_name for kw in operation_keywords):
                categories['operations'].append((i, method_name, current_class))
            elif current_class == 'GroupedDataFrame':
                categories['grouped'].append((i, method_name, current_class))
            elif current_class == 'WindowedDataFrame':
                categories['windowed'].append((i, method_name, current_class))
    
    return categories


def print_analysis(categories: Dict[str, List[Tuple[int, str, str]]]):
    """Print analysis results."""
    print("DataFrame API Method Analysis\n" + "=" * 50)
    
    for category, methods in categories.items():
        if methods:
            print(f"\n{category.upper()} ({len(methods)} methods):")
            for line_num, method_name, class_name in methods[:10]:
                print(f"  Line {line_num}: {class_name}.{method_name}()")
            if len(methods) > 10:
                print(f"  ... and {len(methods) - 10} more")


if __name__ == '__main__':
    file_path = Path(__file__).parent.parent / 'pipeline' / 'api' / 'dataframe.py'
    
    if not file_path.exists():
        print(f"Error: {file_path} not found")
        exit(1)
    
    categories = analyze_dataframe_file(str(file_path))
    print_analysis(categories)
    
    print("\n\nRecommended Module Structure:")
    print("- io.py: Input/output operations")
    print("- transformations.py: Data transformations")
    print("- aggregations.py: Grouping operations")
    print("- joins.py: Join and union")
    print("- operations.py: Other operations")
    print("- grouped.py: GroupedDataFrame class")
    print("- windowed.py: WindowedDataFrame class")

