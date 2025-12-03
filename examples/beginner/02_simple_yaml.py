"""Simple YAML Runner - Beginner Level.

Load and run a pipeline from a YAML configuration file.
"""

import sys
from pathlib import Path

from pipeline.api import load_from_yaml

if __name__ == "__main__":
    # Get YAML file path (same directory as this script)
    yaml_file = Path(__file__).parent / "02_simple_yaml.yaml"
    
    if not yaml_file.exists():
        print(f"Error: YAML file not found: {yaml_file}")
        sys.exit(1)
    
    # Load pipeline from YAML
    pipeline = load_from_yaml(str(yaml_file))
    
    # Run the pipeline
    results = pipeline.run()
    
    print(f"✓ Pipeline completed!")
    print(f"  Processed: {results.get('total_samples', 0)} samples")
    print(f"  Output: {results.get('output_path', 'N/A')}")

