"""YAML Runner - Run any YAML configuration file.

Usage:
    python yaml/run_yaml.py <path_to_yaml_file>
    
Examples:
    python yaml/run_yaml.py ../beginner/02_simple_yaml.yaml
    python yaml/run_yaml.py ../intermediate/02_multiple_sources.yaml
    python yaml/run_yaml.py ../advanced/02_full_pipeline.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

from pipeline.api import Pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Run pipeline from YAML configuration file."""
    parser = argparse.ArgumentParser(
        description="Run pipeline from YAML configuration file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run beginner example
  python yaml/run_yaml.py ../beginner/02_simple_yaml.yaml
  
  # Run intermediate example
  python yaml/run_yaml.py ../intermediate/02_multiple_sources.yaml
  
  # Run advanced example
  python yaml/run_yaml.py ../advanced/02_full_pipeline.yaml
        """,
    )
    parser.add_argument(
        "yaml_file",
        type=str,
        help="Path to YAML configuration file",
    )
    args = parser.parse_args()
    
    # Resolve YAML file path
    yaml_path = Path(args.yaml_file)
    if not yaml_path.is_absolute():
        # Relative to examples directory
        examples_dir = Path(__file__).parent.parent
        yaml_path = examples_dir / args.yaml_file
    
    if not yaml_path.exists():
        logger.error(f"YAML file not found: {yaml_path}")
        logger.info("\nAvailable YAML examples:")
        logger.info("  Beginner:")
        logger.info("    - beginner/02_simple_yaml.yaml")
        logger.info("  Intermediate:")
        logger.info("    - intermediate/02_multiple_sources.yaml")
        logger.info("  Advanced:")
        logger.info("    - advanced/02_full_pipeline.yaml")
        sys.exit(1)
    
    logger.info(f"Loading pipeline from: {yaml_path}")
    
    try:
        # Load pipeline from YAML
        from pipeline.api import load_from_yaml
        pipeline = load_from_yaml(str(yaml_path))
    except Exception as e:
        logger.error(f"Failed to load YAML configuration: {e}")
        logger.error("Please check the YAML syntax and file path.")
        sys.exit(1)
    
    logger.info("Running pipeline...")
    try:
        results = pipeline.run()
        
        logger.info("=" * 60)
        logger.info("Pipeline execution completed successfully")
        logger.info("=" * 60)
        logger.info(f"Total samples: {results.get('total_samples', 0):,}")
        logger.info(f"Total duration: {results.get('total_duration', 0.0):.2f}s")
        logger.info(f"Output path: {results.get('output_path', 'N/A')}")
        
        if "dedup_rate" in results:
            logger.info(f"Deduplication rate: {results['dedup_rate']:.2%}")
        
        if "stages" in results:
            logger.info(f"\nStages completed: {len(results['stages'])}")
            for stage in results["stages"]:
                logger.info(
                    f"  - {stage.get('name', 'Unknown')}: "
                    f"{stage.get('duration', 0.0):.2f}s "
                    f"({stage.get('throughput', 0.0):.0f} samples/s)"
                )
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

