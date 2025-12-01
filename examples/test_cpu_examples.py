"""Test CPU-only examples with local test data.

This script tests all CPU-only examples to ensure they work correctly
on systems without GPU support.
"""

import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def test_basic_declarative_api():
    """Test basic declarative API example."""
    logger.info("Testing: Basic Declarative API")
    
    try:
        from examples.basic_declarative_api import main
        main()
        logger.info("Basic declarative API test passed")
        return True
    except Exception as e:
        logger.error(f"Basic declarative API test failed: {e}", exc_info=True)
        return False


def test_advanced_declarative_api():
    """Test advanced declarative API example."""
    logger.info("Testing: Advanced Declarative API")
    
    try:
        from examples.advanced_declarative_api import main
        main()
        logger.info("Advanced declarative API test passed")
        return True
    except Exception as e:
        logger.error(f"Advanced declarative API test failed: {e}", exc_info=True)
        return False


def test_yaml_config():
    """Test YAML configuration example."""
    logger.info("Testing: YAML Configuration")
    
    try:
        from examples.declarative_yaml import main
        main()
        logger.info("YAML configuration test passed")
        return True
    except Exception as e:
        logger.error(f"YAML configuration test failed: {e}", exc_info=True)
        return False


def create_test_data():
    """Create test data files for testing."""
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)
    
    test_output = Path(__file__).parent / "test_output"
    test_output.mkdir(exist_ok=True)
    
    # Create a simple parquet test file
    try:
        import pandas as pd
        
        test_data = pd.DataFrame({
            "id": range(100),
            "text": [f"Sample text data {i}" for i in range(100)],
            "value": [float(i * 1.5) for i in range(100)],
            "category": [f"cat_{i % 5}" for i in range(100)],
        })
        
        parquet_path = test_output / "test_data.parquet"
        test_data.to_parquet(parquet_path, index=False)
        logger.info(f"Created test parquet file: {parquet_path}")
        
        return True
    except ImportError:
        logger.warning("pandas not available, skipping test data creation")
        return False
    except Exception as e:
        logger.error(f"Failed to create test data: {e}", exc_info=True)
        return False


def main():
    """Run all CPU-only tests."""
    logger.info("Starting CPU-only examples test suite")
    
    # Check environment
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        logger.info(f"Environment: CUDA available: {cuda_available}")
        if cuda_available:
            logger.info(f"Environment: GPU count: {torch.cuda.device_count()}")
        else:
            logger.info("Environment: Running in CPU-only mode")
    except ImportError:
        logger.info("Environment: PyTorch not available")
        cuda_available = False
    
    # Create test data
    logger.info("Creating test data...")
    create_test_data()
    
    # Initialize Ray in CPU-only mode
    logger.info("Initializing Ray...")
    try:
        import ray
        
        if not ray.is_initialized():
            ray.init(
                num_cpus=4,
                num_gpus=0,  # Force CPU-only
                ignore_reinit_error=True,
                log_to_driver=False,
            )
            logger.info("Ray initialized in CPU-only mode")
        else:
            logger.info("Ray already initialized")
    except Exception as e:
        logger.error(f"Failed to initialize Ray: {e}", exc_info=True)
        logger.warning("Some tests may fail without Ray")
    
    # Run tests
    results = []
    
    # Test 1: Basic declarative API
    try:
        results.append(("Basic Declarative API", test_basic_declarative_api()))
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        results.append(("Basic Declarative API", False))
    
    # Test 2: Advanced declarative API
    try:
        results.append(("Advanced Declarative API", test_advanced_declarative_api()))
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        results.append(("Advanced Declarative API", False))
    
    # Test 3: YAML configuration (may fail if S3 not configured)
    try:
        results.append(("YAML Configuration", test_yaml_config()))
    except Exception as e:
        logger.warning(f"YAML test skipped (may need S3): {e}")
        results.append(("YAML Configuration", None))  # None = skipped
    
    # Log summary
    logger.info("Test Summary")
    
    passed = 0
    failed = 0
    skipped = 0
    
    for test_name, result in results:
        if result is True:
            logger.info(f"{test_name}: PASSED")
            passed += 1
        elif result is False:
            logger.error(f"{test_name}: FAILED")
            failed += 1
        else:
            logger.warning(f"{test_name}: SKIPPED")
            skipped += 1
    
    logger.info(f"Total: {len(results)} tests - Passed: {passed}, Failed: {failed}, Skipped: {skipped}")
    
    # Cleanup
    try:
        import ray
        if ray.is_initialized():
            ray.shutdown()
            logger.info("Ray shutdown complete")
    except Exception:
        pass
    
    # Return success if all non-skipped tests passed
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

