"""Examples for automatic reader detection and explicit format specification.

Demonstrates:
1. Automatic format detection from file extensions
2. Explicit format specification
3. Listing available formats
4. Using both native Ray Data readers and custom datasources
"""

from pipeline.api import PipelineDataFrame
from pipeline.utils.reader_registry import list_formats, detect_reader, get_reader


def example_list_formats():
    """Example: List all supported formats."""
    print("=== Supported Formats ===\n")
    
    formats = list_formats()
    
    print("Native Ray Data formats:")
    for fmt in sorted(formats["native"]):
        print(f"  - {fmt}")
    
    print("\nCustom datasource formats:")
    for fmt in sorted(formats["custom"]):
        print(f"  - {fmt}")


def example_auto_detection():
    """Example: Automatic format detection."""
    print("\n=== Automatic Format Detection ===\n")
    
    # Auto-detect from file extension
    print("1. Auto-detect from extension:")
    reader = detect_reader("s3://bucket/data.parquet")
    print(f"   .parquet -> {reader.__name__ if reader else 'None'}")
    
    reader = detect_reader("s3://bucket/data.json")
    print(f"   .json -> {reader.__name__ if reader else 'None'}")
    
    reader = detect_reader("s3://bucket/data.bag")
    print(f"   .bag -> {reader.__name__ if reader else 'None'}")
    
    reader = detect_reader("s3://bucket/data.mcap")
    print(f"   .mcap -> {reader.__name__ if reader else 'None'}")
    
    # Auto-detect from directory pattern
    print("\n2. Auto-detect from directory pattern:")
    reader = detect_reader("s3://bucket/data/parquet/")
    print(f"   /parquet/ -> {reader.__name__ if reader else 'None'}")
    
    reader = detect_reader("s3://bucket/data/images/")
    print(f"   /images/ -> {reader.__name__ if reader else 'None'}")


def example_explicit_format():
    """Example: Explicit format specification."""
    print("\n=== Explicit Format Specification ===\n")
    
    # Explicit format overrides auto-detection
    print("1. Explicit format specification:")
    reader = get_reader("parquet")
    print(f"   format='parquet' -> {reader.__name__ if reader else 'None'}")
    
    reader = get_reader("rosbag")
    print(f"   format='rosbag' -> {reader.__name__ if reader else 'None'}")
    
    reader = get_reader("mcap")
    print(f"   format='mcap' -> {reader.__name__ if reader else 'None'}")


def example_pipeline_dataframe():
    """Example: Using PipelineDataFrame with auto-detection."""
    print("\n=== PipelineDataFrame Usage ===\n")
    
    print("1. Auto-detect from extension:")
    print("   df = PipelineDataFrame.from_paths('s3://bucket/data.parquet')")
    
    print("\n2. Explicit format specification:")
    print("   df = PipelineDataFrame.from_paths('s3://bucket/data/', format='parquet')")
    
    print("\n3. Custom datasource:")
    print("   df = PipelineDataFrame.from_paths('s3://bucket/data.bag', format='rosbag')")
    
    print("\n4. Multiple paths:")
    print("   df = PipelineDataFrame.from_paths([")
    print("       's3://bucket/data1.parquet',")
    print("       's3://bucket/data2.parquet'")
    print("   ])")


def example_native_vs_custom():
    """Example: Native vs custom readers."""
    print("\n=== Native vs Custom Readers ===\n")
    
    print("Native Ray Data readers (optimized, built-in):")
    native_formats = ["parquet", "csv", "json", "images", "videos", "audio", "numpy", "tfrecords"]
    for fmt in native_formats:
        reader = get_reader(fmt)
        if reader:
            print(f"  ✓ {fmt}: {reader.__module__}.{reader.__name__}")
    
    print("\nCustom datasources (robotics-specific formats):")
    custom_formats = ["mcap", "rosbag", "ros2bag", "hdf5", "pointcloud", "urdf"]
    for fmt in custom_formats:
        reader = get_reader(fmt)
        if reader:
            print(f"  ✓ {fmt}: Custom datasource")


if __name__ == "__main__":
    try:
        example_list_formats()
        example_auto_detection()
        example_explicit_format()
        example_pipeline_dataframe()
        example_native_vs_custom()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

