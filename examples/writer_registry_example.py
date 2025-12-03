"""Examples for automatic writer detection and explicit format specification.

Demonstrates:
1. Automatic format detection from file extensions
2. Explicit format specification
3. Listing available write formats
4. Using both native Ray Data writers and custom writers
"""

from pipeline.api import PipelineDataFrame
from pipeline.utils.writer_registry import list_formats, detect_writer, get_writer


def example_list_formats():
    """Example: List all supported write formats."""
    print("=== Supported Write Formats ===\n")
    
    formats = list_formats()
    
    print("Native Ray Data formats:")
    for fmt in sorted(formats["native"]):
        print(f"  - {fmt}")
    
    print("\nCustom writer formats:")
    for fmt in sorted(formats["custom"]):
        print(f"  - {fmt}")


def example_auto_detection():
    """Example: Automatic format detection."""
    print("\n=== Automatic Format Detection ===\n")
    
    # Auto-detect from file extension
    print("1. Auto-detect from extension:")
    writer = detect_writer("s3://bucket/output.parquet")
    print(f"   .parquet -> {'detected' if writer else 'not detected'}")
    
    writer = detect_writer("s3://bucket/output.json")
    print(f"   .json -> {'detected' if writer else 'not detected'}")
    
    writer = detect_writer("s3://bucket/output.csv")
    print(f"   .csv -> {'detected' if writer else 'not detected'}")
    
    writer = detect_writer("s3://bucket/output.h5")
    print(f"   .h5 -> {'detected' if writer else 'not detected'}")
    
    # Auto-detect from directory pattern
    print("\n2. Auto-detect from directory pattern:")
    writer = detect_writer("s3://bucket/output/parquet/")
    print(f"   /parquet/ -> {'detected' if writer else 'not detected'}")
    
    writer = detect_writer("s3://bucket/output/images/")
    print(f"   /images/ -> {'detected' if writer else 'not detected'}")


def example_explicit_format():
    """Example: Explicit format specification."""
    print("\n=== Explicit Format Specification ===\n")
    
    # Explicit format overrides auto-detection
    print("1. Explicit format specification:")
    writer = get_writer("parquet")
    print(f"   format='parquet' -> {'available' if writer else 'not available'}")
    
    writer = get_writer("json")
    print(f"   format='json' -> {'available' if writer else 'not available'}")
    
    writer = get_writer("hdf5")
    print(f"   format='hdf5' -> {'available' if writer else 'not available'}")


def example_pipeline_dataframe():
    """Example: Using PipelineDataFrame with auto-detection."""
    print("\n=== PipelineDataFrame Write Usage ===\n")
    
    print("1. Auto-detect from extension:")
    print("   df.write('s3://bucket/output.parquet')")
    print("   df.write('s3://bucket/output.json')")
    print("   df.write('s3://bucket/output.csv')")
    
    print("\n2. Explicit format specification:")
    print("   df.write('s3://bucket/output/', format='parquet')")
    
    print("\n3. Custom format:")
    print("   df.write('s3://bucket/output.h5', format='hdf5')")
    
    print("\n4. Specific write methods:")
    print("   df.write_parquet('s3://bucket/output.parquet')")
    print("   df.write_json('s3://bucket/output.json')")
    print("   df.write_csv('s3://bucket/output.csv')")
    print("   df.write_numpy('s3://bucket/output.npy')")
    print("   df.write_tfrecords('s3://bucket/output.tfrecord')")
    print("   df.write_images('s3://bucket/images/')")


def example_native_vs_custom():
    """Example: Native vs custom writers."""
    print("\n=== Native vs Custom Writers ===\n")
    
    print("Native Ray Data writers (optimized, built-in):")
    native_formats = ["parquet", "csv", "json", "numpy", "tfrecords", "images"]
    for fmt in native_formats:
        writer = get_writer(fmt)
        if writer:
            print(f"  ✓ {fmt}: Native Ray Data writer")
    
    print("\nCustom writers (specialized formats):")
    custom_formats = ["hdf5", "msgpack", "protobuf"]
    for fmt in custom_formats:
        writer = get_writer(fmt)
        if writer:
            print(f"  ✓ {fmt}: Custom writer")


def example_write_workflow():
    """Example: Complete write workflow."""
    print("\n=== Complete Write Workflow ===\n")
    
    print("1. Process data:")
    print("   df = PipelineDataFrame.from_paths('s3://bucket/input.parquet')")
    print("   processed = df.filter(lambda x: x['quality'] > 0.8)")
    
    print("\n2. Write to multiple formats:")
    print("   processed.write('s3://bucket/output.parquet')  # Auto-detect")
    print("   processed.write('s3://bucket/output.json')     # Auto-detect")
    print("   processed.write('s3://bucket/output.csv')      # Auto-detect")
    
    print("\n3. Write with explicit format:")
    print("   processed.write('s3://bucket/output/', format='parquet')")
    
    print("\n4. Write to custom format:")
    print("   processed.write('s3://bucket/output.h5', format='hdf5')")


if __name__ == "__main__":
    try:
        example_list_formats()
        example_auto_detection()
        example_explicit_format()
        example_pipeline_dataframe()
        example_native_vs_custom()
        example_write_workflow()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

