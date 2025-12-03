# Examples Organization Summary

Examples are now organized by complexity level in subdirectories, using **real data sources**.

## Directory Structure

```
examples/
├── beginner/                    # Simplest examples (3 Python, 1 YAML)
│   ├── 01_simple_python.py      # One-liner example
│   ├── 02_simple_yaml.yaml      # Simplest YAML config
│   └── 02_simple_yaml.py        # YAML runner
│
├── intermediate/                # Medium complexity (3 Python, 1 YAML)
│   ├── 01_multiple_sources.py  # Multiple data sources
│   ├── 02_multiple_sources.yaml # Multiple sources YAML
│   └── 03_dataframe_api.py     # DataFrame API with Pythonic features
│
├── advanced/                    # Full-featured (2 Python, 1 YAML)
│   ├── 01_full_pipeline.py    # Full pipeline with MLOps
│   └── 02_full_pipeline.yaml   # Full pipeline YAML
│
├── yaml/                        # YAML utilities (1 Python)
│   └── run_yaml.py             # Universal YAML runner
│
├── data/                        # Test data (auto-detected)
│   ├── parquet/
│   ├── jsonl/
│   └── ...
│
└── [root level]                 # Specialized examples
    ├── groot_model_training.py
    ├── mlops_batch_inference.py
    ├── dataframe_api_example.py
    └── ...
```

## Data Sources

All examples use **real data**:
- ✅ **Public S3 video**: `s3://anonymous@ray-example-data/basketball.mp4` (no credentials needed)
- ✅ **Local test data**: Auto-detected from `examples/data/` if available

## Quick Commands

```bash
# Beginner
python examples/beginner/01_simple_python.py
python examples/beginner/02_simple_yaml.py

# Intermediate
python examples/intermediate/01_multiple_sources.py
python examples/yaml/run_yaml.py intermediate/02_multiple_sources.yaml
python examples/intermediate/03_dataframe_api.py

# Advanced
python examples/advanced/01_full_pipeline.py
python examples/yaml/run_yaml.py advanced/02_full_pipeline.yaml
```

## Key Features

- ✅ **Organized by complexity** - Easy to find the right example
- ✅ **Real data** - No fake/mock data, uses public S3 video or local test data
- ✅ **Both APIs** - Python and YAML examples at each level
- ✅ **Auto-detection** - Automatically uses local test data if available
- ✅ **Clear naming** - Numbered files show progression (01_, 02_, 03_)
