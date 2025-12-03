# Examples

This directory contains examples organized by complexity level, demonstrating both Python and YAML APIs using **real data sources**.

## Directory Structure

```
examples/
├── beginner/          # Simplest examples for first-time users
├── intermediate/      # Examples with multiple features
├── advanced/          # Full-featured examples with MLOps
├── yaml/              # YAML utilities and runners
├── data/              # Test data files
└── [specialized]      # Specialized examples (training, MLOps, etc.)
```

## Quick Start

All examples use **real data**:
- **Public S3 video**: `s3://anonymous@ray-example-data/basketball.mp4` (no credentials needed)
- **Local test data**: Automatically detected if available in `data/` directory

## Example Organization

Examples are organized into three complexity levels:

### Beginner Level (`beginner/`)

**Simplest examples to get started:**

1. **Simple Python** (`beginner/01_simple_python.py`)
   ```bash
   python examples/beginner/01_simple_python.py
   ```
   - One-liner pipeline creation
   - Uses public S3 video (`s3://anonymous@ray-example-data/basketball.mp4`)
   - Minimal configuration
   - **Real data**: Public S3 video (no credentials needed)

2. **Simple YAML** (`beginner/02_simple_yaml.yaml` + `beginner/02_simple_yaml.py`)
   ```bash
   python examples/beginner/02_simple_yaml.py
   ```
   - Simplest YAML configuration
   - Uses public S3 video
   - Perfect for YAML users
   - **Real data**: Public S3 video (no credentials needed)

### Intermediate Level (`intermediate/`)

**Examples with multiple sources and features:**

1. **Multiple Sources** (`intermediate/01_multiple_sources.py`)
   ```bash
   python examples/intermediate/01_multiple_sources.py
   ```
   - Multiple data sources
   - Public S3 video + local test data (auto-detected)
   - GPU acceleration (if available)
   - Result inspection
   - **Real data**: Public S3 video + local test data

2. **Multiple Sources YAML** (`intermediate/02_multiple_sources.yaml`)
   ```bash
   python examples/yaml/run_yaml.py intermediate/02_multiple_sources.yaml
   ```
   - YAML version of multiple sources example
   - Same features as Python version
   - **Real data**: Public S3 video + local test data

3. **DataFrame API** (`intermediate/03_dataframe_api.py`)
   ```bash
   python examples/intermediate/03_dataframe_api.py
   ```
   - Pythonic DataFrame API
   - Standard Python built-ins (`len()`, `iter()`, etc.)
   - Operator overloading (`+`, `|`, etc.)
   - Indexing and slicing (`df[0:10]`, `df.column`)
   - **Real data**: Demonstrates with test data

### Advanced Level (`advanced/`)

**Full-featured examples with MLOps integration:**

1. **Full Pipeline** (`advanced/01_full_pipeline.py`)
   ```bash
   python examples/advanced/01_full_pipeline.py
   ```
   - Complex multi-source pipeline
   - Data quality checks (profiling, validation)
   - Experiment tracking (MLflow/W&B)
   - Advanced configuration
   - **Real data**: Public S3 video + local test data

2. **Full Pipeline YAML** (`advanced/02_full_pipeline.yaml`)
   ```bash
   python examples/yaml/run_yaml.py advanced/02_full_pipeline.yaml
   ```
   - YAML version with all advanced features
   - MLOps integration
   - Data quality stages
   - **Real data**: Public S3 video + local test data

## Data Sources

All examples use **real data**:

### Public S3 Video (Always Available)
- **Path**: `s3://anonymous@ray-example-data/basketball.mp4`
- **No credentials needed**
- **Used in**: All beginner and intermediate examples

### Local Test Data (Optional)
- **Location**: `examples/data/`
- **Formats**: Parquet, JSONL, HDF5
- **Auto-detected**: Examples automatically use local data if available
- **Create test data**: Run `python examples/create_mock_data.py`

## Running Examples

### Python Examples
```bash
# Beginner - Simplest example
python examples/beginner/01_simple_python.py

# Intermediate - Multiple sources
python examples/intermediate/01_multiple_sources.py

# Advanced - Full pipeline with MLOps
python examples/advanced/01_full_pipeline.py
```

### YAML Examples
```bash
# Using the YAML runner (from project root)
python examples/yaml/run_yaml.py beginner/02_simple_yaml.yaml
python examples/yaml/run_yaml.py intermediate/02_multiple_sources.yaml
python examples/yaml/run_yaml.py advanced/02_full_pipeline.yaml

# Or from examples directory
cd examples
python yaml/run_yaml.py beginner/02_simple_yaml.yaml
python yaml/run_yaml.py intermediate/02_multiple_sources.yaml
python yaml/run_yaml.py advanced/02_full_pipeline.yaml
```

## Example Files Reference

### Beginner Level (`beginner/`)
| File | Description | Data Source |
|------|-------------|-------------|
| `01_simple_python.py` | Simplest Python example (one-liner) | Public S3 video |
| `02_simple_yaml.yaml` | Simplest YAML configuration | Public S3 video |
| `02_simple_yaml.py` | YAML runner script | Public S3 video |

### Intermediate Level (`intermediate/`)
| File | Description | Data Source |
|------|-------------|-------------|
| `01_multiple_sources.py` | Multiple data sources | Public S3 video + local |
| `02_multiple_sources.yaml` | Multiple sources YAML | Public S3 video + local |
| `03_dataframe_api.py` | DataFrame API with Pythonic features | Test data |

### Advanced Level (`advanced/`)
| File | Description | Data Source |
|------|-------------|-------------|
| `01_full_pipeline.py` | Full pipeline with MLOps | Public S3 video + local |
| `02_full_pipeline.yaml` | Full pipeline YAML | Public S3 video + local |

### YAML Utilities (`yaml/`)
| File | Description |
|------|-------------|
| `run_yaml.py` | Universal YAML runner for any config file |

### Specialized Examples (Root Level)
| File | Description | Data Source |
|------|-------------|-------------|
| `dataframe_api_example.py` | Comprehensive DataFrame API examples | Various |
| `fluent_api_example.py` | Fluent Builder API examples | Various |
| `api_quick_start.py` | Quick start examples for all APIs | Various |
| `groot_model_training.py` | Complete GR00T model training | Training data |
| `mlops_batch_inference.py` | MLOps batch inference | Inference data |
| `wandb_example.py` | Weights & Biases integration | Various |
| `basic_declarative_api.py` | Basic declarative API (legacy) | Public S3 video + local |
| `advanced_declarative_api.py` | Advanced declarative API (legacy) | Public S3 video + local |
| `declarative_yaml.py` | YAML loader (legacy) | From YAML config |
| `pipeline_config.yaml` | Complete YAML reference | Various |

## Creating Test Data

To create local test data for examples (optional):

```bash
python examples/create_mock_data.py
```

This creates test files in `examples/data/`:
- `parquet/mock_data.parquet`
- `jsonl/mock_data.jsonl`
- `hdf5/mock_sensor_data.h5`
- And more...

**Note:** Examples work without local test data - they use the public S3 video by default.
Local test data is automatically detected and used if available.

## Configuration

All examples:
- **Use real data** (public S3 video or local test data)
- **Auto-detect** available resources (CPU/GPU)
- **Work out of the box** with minimal setup
- **Provide clear output** showing what was processed

## Next Steps

1. **Start with beginner examples** (`beginner/`) to understand the basics
2. **Try intermediate examples** (`intermediate/`) to see multiple features
3. **Explore advanced examples** (`advanced/`) for full MLOps integration
4. **Check specialized examples** (root level) for specific use cases

## Quick Reference

- **Simplest Python**: `python examples/beginner/01_simple_python.py`
- **Simplest YAML**: `python examples/beginner/02_simple_yaml.py`
- **Multiple Sources**: `python examples/intermediate/01_multiple_sources.py`
- **DataFrame API**: `python examples/intermediate/03_dataframe_api.py`
- **Full Pipeline**: `python examples/advanced/01_full_pipeline.py`
- **YAML Runner**: `python examples/yaml/run_yaml.py <config.yaml>`

See `INDEX.md` for a complete quick reference guide.
