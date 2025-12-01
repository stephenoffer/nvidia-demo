# Examples

This directory contains essential examples demonstrating how to use the multimodal data pipeline.

## Quick Start

### 1. YAML Configuration Example

The simplest way to get started is using YAML configuration:

```bash
python examples/declarative_yaml.py
```

This example loads a pipeline configuration from `pipeline_config.yaml` and runs it. The YAML file demonstrates all supported data source types and configuration options.

**Files:**
- `declarative_yaml.py` - Example script
- `pipeline_config.yaml` - Complete YAML configuration example

### 2. Basic Declarative API Example

Simple Python API example for basic use cases:

```bash
python examples/basic_declarative_api.py
```

Demonstrates:
- Basic pipeline configuration
- Simple data sources (video, parquet)
- GPU acceleration
- Streaming execution

**File:** `basic_declarative_api.py`

### 3. Advanced Declarative API Example

Advanced Python API example with complex configurations:

```bash
python examples/advanced_declarative_api.py
```

Demonstrates:
- Multiple data source types (video, MCAP, HDF5, point clouds, etc.)
- GPU-accelerated deduplication
- Isaac Lab and Cosmos Dreams integration
- Advanced configuration options
- YAML export functionality

**File:** `advanced_declarative_api.py`

### 4. Model Training Example

Complete example of training the GR00T Vision-Language-Action (VLA) foundation model:

```bash
python examples/groot_model_training.py
```

Demonstrates:
- Full GR00T model architecture (System 2 VLM + System 1 Diffusion)
- Ray Data for data processing
- Ray Train for distributed training
- Latest NVIDIA GPU optimizations (Flash Attention, FSDP, mixed precision)
- Latest Ray features (streaming, GPU object store, RDMA)

**File:** `groot_model_training.py`

## Example Files

| File | Description |
|------|-------------|
| `declarative_yaml.py` | YAML-based pipeline configuration |
| `pipeline_config.yaml` | Complete YAML configuration example |
| `basic_declarative_api.py` | Simple Python API example |
| `advanced_declarative_api.py` | Advanced Python API example |
| `groot_model_training.py` | Complete model training example |

## Data Directory

The `data/` directory contains sample data files for testing:
- Test archives, binaries, calibration files
- Sample HDF5, MessagePack, Protobuf files
- Point cloud data (PCD, PLY formats)
- URDF robot models
- JSONL test data

## Getting Started

1. **Start with YAML**: Run `declarative_yaml.py` to see a complete example
2. **Try Basic API**: Run `basic_declarative_api.py` for simple use cases
3. **Explore Advanced**: Run `advanced_declarative_api.py` for complex scenarios
4. **Train Models**: Run `groot_model_training.py` for full model training

## Configuration

All examples use the same underlying pipeline configuration system. You can:
- Configure pipelines via YAML files
- Configure pipelines via Python API
- Export Python configurations to YAML
- Load YAML configurations in Python

See `pipeline_config.yaml` for a complete reference of all configuration options.

## Visualization

The pipeline uses **Grafana** as the primary visualization tool for production-grade monitoring:

- **Grafana Dashboard**: Automatically generated JSON configuration that can be imported into Grafana
- **Prometheus Integration**: Metrics are exported to Prometheus for real-time monitoring
- **Alert Rules**: Prometheus alert rules are generated for automated alerting
- **HTML Summary**: A simple HTML summary page is created with instructions for Grafana setup

After running an example, you'll find:
- `grafana_dashboard.json` - Grafana dashboard configuration (import into Grafana)
- `prometheus_alerts.json` - Prometheus alert rules (import into Alertmanager)
- `pipeline_summary.html` - HTML summary with Grafana setup instructions

### Using Grafana Dashboards

1. **Start Grafana** (if not already running):
   ```bash
   docker run -d -p 3000:3000 grafana/grafana
   ```

2. **Configure Prometheus Datasource**:
   - Open Grafana at http://localhost:3000
   - Go to Configuration → Data Sources
   - Add Prometheus datasource (default: `http://localhost:9090`)

3. **Import Dashboard**:
   - Go to Dashboards → Import
   - Upload `grafana_dashboard.json`
   - Select the Prometheus datasource
   - View your pipeline metrics!

### Fallback Visualization

If Grafana is not available, the pipeline falls back to:
- **Plotly** (for interactive web-based dashboards) - Primary backup
- **Basic text summary** (if Plotly is also unavailable)

To use a different visualization backend, set `dashboard_mode` when creating the pipeline:
```python
pipeline = Pipeline(..., dashboard_mode="plotly")  # Use Plotly interactive dashboard
pipeline = Pipeline(..., dashboard_mode="grafana")  # Use Grafana (default)
```

**Plotly Features:**
- Interactive HTML dashboards that can be opened in any web browser
- Zoom, pan, and hover interactions
- Professional-looking charts and graphs
- No server required - works offline
