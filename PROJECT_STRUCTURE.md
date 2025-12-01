# Project Structure

This document describes the complete structure and organization of the multimodal data pipeline project.

## Directory Layout

```
nvidia-demo/                          # Project root
├── pipeline/                         # Main Python package
│   ├── __init__.py                  # Package initialization
│   ├── __version__.py               # Version information
│   ├── cli.py                       # Command-line interface
│   ├── config.py                    # Configuration management
│   ├── exceptions.py                # Exception hierarchy
│   ├── health.py                    # Health check utilities
│   ├── core.py                      # Compatibility shim
│   │
│   ├── api/                         # Public API
│   │   ├── declarative.py          # Declarative API
│   │   ├── multipipeline.py        # Multi-pipeline runner
│   │   └── types.py                # API type definitions
│   │
│   ├── core/                        # Core pipeline components
│   │   ├── orchestrator.py         # Main orchestrator
│   │   ├── execution.py            # Stage execution
│   │   ├── lifecycle.py            # Lifecycle management
│   │   └── visualization.py        # Visualization management
│   │
│   ├── config/                      # Configuration modules
│   │   └── ray_data.py             # Ray Data configuration
│   │
│   ├── datasources/                 # Data source implementations
│   │   ├── base.py                 # Base datasource class
│   │   ├── groot.py                # GR00T format support
│   │   ├── ros2bag.py              # ROS2 bag support
│   │   └── ...                     # Other datasources
│   │
│   ├── dedup/                       # Deduplication modules
│   │   ├── gpu_dedup.py            # GPU deduplication orchestrator
│   │   ├── lsh.py                  # LSH deduplication
│   │   ├── semantic.py             # Semantic deduplication
│   │   └── sequence_dedup.py       # Sequence-level deduplication
│   │
│   ├── integrations/                # External integrations
│   │   ├── cosmos.py               # Cosmos Dreams integration
│   │   ├── isaac_lab.py            # Isaac Lab integration
│   │   ├── omniverse.py             # Omniverse integration
│   │   └── ...                     # Other integrations
│   │
│   ├── loaders/                      # Data loaders
│   │   ├── formats.py              # Format detection
│   │   └── multimodal.py            # Multimodal loader
│   │
│   ├── observability/                # Monitoring and observability
│   │   ├── metrics.py              # Metrics collection
│   │   ├── prometheus.py           # Prometheus exporter
│   │   └── grafana.py              # Grafana dashboard generation
│   │
│   ├── server/                       # HTTP server components
│   │   └── health_server.py         # Health check server
│   │
│   ├── stages/                       # Processing stages
│   │   ├── base.py                 # Base stage classes
│   │   ├── temporal_alignment.py   # Temporal alignment
│   │   ├── episode_detector.py     # Episode detection
│   │   └── ...                     # Other stages
│   │
│   ├── synthetic/                   # Synthetic data generation
│   │   ├── generator.py            # Base generator
│   │   ├── robotics.py             # Robotics data
│   │   ├── text.py                 # Text data
│   │   └── video.py                # Video data
│   │
│   ├── training/                     # Training integration
│   │   ├── integration.py          # Training integration
│   │   ├── eval.py                 # Evaluation utilities
│   │   ├── sequence_packing.py     # Sequence packing
│   │   └── attention_masks.py      # Attention mask generation
│   │
│   ├── utils/                        # Utility modules
│   │   ├── gpu/                    # GPU utilities
│   │   ├── ray/                    # Ray utilities
│   │   ├── context_managers.py     # Context managers
│   │   ├── logging_config.py      # Logging configuration
│   │   └── ...                     # Other utilities
│   │
│   └── visualization/                # Visualization modules
│       ├── dashboard.py            # Dashboard generation
│       ├── renderer.py             # Rendering utilities
│       └── video_generator.py      # Video generation
│
├── tests/                            # Test suite
│   ├── unit/                        # Unit tests
│   ├── integration/                 # Integration tests
│   ├── benchmarks/                  # Performance benchmarks
│   ├── fixtures/                    # Test fixtures
│   └── conftest.py                  # Pytest configuration
│
├── examples/                         # Example scripts and demos
│   ├── complete_groot_pipeline.py  # Complete GR00T pipeline example
│   ├── isaac_lab_integration.py     # Isaac Lab example
│   └── ...                         # Other examples
│
├── docs/                             # Documentation
│   ├── api/                         # API documentation
│   ├── architecture/                # Architecture documentation
│   ├── deployment/                  # Deployment guides
│   ├── guides/                      # User guides
│   └── operations/                  # Operations documentation
│
├── deployment/                       # Deployment configurations
│   ├── helm/                        # Helm charts
│   ├── kustomize/                   # Kustomize overlays
│   ├── kubernetes.yaml              # Kubernetes manifests
│   └── ...                         # Other deployment files
│
├── terraform/                        # Infrastructure as Code
│   ├── main.tf                      # Main Terraform config
│   ├── variables.tf                 # Variables
│   └── outputs.tf                   # Outputs
│
├── scripts/                          # Utility scripts
│   ├── deploy.sh                    # Deployment script
│   ├── setup-dev.sh                 # Development setup
│   └── ...                         # Other scripts
│
├── .github/                          # GitHub configuration
│   ├── workflows/                   # CI/CD workflows
│   ├── ISSUE_TEMPLATE/              # Issue templates
│   └── dependabot.yml               # Dependabot config
│
├── pyproject.toml                    # Project metadata and build config
├── setup.py                          # Setup script (legacy)
├── requirements.txt                  # Production dependencies
├── requirements-dev.txt              # Development dependencies
├── requirements-test.txt             # Test dependencies
├── MANIFEST.in                       # Package data manifest
├── README.md                         # Project README
├── CHANGELOG.md                      # Version changelog
├── CONTRIBUTING.md                   # Contribution guidelines
├── LICENSE                           # License file
├── Dockerfile                        # Container image definition
├── docker-compose.yml                # Local development environment
├── Makefile                          # Build automation
└── pytest.ini                       # Pytest configuration
```

## Package Naming

- **Project Name**: `multimodal-data-pipeline` (PyPI package name)
- **Package Name**: `pipeline` (Python import name)
- **Repository Name**: `nvidia-demo` (Git repository)

## Import Structure

```python
# Main package imports
from pipeline import MultimodalPipeline, PipelineConfig
from pipeline.core import PipelineExecutor
from pipeline.stages import TemporalAlignmentStage
from pipeline.integrations import IsaacLabLoader
```

## Key Design Decisions

1. **Flat package structure**: Package modules are directly under `pipeline/` rather than nested deeper
2. **Separation of concerns**: Clear separation between core, stages, integrations, etc.
3. **Backward compatibility**: `pipeline/core.py` provides compatibility shim
4. **Examples over demos**: All example code in `examples/` directory
5. **Documentation organization**: All docs in `docs/` with clear subdirectories

