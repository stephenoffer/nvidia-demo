# Architecture Overview

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Data Sources                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │  Videos  │  │   Text   │  │  Sensor  │  │  Isaac   │   │
│  │          │  │          │  │   Data   │  │   Lab    │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              Ray Data Loaders                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  MultimodalLoader (Format Detection & Loading)      │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              Processing Pipeline                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Temporal   │  │    Episode   │  │ Transition   │     │
│  │   Alignment  │  │   Detection   │  │  Alignment   │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Instruction  │  │   Quality    │  │   Physics    │     │
│  │  Grounding   │  │   Scoring    │  │ Validation   │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│            GPU-Accelerated Deduplication                    │
│  ┌──────────────┐              ┌──────────────┐            │
│  │  LSH (Fuzzy) │              │  Semantic    │            │
│  │  Deduplication│              │ Deduplication│            │
│  └──────────────┘              └──────────────┘            │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              Output Generation                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Parquet/Arrow Format with Compression              │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Component Architecture

### Core Components

1. **MultimodalPipeline**: Main orchestrator
2. **PipelineExecutor**: Stage execution engine
3. **PipelineLifecycleManager**: Resource management
4. **MultimodalLoader**: Data loading abstraction

### Processing Stages

- **TemporalAlignmentStage**: Aligns multimodal streams
- **EpisodeBoundaryDetector**: Detects episode boundaries
- **TransitionAlignmentStage**: Creates (s, a, r, s') tuples
- **InstructionGroundingStage**: Pairs instructions with demos
- **DataQualityScorer**: Scores data quality
- **GPUDeduplicator**: GPU-accelerated deduplication

### Infrastructure

- **Ray Cluster**: Distributed processing
- **Kubernetes**: Container orchestration
- **Prometheus**: Metrics collection
- **Grafana**: Visualization

## Data Flow

1. **Ingestion**: Data loaded from various sources
2. **Alignment**: Temporal and semantic alignment
3. **Validation**: Quality checks and validation
4. **Deduplication**: Remove duplicates
5. **Transformation**: Format for training
6. **Output**: Write curated dataset

## Scalability

- **Horizontal**: Add more Ray workers
- **Vertical**: Increase GPU/CPU per pod
- **Streaming**: Process data incrementally
- **Checkpointing**: Resume from failures

