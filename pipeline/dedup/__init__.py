"""GPU-accelerated deduplication modules."""

from pipeline.dedup.gpu_dedup import GPUDeduplicator
from pipeline.dedup.lsh import LSHDeduplicator
from pipeline.dedup.semantic import SemanticDeduplicator

__all__ = [
    "GPUDeduplicator",
    "LSHDeduplicator",
    "SemanticDeduplicator",
]
