"""Utility modules for the multimodal data pipeline."""

from pipeline.utils.gpu.memory import (
    check_gpu_memory,
    clear_gpu_cache,
    get_cuda_device,
    get_gpu_memory_info,
    gpu_memory_cleanup,
    set_cuda_device,
    synchronize_cuda,
)
from pipeline.utils.gpu.monitoring import (
    check_gpu_health,
    get_gpu_utilization,
    log_gpu_status,
)
from pipeline.utils.gpu.rapids import (
    check_cudf_compatibility,
    check_cuml_compatibility,
    get_rmm_memory_info,
    initialize_rapids_environment,
    initialize_rmm_pool,
    optimize_cudf_settings,
)
from pipeline.utils.nccl import (
    NCCLGroup,
    gather_tensors_across_gpus,
    initialize_nccl_for_ray_actors,
)
from pipeline.utils.resource_manager import (
    ResourceManager,
    check_disk_space,
    get_resource_manager,
    validate_path,
)
from pipeline.utils.context import PipelineContext, get_context
from pipeline.utils.data.reader_registry import (
    ReaderRegistry,
    detect_reader,
    get_reader,
    list_formats,
)
from pipeline.utils.data.writer_registry import (
    WriterRegistry,
    detect_writer,
    get_writer as get_writer_func,
    list_formats as list_write_formats,
)
from pipeline.utils.execution import (
    RetryConfig,
    retry_with_backoff,
    retry_cloud_storage,
    retry_with_exponential_backoff,
    with_timeout,
)

# Note: Caching utilities removed - use functools.lru_cache or cachetools directly
# Note: Circuit breaker removed - use pybreaker library if needed
# Note: Connection pool removed - use database-specific pools (SQLAlchemy, etc.)
# Note: Secret manager removed - use python-dotenv or cloud SDKs

__all__ = [
    # NCCL utilities
    "NCCLGroup",
    "initialize_nccl_for_ray_actors",
    "gather_tensors_across_gpus",
    # GPU utilities
    "get_cuda_device",
    "get_gpu_memory_info",
    "check_gpu_memory",
    "gpu_memory_cleanup",
    "clear_gpu_cache",
    "set_cuda_device",
    "synchronize_cuda",
    # GPU monitoring
    "get_gpu_utilization",
    "check_gpu_health",
    "log_gpu_status",
    # RAPIDS initialization
    "initialize_rapids_environment",
    "initialize_rmm_pool",
    "check_cudf_compatibility",
    "check_cuml_compatibility",
    "optimize_cudf_settings",
    "get_rmm_memory_info",
    # Resource management
    "ResourceManager",
    "get_resource_manager",
    "check_disk_space",
    "validate_path",
    # Retry utilities
    "RetryConfig",
    "retry_with_backoff",
    "with_timeout",
    # Context management
    "PipelineContext",
    "get_context",
    # Reader registry
    "ReaderRegistry",
    "detect_reader",
    "get_reader",
    "list_formats",
    # Writer registry
    "WriterRegistry",
    "detect_writer",
    "get_writer_func",
    "list_write_formats",
]

