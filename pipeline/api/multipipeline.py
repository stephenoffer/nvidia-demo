"""Multi-pipeline orchestration utilities for the declarative API."""

from __future__ import annotations

import logging
from typing import Any, Optional


import ray  # https://docs.ray.io/

from pipeline.api.types import CombineConfig, CombineMode, PipelineSpecConfig

logger = logging.getLogger(__name__)


class MultiPipelineRunner:
    """Run multiple declarative pipelines concurrently via Ray Core tasks."""

    def __init__(
        self,
        pipeline_specs: list[PipelineSpecConfig],
        combine_config: Optional[CombineConfig] = None,
        name: Optional[str] = None,
    ) -> None:
        if not pipeline_specs:
            raise ValueError("pipeline_specs cannot be empty")

        self.name = name or "multi_pipeline"
        self.pipeline_specs = pipeline_specs
        self.combine_config = combine_config or CombineConfig()

    def run(self) -> dict[str, Any]:
        """Run pipelines concurrently and optionally combine their outputs."""
        if not ray.is_initialized():
            # Use centralized Ray initialization
            from pipeline.utils.ray.init import initialize_ray
            
            init_result = initialize_ray(
                address=None,  # Auto-detect from RAY_ADDRESS
                ignore_reinit_error=True,
                configure_logging=True,
            )
            if not init_result.get("initialized"):
                raise RuntimeError("Failed to initialize Ray for multi-pipeline execution")

        tasks = []
        for spec in self.pipeline_specs:
            payload = {
                "pipeline_args": spec.pipeline_args,
                "name": spec.name,
            }
            # Create remote handle with resource specifications
            # Reuse handle for better performance (don't recreate each time)
            remote_handle = _execute_pipeline_spec.options(
                num_cpus=spec.task_cpus,
                num_gpus=spec.task_gpus,
                memory=spec.task_memory if hasattr(spec, "task_memory") else None,
                max_retries=2,  # Retry failed tasks up to 2 times
            )
            tasks.append(remote_handle.remote(payload))

        # Use ray.wait() for better async processing and error handling
        # Process tasks as they complete instead of waiting for all
        pipeline_results = []
        remaining_tasks = tasks
        
        while remaining_tasks:
            # Wait for at least one task to complete
            ready, remaining = ray.wait(remaining_tasks, num_returns=1, timeout=300.0)
            if ready:
                try:
                    result = ray.get(ready[0])
                    pipeline_results.append(result)
                    remaining_tasks = remaining
                except Exception as e:
                    logger.error(f"Pipeline task failed: {e}", exc_info=True)
                    # Remove failed task
                    remaining_tasks = [t for t in remaining_tasks if t != ready[0]]
            else:
                # Timeout - log warning and continue
                logger.warning("Timeout waiting for pipeline tasks, continuing with completed tasks")
                break
        
        # Get any remaining tasks synchronously with timeout
        if remaining_tasks:
            try:
                from pipeline.utils.constants import _DEFAULT_RAY_TASK_TIMEOUT
                remaining_results = ray.get(remaining_tasks, timeout=_DEFAULT_RAY_TASK_TIMEOUT)
                pipeline_results.extend(remaining_results)
            except ray.exceptions.GetTimeoutError:
                logger.error("Timeout getting remaining pipeline results")
                # Continue with partial results rather than failing completely
            except Exception as e:
                logger.error(f"Error getting remaining pipeline results: {e}", exc_info=True)

        response: dict[str, Any] = {"pipelines": pipeline_results}

        mode = self.combine_config.mode
        if mode == CombineMode.INDEPENDENT:
            return response

        combine_meta = self._combine_pipeline_outputs(pipeline_results)
        response["combine"] = combine_meta
        return response

    def _combine_pipeline_outputs(
        self, pipeline_results: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Union or join pipeline outputs according to combine_config."""
        dataset_map = {}
        for result in pipeline_results:
            output_path = result["results"].get("output_path")
            if not output_path:
                logger.warning(
                    "Pipeline %s did not produce an output_path; skipping in combine step",
                    result["name"],
                )
                continue
            # Check if output file exists before reading
            try:
                import os
                if not os.path.exists(output_path) and not output_path.startswith(("s3://", "gs://", "hdfs://")):
                    logger.warning(
                        "Output path %s does not exist for pipeline %s; skipping",
                        output_path,
                        result["name"],
                    )
                    continue
                dataset_map[result["name"]] = ray.data.read_parquet(output_path)
            except (OSError, IOError, ValueError) as e:
                logger.warning(
                    "Failed to read output from pipeline %s at %s: %s",
                    result["name"],
                    output_path,
                    e,
                )
                continue

        if not dataset_map:
            logger.warning("No datasets available for combine step")
            return {"mode": self.combine_config.mode.value, "datasets": []}

        mode = self.combine_config.mode
        combined_dataset = None

        if mode == CombineMode.UNION:
            datasets = list(dataset_map.values())
            combined_dataset = datasets[0] if len(datasets) == 1 else ray.data.union(*datasets)
        elif mode == CombineMode.JOIN:
            join_keys = self.combine_config.join_keys or []
            join_type = self.combine_config.join_type
            datasets_items = list(dataset_map.items())
            
            # Use GPU-accelerated join operations when available
            combined_dataset = datasets_items[0][1]
            for _name, ds in datasets_items[1:]:
                # Use GPU-accelerated join via custom map_batches
                try:
                    from pipeline.utils.gpu.joins import gpu_join_operator
                    
                    # CRITICAL: Do NOT materialize datasets with iter_batches() - breaks streaming!
                    # Ray Data's native join() is already optimized and streaming-compatible
                    # Custom GPU join would require materialization which defeats streaming purpose
                    # Use Ray Data's native join which handles streaming correctly
                    combined_dataset = combined_dataset.join(ds, on=join_keys, how=join_type)
                except ImportError:
                    # Fallback to standard Ray Data join
                    combined_dataset = combined_dataset.join(ds, on=join_keys, how=join_type)
        else:
            raise ValueError(f"Unsupported combine mode: {mode.value}")

        combine_meta: dict[str, Any] = {
            "mode": mode.value,
            "datasets": list(dataset_map.keys()),
        }

        combined_output = self.combine_config.output
        if combined_output:
            # Use compression for large-scale combined outputs
            combined_dataset.write_parquet(
                combined_output,
                compression="snappy",  # Fast compression
                num_rows_per_file=1000000,  # Optimize file sizes
            )
            combine_meta["output_path"] = combined_output

        if self.combine_config.collect_stats:
            # Avoid materialization - stats collection deferred
            # Users can read output files or query dataset if count needed
            combine_meta["stats_collected"] = False
            logger.info(
                "Stats collection skipped to avoid materialization. "
                "Read output dataset or files for row counts."
            )

        return combine_meta


@ray.remote(max_retries=2, num_cpus=1)  # Default resources, can be overridden
def _execute_pipeline_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """Ray task that executes a declarative pipeline spec."""
    from pipeline.api.declarative import Pipeline  # Local import to avoid cycle

    pipeline_args = spec["pipeline_args"]
    pipeline_name = spec["name"]

    pipeline = Pipeline(**pipeline_args)
    results = pipeline.run()
    results["pipeline_name"] = pipeline_name
    return {
        "name": pipeline_name,
        "results": results,
        "output_path": results.get("output_path"),
    }

