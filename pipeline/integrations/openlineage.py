"""OpenLineage integration for data lineage tracking.

Replaces custom DataLineageTracker with OpenLineage.
OpenLineage provides:
- Standard data lineage format
- Integration with Airflow, Spark, dbt, etc.
- Lineage visualization
- Data governance
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from pipeline.exceptions import TracingError
from pipeline.utils.decorators import handle_errors, log_execution_time

logger = logging.getLogger(__name__)

# Try to import OpenLineage
try:
    from openlineage.client import OpenLineageClient
    from openlineage.client.facet import (
        DataSourceDatasetFacet,
        SchemaDatasetFacet,
        SchemaField,
    )
    from openlineage.client.run import RunEvent, RunState, Run, Job, Dataset
    _OPENLINEAGE_AVAILABLE = True
except ImportError:
    _OPENLINEAGE_AVAILABLE = False
    logger.warning("OpenLineage not available. Install with: pip install openlineage-python")


class OpenLineageTracker:
    """OpenLineage integration for data lineage tracking.
    
    Replaces custom DataLineageTracker with OpenLineage standard.
    """

    def __init__(
        self,
        url: Optional[str] = None,
        namespace: str = "multimodal-pipeline",
        enabled: bool = True,
    ):
        """Initialize OpenLineage tracker.

        Args:
            url: OpenLineage backend URL (None = use environment variable)
            namespace: Namespace for lineage events
            enabled: Whether lineage tracking is enabled

        Raises:
            ValueError: If parameters are invalid
        """
        # Validate parameters
        if url is not None and not isinstance(url, str):
            raise ValueError(f"url must be str or None, got {type(url)}")
        if url is not None and not url.strip():
            raise ValueError("url cannot be empty")
        
        if not isinstance(namespace, str) or not namespace.strip():
            raise ValueError(f"namespace must be non-empty str, got {type(namespace)}")
        
        self.enabled = enabled and _OPENLINEAGE_AVAILABLE
        self.namespace = namespace

        if not self.enabled:
            if not _OPENLINEAGE_AVAILABLE:
                logger.warning("OpenLineage not available, lineage tracking disabled")
            return

        try:
            self.client = OpenLineageClient(url=url) if url else OpenLineageClient()
            logger.info(f"Initialized OpenLineage client (namespace: {namespace})")
        except Exception as e:
            logger.error(f"Failed to initialize OpenLineage client: {e}")
            self.enabled = False

    @log_execution_time
    @handle_errors(error_class=TracingError)
    def start_run(
        self,
        run_id: str,
        job_name: str,
        job_description: Optional[str] = None,
    ) -> None:
        """Start a lineage run.

        Args:
            run_id: Unique run identifier
            job_name: Name of the job/pipeline stage
            job_description: Optional job description

        Raises:
            ValueError: If parameters are invalid
            TracingError: If starting run fails
        """
        if not self.enabled:
            return

        # Validate parameters
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError(f"run_id must be non-empty str, got {type(run_id)}")
        
        if not isinstance(job_name, str) or not job_name.strip():
            raise ValueError(f"job_name must be non-empty str, got {type(job_name)}")
        
        if job_description is not None and not isinstance(job_description, str):
            raise ValueError(f"job_description must be str or None, got {type(job_description)}")

        try:
            event = RunEvent(
                eventType=RunState.START,
                eventTime=datetime.now().isoformat(),
                run=Run(runId=run_id),
                job=Job(
                    namespace=self.namespace,
                    name=job_name,
                    description=job_description,
                ),
            )
            self.client.emit(event)
            logger.debug(f"Emitted START event for run {run_id}")
        except Exception as e:
            logger.error(f"Failed to emit START event: {e}")
            raise TracingError(f"Failed to emit START event: {e}") from e

    @log_execution_time
    @handle_errors(error_class=TracingError)
    def complete_run(
        self,
        run_id: str,
        job_name: str,
        inputs: List[Dict[str, Any]],
        outputs: List[Dict[str, Any]],
    ) -> None:
        """Complete a lineage run with input/output datasets.

        Args:
            run_id: Unique run identifier
            job_name: Name of the job/pipeline stage
            inputs: List of input dataset metadata
            outputs: List of output dataset metadata

        Raises:
            ValueError: If parameters are invalid
            TracingError: If completing run fails
        """
        if not self.enabled:
            return

        # Validate parameters
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError(f"run_id must be non-empty str, got {type(run_id)}")
        
        if not isinstance(job_name, str) or not job_name.strip():
            raise ValueError(f"job_name must be non-empty str, got {type(job_name)}")
        
        if not isinstance(inputs, list):
            raise ValueError(f"inputs must be list, got {type(inputs)}")
        
        if not isinstance(outputs, list):
            raise ValueError(f"outputs must be list, got {type(outputs)}")

        try:
            # Convert inputs/outputs to OpenLineage Dataset format
            input_datasets = [
                self._create_dataset(dataset_info) for dataset_info in inputs
            ]
            output_datasets = [
                self._create_dataset(dataset_info) for dataset_info in outputs
            ]

            event = RunEvent(
                eventType=RunState.COMPLETE,
                eventTime=datetime.now().isoformat(),
                run=Run(runId=run_id),
                job=Job(namespace=self.namespace, name=job_name),
                inputs=input_datasets,
                outputs=output_datasets,
            )
            self.client.emit(event)
            logger.debug(f"Emitted COMPLETE event for run {run_id}")
        except Exception as e:
            logger.error(f"Failed to emit COMPLETE event: {e}")
            raise TracingError(f"Failed to emit COMPLETE event: {e}") from e

    @log_execution_time
    @handle_errors(error_class=TracingError)
    def fail_run(self, run_id: str, job_name: str, error: str) -> None:
        """Mark a lineage run as failed.

        Args:
            run_id: Unique run identifier
            job_name: Name of the job/pipeline stage
            error: Error message

        Raises:
            ValueError: If parameters are invalid
            TracingError: If failing run fails
        """
        if not self.enabled:
            return

        # Validate parameters
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError(f"run_id must be non-empty str, got {type(run_id)}")
        
        if not isinstance(job_name, str) or not job_name.strip():
            raise ValueError(f"job_name must be non-empty str, got {type(job_name)}")
        
        if not isinstance(error, str):
            raise ValueError(f"error must be str, got {type(error)}")

        try:
            event = RunEvent(
                eventType=RunState.FAIL,
                eventTime=datetime.now().isoformat(),
                run=Run(runId=run_id),
                job=Job(namespace=self.namespace, name=job_name),
            )
            self.client.emit(event)
            logger.debug(f"Emitted FAIL event for run {run_id}")
        except Exception as e:
            logger.error(f"Failed to emit FAIL event: {e}")
            raise TracingError(f"Failed to emit FAIL event: {e}") from e

    def _create_dataset(self, dataset_info: Dict[str, Any]) -> Dataset:
        """Create OpenLineage Dataset from metadata.

        Args:
            dataset_info: Dataset metadata dictionary

        Returns:
            OpenLineage Dataset object

        Raises:
            ValueError: If dataset_info is invalid
        """
        if not isinstance(dataset_info, dict):
            raise ValueError(f"dataset_info must be dict, got {type(dataset_info)}")
        
        name = dataset_info.get("name", "unknown")
        if not isinstance(name, str):
            name = str(name)
        
        source = dataset_info.get("source", dataset_info.get("source_path", "unknown"))
        if not isinstance(source, str):
            source = str(source)

        # Create dataset facets
        facets: Dict[str, Any] = {}
        
        if "schema" in dataset_info:
            schema_data = dataset_info["schema"]
            if isinstance(schema_data, list):
                try:
                    schema_fields = [
                        SchemaField(
                            name=field.get("name", "unknown") if isinstance(field, dict) else str(field),
                            type=field.get("type", "string") if isinstance(field, dict) else "string"
                        )
                        for field in schema_data
                    ]
                    facets["schema"] = SchemaDatasetFacet(fields=schema_fields)
                except Exception as e:
                    logger.warning(f"Failed to create schema facet: {e}")

        if source:
            try:
                facets["dataSource"] = DataSourceDatasetFacet(
                    name=source,
                    uri=source,
                )
            except Exception as e:
                logger.warning(f"Failed to create data source facet: {e}")

        return Dataset(
            namespace=self.namespace,
            name=name,
            facets=facets,
        )

    @log_execution_time
    @handle_errors(error_class=TracingError)
    def track_lineage(
        self,
        stage_name: str,
        input_paths: List[str],
        output_path: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Track data lineage for a pipeline stage.

        Args:
            stage_name: Name of the pipeline stage
            input_paths: List of input data paths
            output_path: Output data path
            metadata: Optional additional metadata

        Returns:
            Run ID for this lineage event

        Raises:
            ValueError: If parameters are invalid
            TracingError: If tracking fails
        """
        if not self.enabled:
            return ""

        # Validate parameters
        if not isinstance(stage_name, str) or not stage_name.strip():
            raise ValueError(f"stage_name must be non-empty str, got {type(stage_name)}")
        
        if not isinstance(input_paths, list):
            raise ValueError(f"input_paths must be list, got {type(input_paths)}")
        
        for i, path in enumerate(input_paths):
            if not isinstance(path, str) or not path.strip():
                raise ValueError(f"input_paths[{i}] must be non-empty str, got {type(path)}")
        
        if not isinstance(output_path, str) or not output_path.strip():
            raise ValueError(f"output_path must be non-empty str, got {type(output_path)}")
        
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError(f"metadata must be dict or None, got {type(metadata)}")

        run_id = str(uuid.uuid4())

        # Start run
        self.start_run(run_id=run_id, job_name=stage_name)

        # Create input/output datasets
        inputs = [
            {
                "name": f"input_{i}",
                "source": path,
                "source_path": path,
                "metadata": metadata or {},
            }
            for i, path in enumerate(input_paths)
        ]
        outputs = [
            {
                "name": "output",
                "source": output_path,
                "source_path": output_path,
                "metadata": metadata or {},
            }
        ]

        # Complete run
        self.complete_run(
            run_id=run_id,
            job_name=stage_name,
            inputs=inputs,
            outputs=outputs,
        )

        return run_id


def create_openlineage_tracker(
    url: Optional[str] = None,
    namespace: str = "multimodal-pipeline",
    enabled: bool = True,
) -> OpenLineageTracker:
    """Create an OpenLineage tracker instance.

    Args:
        url: OpenLineage backend URL
        namespace: Namespace for lineage events
        enabled: Whether tracking is enabled

    Returns:
        OpenLineageTracker instance
    """
    return OpenLineageTracker(
        url=url,
        namespace=namespace,
        enabled=enabled,
    )
