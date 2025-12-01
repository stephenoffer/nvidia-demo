"""MLflow integration for experiment tracking and model versioning.

Replaces custom metrics tracking and data versioning with MLflow.
MLflow provides:
- Experiment tracking
- Model versioning
- Metrics logging
- Artifact storage
- Model registry
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from pipeline.exceptions import MetricsError
from pipeline.utils.decorators import handle_errors, log_execution_time

logger = logging.getLogger(__name__)

# Try to import MLflow
try:
    import mlflow
    import mlflow.tracking
    from mlflow.tracking import MlflowClient
    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False
    logger.warning("MLflow not available. Install with: pip install mlflow")


class MLflowTracker:
    """MLflow integration for experiment tracking and model versioning.
    
    Replaces custom PipelineMetrics and DataVersionManager with MLflow.
    """

    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        experiment_name: str = "multimodal-pipeline",
        enabled: bool = True,
    ):
        """Initialize MLflow tracker.

        Args:
            tracking_uri: MLflow tracking URI (None = use default)
            experiment_name: Name of MLflow experiment
            enabled: Whether tracking is enabled

        Raises:
            ValueError: If parameters are invalid
        """
        # Validate parameters
        if tracking_uri is not None and not isinstance(tracking_uri, str):
            raise ValueError(f"tracking_uri must be str or None, got {type(tracking_uri)}")
        if tracking_uri is not None and not tracking_uri.strip():
            raise ValueError("tracking_uri cannot be empty")
        
        if not isinstance(experiment_name, str) or not experiment_name.strip():
            raise ValueError(f"experiment_name must be non-empty str, got {type(experiment_name)}")
        
        self.enabled = enabled and _MLFLOW_AVAILABLE
        if not self.enabled:
            if not _MLFLOW_AVAILABLE:
                logger.warning("MLflow not available, tracking disabled")
            return

        # Set tracking URI if provided
        if tracking_uri:
            try:
                mlflow.set_tracking_uri(tracking_uri)
            except Exception as e:
                logger.error(f"Failed to set MLflow tracking URI: {e}")
                self.enabled = False
                return

        # Set or create experiment
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(experiment_name)
                logger.info(f"Created MLflow experiment: {experiment_name}")
            else:
                experiment_id = experiment.experiment_id
            mlflow.set_experiment(experiment_name)
            self.experiment_id = experiment_id
        except Exception as e:
            logger.error(f"Failed to setup MLflow experiment: {e}")
            self.enabled = False
            return

        try:
            self.client = MlflowClient()
        except Exception as e:
            logger.error(f"Failed to create MLflow client: {e}")
            self.enabled = False
            return
        
        self.run_id: Optional[str] = None

    @log_execution_time
    @handle_errors(error_class=MetricsError)
    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None) -> None:
        """Start a new MLflow run.

        Args:
            run_name: Name for this run
            tags: Tags to add to the run

        Raises:
            MetricsError: If starting run fails
        """
        if not self.enabled:
            return

        # Validate tags
        if tags is not None:
            if not isinstance(tags, dict):
                raise ValueError(f"tags must be dict, got {type(tags)}")
            # Validate tag values are strings
            for key, value in tags.items():
                if not isinstance(key, str):
                    raise ValueError(f"Tag key must be str, got {type(key)}")
                if not isinstance(value, str):
                    raise ValueError(f"Tag value must be str, got {type(value)}")

        try:
            mlflow.start_run(run_name=run_name, tags=tags or {})
            active_run = mlflow.active_run()
            if active_run:
                self.run_id = active_run.info.run_id
                logger.info(f"Started MLflow run: {self.run_id}")
            else:
                logger.warning("MLflow run started but no active run found")
                self.enabled = False
        except Exception as e:
            logger.error(f"Failed to start MLflow run: {e}")
            self.enabled = False
            raise MetricsError(f"Failed to start MLflow run: {e}") from e

    @log_execution_time
    @handle_errors(error_class=MetricsError)
    def end_run(self, status: str = "FINISHED") -> None:
        """End the current MLflow run.

        Args:
            status: Run status (FINISHED, FAILED, KILLED)

        Raises:
            ValueError: If status is invalid
            MetricsError: If ending run fails
        """
        if not self.enabled or self.run_id is None:
            return

        # Validate status
        valid_statuses = {"FINISHED", "FAILED", "KILLED"}
        if status not in valid_statuses:
            raise ValueError(f"status must be one of {valid_statuses}, got {status}")

        try:
            mlflow.end_run(status=status)
            logger.info(f"Ended MLflow run: {self.run_id}")
            self.run_id = None
        except Exception as e:
            logger.error(f"Failed to end MLflow run: {e}")
            raise MetricsError(f"Failed to end MLflow run: {e}") from e

    @log_execution_time
    @handle_errors(error_class=MetricsError)
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log pipeline parameters.

        Args:
            params: Dictionary of parameters to log

        Raises:
            ValueError: If params is invalid
            MetricsError: If logging fails
        """
        if not self.enabled:
            return

        if not isinstance(params, dict):
            raise ValueError(f"params must be dict, got {type(params)}")
        
        if not params:
            logger.warning("Empty params dictionary, nothing to log")
            return

        try:
            for key, value in params.items():
                if not isinstance(key, str):
                    logger.warning(f"Skipping non-string param key: {type(key)}")
                    continue
                # Convert value to string for MLflow
                mlflow.log_param(key, str(value))
        except Exception as e:
            logger.error(f"Failed to log parameters: {e}")
            raise MetricsError(f"Failed to log parameters: {e}") from e

    @log_execution_time
    @handle_errors(error_class=MetricsError)
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log pipeline metrics.

        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number for time-series metrics

        Raises:
            ValueError: If metrics is invalid
            MetricsError: If logging fails
        """
        if not self.enabled:
            return

        if not isinstance(metrics, dict):
            raise ValueError(f"metrics must be dict, got {type(metrics)}")
        
        if not metrics:
            logger.warning("Empty metrics dictionary, nothing to log")
            return

        # Validate metric values are numeric
        for key, value in metrics.items():
            if not isinstance(key, str):
                logger.warning(f"Skipping non-string metric key: {type(key)}")
                continue
            if not isinstance(value, (int, float)):
                logger.warning(f"Skipping non-numeric metric value for {key}: {type(value)}")
                continue

        try:
            for key, value in metrics.items():
                if not isinstance(key, str) or not isinstance(value, (int, float)):
                    continue
                if step is not None:
                    if not isinstance(step, int) or step < 0:
                        logger.warning(f"Invalid step value: {step}")
                        mlflow.log_metric(key, float(value))
                    else:
                        mlflow.log_metric(key, float(value), step=step)
                else:
                    mlflow.log_metric(key, float(value))
        except Exception as e:
            logger.error(f"Failed to log metrics: {e}")
            raise MetricsError(f"Failed to log metrics: {e}") from e

    @log_execution_time
    @handle_errors(error_class=MetricsError)
    def log_artifacts(self, local_dir: str, artifact_path: Optional[str] = None) -> None:
        """Log artifacts (files/directories).

        Args:
            local_dir: Local directory to upload
            artifact_path: Optional path within artifact directory

        Raises:
            ValueError: If local_dir is invalid
            MetricsError: If logging fails
        """
        if not self.enabled:
            return

        if not isinstance(local_dir, str) or not local_dir.strip():
            raise ValueError(f"local_dir must be non-empty str, got {type(local_dir)}")
        
        import os
        if not os.path.exists(local_dir):
            raise ValueError(f"local_dir does not exist: {local_dir}")
        
        if not os.path.isdir(local_dir):
            raise ValueError(f"local_dir is not a directory: {local_dir}")

        try:
            mlflow.log_artifacts(local_dir, artifact_path=artifact_path)
        except Exception as e:
            logger.error(f"Failed to log artifacts: {e}")
            raise MetricsError(f"Failed to log artifacts: {e}") from e

    @log_execution_time
    @handle_errors(error_class=MetricsError)
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """Log a single artifact file.

        Args:
            local_path: Local file path
            artifact_path: Optional path within artifact directory

        Raises:
            ValueError: If local_path is invalid
            MetricsError: If logging fails
        """
        if not self.enabled:
            return

        if not isinstance(local_path, str) or not local_path.strip():
            raise ValueError(f"local_path must be non-empty str, got {type(local_path)}")
        
        import os
        if not os.path.exists(local_path):
            raise ValueError(f"local_path does not exist: {local_path}")
        
        if not os.path.isfile(local_path):
            raise ValueError(f"local_path is not a file: {local_path}")

        try:
            mlflow.log_artifact(local_path, artifact_path=artifact_path)
        except Exception as e:
            logger.error(f"Failed to log artifact: {e}")
            raise MetricsError(f"Failed to log artifact: {e}") from e

    @log_execution_time
    @handle_errors(error_class=MetricsError)
    def log_model(
        self,
        model_path: str,
        model_name: str,
        registered_model_name: Optional[str] = None,
    ) -> None:
        """Log and optionally register a model.

        Args:
            model_path: Path to model directory
            model_name: Name of the model
            registered_model_name: Optional name for model registry

        Raises:
            ValueError: If parameters are invalid
            MetricsError: If logging fails
        """
        if not self.enabled:
            return

        if not isinstance(model_path, str) or not model_path.strip():
            raise ValueError(f"model_path must be non-empty str, got {type(model_path)}")
        
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError(f"model_name must be non-empty str, got {type(model_name)}")
        
        import os
        if not os.path.exists(model_path):
            raise ValueError(f"model_path does not exist: {model_path}")
        
        if not os.path.isdir(model_path):
            raise ValueError(f"model_path is not a directory: {model_path}")

        try:
            # Log model as artifact
            mlflow.log_artifacts(model_path, artifact_path=model_name)

            # Register model if name provided
            if registered_model_name:
                if not isinstance(registered_model_name, str) or not registered_model_name.strip():
                    raise ValueError(f"registered_model_name must be non-empty str, got {type(registered_model_name)}")
                
                if self.run_id is None:
                    logger.warning("Cannot register model without active run")
                    return
                
                mlflow.register_model(
                    f"runs:/{self.run_id}/{model_name}",
                    registered_model_name,
                )
        except Exception as e:
            logger.error(f"Failed to log model: {e}")
            raise MetricsError(f"Failed to log model: {e}") from e

    @log_execution_time
    @handle_errors(error_class=MetricsError)
    def log_data_version(
        self,
        input_paths: list[str],
        output_path: str,
        dataset_hash: Optional[str] = None,
    ) -> None:
        """Log data version information.

        Args:
            input_paths: List of input data paths
            output_path: Output data path
            dataset_hash: Optional hash of dataset

        Raises:
            ValueError: If parameters are invalid
            MetricsError: If logging fails
        """
        if not self.enabled:
            return

        if not isinstance(input_paths, list):
            raise ValueError(f"input_paths must be list, got {type(input_paths)}")
        
        for i, path in enumerate(input_paths):
            if not isinstance(path, str) or not path.strip():
                raise ValueError(f"input_paths[{i}] must be non-empty str, got {type(path)}")
        
        if not isinstance(output_path, str) or not output_path.strip():
            raise ValueError(f"output_path must be non-empty str, got {type(output_path)}")
        
        if dataset_hash is not None and not isinstance(dataset_hash, str):
            raise ValueError(f"dataset_hash must be str or None, got {type(dataset_hash)}")

        try:
            # Log data paths as parameters
            mlflow.log_param("data.input_paths", ",".join(input_paths))
            mlflow.log_param("data.output_path", output_path)
            if dataset_hash:
                mlflow.log_param("data.hash", dataset_hash)
        except Exception as e:
            logger.error(f"Failed to log data version: {e}")
            raise MetricsError(f"Failed to log data version: {e}") from e


def create_mlflow_tracker(
    tracking_uri: Optional[str] = None,
    experiment_name: str = "multimodal-pipeline",
    enabled: bool = True,
) -> MLflowTracker:
    """Create an MLflow tracker instance.

    Args:
        tracking_uri: MLflow tracking URI
        experiment_name: Experiment name
        enabled: Whether tracking is enabled

    Returns:
        MLflowTracker instance
    """
    return MLflowTracker(
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        enabled=enabled,
    )
