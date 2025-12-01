"""Weights & Biases (W&B) integration for experiment tracking.

Alternative to MLflow for experiment tracking and metrics logging.
W&B provides:
- Experiment tracking
- Metrics visualization
- Model versioning
- Hyperparameter optimization
- Team collaboration
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from pipeline.exceptions import MetricsError
from pipeline.utils.decorators import handle_errors, log_execution_time

logger = logging.getLogger(__name__)

# Try to import W&B
try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False
    logger.warning("Weights & Biases not available. Install with: pip install wandb")


class WandBTracker:
    """Weights & Biases integration for experiment tracking.
    
    Alternative to MLflow for teams preferring W&B.
    """

    def __init__(
        self,
        project: str = "multimodal-pipeline",
        entity: Optional[str] = None,
        enabled: bool = True,
        **kwargs: Any,
    ):
        """Initialize W&B tracker.

        Args:
            project: W&B project name
            entity: W&B entity/team name (None = use default)
            enabled: Whether tracking is enabled
            **kwargs: Additional arguments passed to wandb.init()

        Raises:
            ValueError: If parameters are invalid
        """
        # Validate parameters
        if not isinstance(project, str) or not project.strip():
            raise ValueError(f"project must be non-empty str, got {type(project)}")
        
        if entity is not None and not isinstance(entity, str):
            raise ValueError(f"entity must be str or None, got {type(entity)}")
        if entity is not None and not entity.strip():
            raise ValueError("entity cannot be empty")
        
        self.enabled = enabled and _WANDB_AVAILABLE
        if not self.enabled:
            if not _WANDB_AVAILABLE:
                logger.warning("W&B not available, tracking disabled")
            return

        self.project = project
        self.entity = entity
        self.run: Optional[Any] = None
        self.init_kwargs = kwargs

    @log_execution_time
    @handle_errors(error_class=MetricsError)
    def start_run(
        self,
        run_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ) -> None:
        """Start a new W&B run.

        Args:
            run_name: Name for this run
            config: Configuration dictionary
            tags: List of tags

        Raises:
            ValueError: If parameters are invalid
            MetricsError: If starting run fails
        """
        if not self.enabled:
            return

        # Validate parameters
        if run_name is not None and not isinstance(run_name, str):
            raise ValueError(f"run_name must be str or None, got {type(run_name)}")
        
        if config is not None and not isinstance(config, dict):
            raise ValueError(f"config must be dict or None, got {type(config)}")
        
        if tags is not None:
            if not isinstance(tags, list):
                raise ValueError(f"tags must be list or None, got {type(tags)}")
            # Validate all tags are strings
            for i, tag in enumerate(tags):
                if not isinstance(tag, str):
                    raise ValueError(f"tags[{i}] must be str, got {type(tag)}")

        try:
            init_kwargs = {
                "project": self.project,
                "name": run_name,
                "tags": tags or [],
                "config": config or {},
                **self.init_kwargs,
            }
            if self.entity:
                init_kwargs["entity"] = self.entity

            self.run = wandb.init(**init_kwargs)
            if self.run:
                logger.info(f"Started W&B run: {self.run.id}")
            else:
                logger.warning("W&B run started but no run object returned")
                self.enabled = False
        except Exception as e:
            logger.error(f"Failed to start W&B run: {e}")
            self.enabled = False
            raise MetricsError(f"Failed to start W&B run: {e}") from e

    @log_execution_time
    @handle_errors(error_class=MetricsError)
    def end_run(self) -> None:
        """End the current W&B run.

        Raises:
            MetricsError: If ending run fails
        """
        if not self.enabled or self.run is None:
            return

        try:
            wandb.finish()
            logger.info("Ended W&B run")
            self.run = None
        except Exception as e:
            logger.error(f"Failed to end W&B run: {e}")
            raise MetricsError(f"Failed to end W&B run: {e}") from e

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
        if not self.enabled or self.run is None:
            return

        if not isinstance(params, dict):
            raise ValueError(f"params must be dict, got {type(params)}")
        
        if not params:
            logger.warning("Empty params dictionary, nothing to log")
            return

        try:
            wandb.config.update(params)
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
        if not self.enabled or self.run is None:
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
            if step is not None:
                if not isinstance(step, int) or step < 0:
                    logger.warning(f"Invalid step value: {step}, logging without step")
                    wandb.log(metrics)
                else:
                    wandb.log(metrics, step=step)
            else:
                wandb.log(metrics)
        except Exception as e:
            logger.error(f"Failed to log metrics: {e}")
            raise MetricsError(f"Failed to log metrics: {e}") from e

    @log_execution_time
    @handle_errors(error_class=MetricsError)
    def log_artifacts(self, local_dir: str, artifact_name: str) -> None:
        """Log artifacts (files/directories).

        Args:
            local_dir: Local directory to upload
            artifact_name: Name for the artifact

        Raises:
            ValueError: If parameters are invalid
            MetricsError: If logging fails
        """
        if not self.enabled or self.run is None:
            return

        if not isinstance(local_dir, str) or not local_dir.strip():
            raise ValueError(f"local_dir must be non-empty str, got {type(local_dir)}")
        
        if not isinstance(artifact_name, str) or not artifact_name.strip():
            raise ValueError(f"artifact_name must be non-empty str, got {type(artifact_name)}")
        
        import os
        if not os.path.exists(local_dir):
            raise ValueError(f"local_dir does not exist: {local_dir}")
        
        if not os.path.isdir(local_dir):
            raise ValueError(f"local_dir is not a directory: {local_dir}")

        try:
            artifact = wandb.Artifact(artifact_name, type="dataset")
            artifact.add_dir(local_dir)
            wandb.log_artifact(artifact)
        except Exception as e:
            logger.error(f"Failed to log artifacts: {e}")
            raise MetricsError(f"Failed to log artifacts: {e}") from e

    @log_execution_time
    @handle_errors(error_class=MetricsError)
    def log_table(
        self,
        table_name: str,
        data: List[Dict[str, Any]],
        columns: Optional[List[str]] = None,
    ) -> None:
        """Log a table for visualization.

        Args:
            table_name: Name of the table
            data: List of dictionaries representing rows
            columns: Optional list of column names

        Raises:
            ValueError: If parameters are invalid
            MetricsError: If logging fails
        """
        if not self.enabled or self.run is None:
            return

        if not isinstance(table_name, str) or not table_name.strip():
            raise ValueError(f"table_name must be non-empty str, got {type(table_name)}")
        
        if not isinstance(data, list):
            raise ValueError(f"data must be list, got {type(data)}")
        
        if columns is not None:
            if not isinstance(columns, list):
                raise ValueError(f"columns must be list or None, got {type(columns)}")
            for i, col in enumerate(columns):
                if not isinstance(col, str):
                    raise ValueError(f"columns[{i}] must be str, got {type(col)}")

        try:
            import pandas as pd
            df = pd.DataFrame(data, columns=columns)
            wandb.log({table_name: wandb.Table(dataframe=df)})
        except ImportError:
            raise MetricsError("pandas is required for log_table") from None
        except Exception as e:
            logger.error(f"Failed to log table: {e}")
            raise MetricsError(f"Failed to log table: {e}") from e


def create_wandb_tracker(
    project: str = "multimodal-pipeline",
    entity: Optional[str] = None,
    enabled: bool = True,
    **kwargs: Any,
) -> WandBTracker:
    """Create a W&B tracker instance.

    Args:
        project: W&B project name
        entity: W&B entity/team name
        enabled: Whether tracking is enabled
        **kwargs: Additional arguments for wandb.init()

    Returns:
        WandBTracker instance
    """
    return WandBTracker(
        project=project,
        entity=entity,
        enabled=enabled,
        **kwargs,
    )
