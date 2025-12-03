"""Batch inference stage for AI models with MLOps integration.

Provides distributed batch inference using Ray Data with integration
to MLflow, model registries, and monitoring systems. Supports NVIDIA
optimizations including TensorRT, DALI, and cuDF.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import ray
from ray.data import Dataset
from ray.data import ActorPoolStrategy

from pipeline.stages.base import ProcessorBase
from pipeline.utils.constants import _DEFAULT_BATCH_SIZE

logger = logging.getLogger(__name__)


class BatchInferenceStage(ProcessorBase):
    """Perform batch inference using AI models with MLOps integration.

    Supports loading models from MLflow, local paths, or custom loaders.
    Integrates with model registries and monitoring systems. Optionally
    uses NVIDIA optimizations (TensorRT, DALI) for accelerated inference.

    Example:
        ```python
        # Simple usage
        inference = BatchInferenceStage(
            model_uri="models:/groot-model/Production",
            input_column="image",
        )
        
        # Advanced usage with NVIDIA optimizations
        inference = BatchInferenceStage(
            model_uri="models:/groot-model/Production",
            input_column="image",
            use_tensorrt=True,
            use_dali=True,
            ray_remote_args={"num_gpus": 1, "memory": 8_000_000_000},
        )
        ```
    """

    def __init__(
        self,
        model_uri: Optional[str] = None,
        model_loader: Optional[Callable[[], Any]] = None,
        input_column: str = "data",
        output_column: str = "predictions",
        batch_size: int = _DEFAULT_BATCH_SIZE,
        use_gpu: bool = True,
        num_gpus: int = 1,
        model_version: Optional[str] = None,
        mlflow_tracking_uri: Optional[str] = None,
        log_predictions: bool = True,
        prediction_metadata: Optional[dict[str, Any]] = None,
        # NVIDIA optimizations
        use_tensorrt: bool = False,
        use_dali: bool = False,
        tensorrt_precision: str = "fp16",
        # Ray Data options
        ray_remote_args: Optional[dict[str, Any]] = None,
        batch_format: Optional[str] = None,
        concurrency: Optional[int] = None,
        **map_batches_kwargs: Any,
    ):
        """Initialize batch inference stage.

        Args:
            model_uri: MLflow model URI or local model path
            model_loader: Custom model loader function (takes precedence over model_uri)
            input_column: Column name containing input data
            output_column: Column name for predictions
            batch_size: Batch size for inference
            use_gpu: Whether to use GPU for inference
            num_gpus: Number of GPUs per worker
            model_version: Model version to load (for MLflow)
            mlflow_tracking_uri: MLflow tracking URI
            log_predictions: Whether to log predictions to MLflow
            prediction_metadata: Additional metadata to log with predictions
            use_tensorrt: Use NVIDIA TensorRT for optimized inference
            use_dali: Use NVIDIA DALI for data loading acceleration
            tensorrt_precision: TensorRT precision ("fp32", "fp16", "int8")
            ray_remote_args: Additional Ray remote arguments (num_cpus, memory, etc.)
            batch_format: Batch format for map_batches ("pandas", "pyarrow", "numpy")
            concurrency: Maximum number of concurrent batches
            **map_batches_kwargs: Additional kwargs passed to map_batches
        """
        super().__init__(batch_size=batch_size)
        self.model_uri = model_uri
        self.model_loader = model_loader
        self.input_column = input_column
        self.output_column = output_column
        self.use_gpu = use_gpu
        self.num_gpus = num_gpus
        self.model_version = model_version
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.log_predictions = log_predictions
        self.prediction_metadata = prediction_metadata or {}
        self.use_tensorrt = use_tensorrt
        self.use_dali = use_dali
        self.tensorrt_precision = tensorrt_precision
        self.ray_remote_args = ray_remote_args or {}
        self.batch_format = batch_format
        self.concurrency = concurrency
        self.map_batches_kwargs = map_batches_kwargs
        self._model = None

    def process(self, dataset: Dataset) -> Dataset:
        """Run batch inference on dataset.

        Args:
            dataset: Input Ray Dataset

        Returns:
            Dataset with predictions added

        Raises:
            ValueError: If model_uri/model_loader not provided or input_column missing
            RuntimeError: If model loading fails
        """
        logger.info(f"Running batch inference on {self.input_column} column")
        
        if not self.model_loader and not self.model_uri:
            raise ValueError("Either model_uri or model_loader must be provided")
        
        # Validate input column exists
        try:
            sample = dataset.take(1)
            if sample and self.input_column not in sample[0]:
                raise ValueError(f"Input column '{self.input_column}' not found in dataset. Available columns: {list(sample[0].keys())}")
        except Exception:
            logger.debug("Could not validate input column, proceeding anyway")
        
        # Load model lazily (per worker)
        model = None
        
        def inference_batch(batch: dict[str, Any]) -> dict[str, Any]:
            """Process batch through model."""
            nonlocal model
            
            # Lazy model loading per worker
            if model is None:
                if self.model_loader:
                    model = self.model_loader()
                elif self.model_uri:
                    model = self._load_model()
                else:
                    raise ValueError("Either model_uri or model_loader must be provided")
                
                if self.use_tensorrt:
                    model = self._optimize_with_tensorrt(model)
            
            if self.input_column not in batch:
                raise ValueError(f"Input column '{self.input_column}' not found in batch")
            
            inputs = batch[self.input_column]
            
            if inputs is None:
                logger.warning("Input column contains None values")
                batch[self.output_column] = None
                return batch
            
            if self.use_dali:
                inputs = self._preprocess_with_dali(inputs)
            
            try:
                if hasattr(model, 'predict'):
                    predictions = model.predict(inputs)
                elif hasattr(model, '__call__'):
                    predictions = model(inputs)
                else:
                    raise ValueError("Model must have predict() method or be callable")
                
                batch[self.output_column] = predictions
                
                if self.log_predictions:
                    self._log_predictions(batch, predictions)
            except Exception as e:
                logger.error(f"Inference failed for batch: {e}")
                batch[self.output_column] = None
                batch[f"{self.output_column}_error"] = str(e)
            
            return batch

        # Build ray_remote_args with defaults
        remote_args = {
            "num_gpus": self.num_gpus if self.use_gpu else 0,
            **self.ray_remote_args,
        }

        # Build map_batches kwargs
        map_kwargs = {
            "batch_size": self.batch_size,
            "ray_remote_args": remote_args,
            **self.map_batches_kwargs,
        }
        
        if self.batch_format:
            map_kwargs["batch_format"] = self.batch_format
        
        if self.concurrency:
            map_kwargs["concurrency"] = self.concurrency
        
        if self.use_gpu and self.num_gpus > 0:
            pool_size = min(4, max(1, self.num_gpus))
            map_kwargs["compute"] = ActorPoolStrategy(size=pool_size)

        return dataset.map_batches(inference_batch, **map_kwargs)

    def _load_model(self) -> Any:
        """Load model from MLflow or local path."""
        if not self.model_uri:
            raise ValueError("model_uri is required")
        
        if self.model_uri.startswith("models:/") or self.model_uri.startswith("runs:/"):
            try:
                import mlflow
            except ImportError:
                raise ImportError("MLflow is required for MLflow model URIs. Install with: pip install mlflow")
            
            if self.mlflow_tracking_uri:
                mlflow.set_tracking_uri(self.mlflow_tracking_uri)
            
            if self.model_version:
                model_uri = f"{self.model_uri}/{self.model_version}"
            else:
                model_uri = self.model_uri
            
            logger.info(f"Loading model from MLflow: {model_uri}")
            try:
                return mlflow.pyfunc.load_model(model_uri)
            except Exception as e:
                raise RuntimeError(f"Failed to load model from MLflow {model_uri}: {e}") from e
        else:
            import os
            
            if not os.path.exists(self.model_uri):
                raise FileNotFoundError(f"Model path does not exist: {self.model_uri}")
            
            logger.info(f"Loading model from local path: {self.model_uri}")
            
            # Try PyTorch first
            try:
                import torch
                if self.use_gpu and torch.cuda.is_available():
                    return torch.load(self.model_uri, map_location="cuda")
                else:
                    return torch.load(self.model_uri, map_location="cpu")
            except Exception:
                pass
            
            # Fallback to pickle
            try:
                import pickle
                with open(self.model_uri, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                raise RuntimeError(f"Failed to load model from {self.model_uri}: {e}") from e

    def _optimize_with_tensorrt(self, model: Any) -> Any:
        """Optimize model with NVIDIA TensorRT."""
        try:
            import tensorrt as trt
            
            logger.info(f"Optimizing model with TensorRT (precision: {self.tensorrt_precision})")
            
            if hasattr(model, 'to_tensorrt'):
                return model.to_tensorrt(precision=self.tensorrt_precision)
            else:
                logger.warning("Model does not support TensorRT optimization, using original model")
                return model
        except ImportError:
            logger.warning("TensorRT not available, using original model")
            return model
        except Exception as e:
            logger.warning(f"TensorRT optimization failed: {e}, using original model")
            return model

    def _preprocess_with_dali(self, inputs: Any) -> Any:
        """Preprocess inputs using NVIDIA DALI."""
        try:
            import nvidia.dali as dali
            
            logger.debug("Preprocessing with NVIDIA DALI")
            
            if isinstance(inputs, list):
                pipeline = dali.Pipeline(batch_size=len(inputs), num_threads=2, device_id=0)
                with pipeline:
                    images = dali.fn.external_source(source=inputs, device="gpu")
                    images = dali.fn.resize(images, size=(224, 224))
                    pipeline.set_outputs(images)
                pipeline.build()
                return pipeline.run()
            else:
                return inputs
        except ImportError:
            logger.debug("DALI not available, skipping preprocessing")
            return inputs
        except Exception as e:
            logger.warning(f"DALI preprocessing failed: {e}, using original inputs")
            return inputs

    def _log_predictions(self, batch: dict[str, Any], predictions: Any) -> None:
        """Log predictions to MLflow."""
        try:
            import mlflow
            
            if self.mlflow_tracking_uri:
                mlflow.set_tracking_uri(self.mlflow_tracking_uri)
            
            metadata = {
                **self.prediction_metadata,
                "batch_size": len(batch),
                "model_uri": self.model_uri,
            }
            mlflow.log_dict(metadata, "prediction_metadata.json")
        except ImportError:
            logger.debug("MLflow not available, skipping prediction logging")
        except Exception as e:
            logger.warning(f"Failed to log predictions: {e}")

