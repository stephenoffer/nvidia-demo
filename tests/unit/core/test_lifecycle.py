"""Unit tests for pipeline lifecycle management."""

import pytest
from unittest.mock import MagicMock, patch

from pipeline.config import PipelineConfig
from pipeline.core.lifecycle import PipelineLifecycleManager


class TestPipelineLifecycleManager:
    """Test pipeline lifecycle management."""

    def test_initialization(self, sample_config: PipelineConfig):
        """Test lifecycle manager initialization."""
        manager = PipelineLifecycleManager(sample_config)
        assert manager.config == sample_config

    @patch("pipeline.core.lifecycle.validate_inputs")
    def test_validate_inputs(self, mock_validate, sample_config: PipelineConfig):
        """Test input validation."""
        mock_validate.return_value = (True, [])
        manager = PipelineLifecycleManager(sample_config)
        manager._validate_inputs()
        mock_validate.assert_called_once()

    @patch("pipeline.core.lifecycle.validate_inputs")
    def test_validate_inputs_failure(self, mock_validate, sample_config: PipelineConfig):
        """Test input validation failure."""
        mock_validate.return_value = (False, ["Invalid path"])
        manager = PipelineLifecycleManager(sample_config)
        with pytest.raises(ValueError, match="Input validation failed"):
            manager._validate_inputs()

    @patch("pipeline.core.lifecycle.initialize_ray")
    def test_initialize_ray(self, mock_init_ray, sample_config: PipelineConfig):
        """Test Ray initialization."""
        mock_init_ray.return_value = {"initialized": True, "address": "local"}
        manager = PipelineLifecycleManager(sample_config)
        manager._initialize_ray()
        mock_init_ray.assert_called_once()

    @patch("pipeline.core.lifecycle.initialize_rapids_environment")
    def test_initialize_rapids(self, mock_init_rapids, sample_config: PipelineConfig):
        """Test RAPIDS initialization."""
        mock_init_rapids.return_value = {}
        sample_config.num_gpus = 1
        manager = PipelineLifecycleManager(sample_config)
        manager._initialize_rapids()
        mock_init_rapids.assert_called_once()

    @patch("pipeline.core.lifecycle.log_gpu_status")
    def test_log_gpu_status(self, mock_log_gpu, sample_config: PipelineConfig):
        """Test GPU status logging."""
        sample_config.num_gpus = 1
        manager = PipelineLifecycleManager(sample_config)
        manager._log_gpu_status()
        mock_log_gpu.assert_called_once()

    @patch("pipeline.core.lifecycle.clear_gpu_cache")
    @patch("pipeline.core.lifecycle.get_resource_manager")
    @patch("pipeline.core.lifecycle.shutdown_ray")
    def test_shutdown(
        self,
        mock_shutdown_ray,
        mock_resource_mgr,
        mock_clear_gpu,
        sample_config: PipelineConfig,
    ):
        """Test pipeline shutdown."""
        mock_resource_mgr.return_value.cleanup_all = MagicMock()
        manager = PipelineLifecycleManager(sample_config)
        manager.shutdown()
        mock_shutdown_ray.assert_called_once()

