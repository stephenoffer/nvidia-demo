"""Unit tests for pipeline execution."""

import pytest
from unittest.mock import MagicMock, patch

from pipeline.config import PipelineConfig
from pipeline.core.execution import PipelineExecutor
from pipeline.observability.metrics import PipelineMetrics


class TestPipelineExecutor:
    """Test pipeline execution logic."""

    def test_initialization(self, sample_config: PipelineConfig):
        """Test executor initialization."""
        metrics = PipelineMetrics(enabled=False)
        executor = PipelineExecutor(sample_config, metrics)
        assert executor.config == sample_config
        assert executor.metrics == metrics

    def test_execute_stage(self, sample_config: PipelineConfig, sample_dataset):
        """Test single stage execution."""
        metrics = PipelineMetrics(enabled=False)
        executor = PipelineExecutor(sample_config, metrics)
        
        mock_stage = MagicMock()
        mock_stage.process.return_value = sample_dataset
        
        result = executor._execute_stage(sample_dataset, mock_stage, 0)
        assert result == sample_dataset
        mock_stage.process.assert_called_once_with(sample_dataset)

    def test_perform_periodic_cleanup(self, sample_config: PipelineConfig):
        """Test periodic cleanup."""
        metrics = PipelineMetrics(enabled=False)
        executor = PipelineExecutor(sample_config, metrics)
        
        with patch("gc.collect") as mock_gc:
            executor._perform_periodic_cleanup(3)
            mock_gc.assert_called_once()

    @patch("pipeline.core.execution.create_checkpoint_manager")
    def test_initialize_checkpointing(self, mock_create_checkpoint, sample_config: PipelineConfig):
        """Test checkpoint initialization."""
        sample_config.checkpoint_interval = 10
        metrics = PipelineMetrics(enabled=False)
        executor = PipelineExecutor(sample_config, metrics)
        
        mock_checkpoint = MagicMock()
        mock_create_checkpoint.return_value = mock_checkpoint
        
        result = executor._initialize_checkpointing()
        assert result == mock_checkpoint
        mock_create_checkpoint.assert_called_once()

    def test_initialize_checkpointing_disabled(self, sample_config: PipelineConfig):
        """Test checkpoint initialization when disabled."""
        sample_config.checkpoint_interval = 0
        metrics = PipelineMetrics(enabled=False)
        executor = PipelineExecutor(sample_config, metrics)
        
        result = executor._initialize_checkpointing()
        assert result is None

    @patch("pipeline.core.execution.validate_path")
    def test_validate_output_path(self, mock_validate, sample_config: PipelineConfig):
        """Test output path validation."""
        metrics = PipelineMetrics(enabled=False)
        executor = PipelineExecutor(sample_config, metrics)
        
        executor._validate_output_path()
        mock_validate.assert_called_once()

    @patch("pipeline.core.execution.check_disk_space")
    def test_check_disk_space(self, mock_check_space, sample_config: PipelineConfig):
        """Test disk space check."""
        mock_check_space.return_value = (True, 1000000)
        metrics = PipelineMetrics(enabled=False)
        executor = PipelineExecutor(sample_config, metrics)
        
        executor._check_disk_space()
        mock_check_space.assert_called_once()

