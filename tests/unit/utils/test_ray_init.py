"""Unit tests for Ray initialization utilities."""

import pytest
from unittest.mock import MagicMock, patch

from pipeline.utils.ray.init import get_ray_cluster_info, initialize_ray, shutdown_ray


class TestRayInit:
    """Test Ray initialization utilities."""

    @patch("ray.init")
    def test_initialize_ray(self, mock_ray_init):
        """Test Ray initialization."""
        mock_ray_init.return_value = None
        result = initialize_ray(num_cpus=2, num_gpus=0)
        assert result["initialized"]
        mock_ray_init.assert_called_once()

    @patch("ray.is_initialized")
    def test_initialize_ray_already_initialized(self, mock_is_init):
        """Test Ray initialization when already initialized."""
        mock_is_init.return_value = True
        result = initialize_ray(num_cpus=2, num_gpus=0)
        assert result["initialized"]

    @patch("ray.shutdown")
    def test_shutdown_ray(self, mock_shutdown):
        """Test Ray shutdown."""
        shutdown_ray(graceful=True, timeout=30.0)
        mock_shutdown.assert_called_once()

    @patch("ray.is_initialized")
    @patch("ray.util.client.ray.get_runtime_context")
    def test_get_ray_cluster_info(self, mock_context, mock_is_init):
        """Test getting Ray cluster info."""
        mock_is_init.return_value = True
        mock_context.return_value.address_info = {"address": "localhost:10001"}
        
        info = get_ray_cluster_info()
        assert "address" in info

