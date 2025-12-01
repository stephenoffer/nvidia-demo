"""HTTP health check server for Kubernetes liveness/readiness probes.

Provides production-ready health check endpoints for container orchestration.
"""

from __future__ import annotations

import logging
import os
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional

from pipeline.health import HealthChecker

logger = logging.getLogger(__name__)


class HealthCheckHandler(BaseHTTPRequestHandler):
    """HTTP request handler for health check endpoints."""

    def log_message(self, format: str, *args: object) -> None:
        """Override to use our logger instead of stderr."""
        logger.debug(f"{self.address_string()} - {format % args}")

    def do_GET(self) -> None:
        """Handle GET requests for health checks."""
        if self.path == "/health":
            self._handle_health()
        elif self.path == "/ready":
            self._handle_ready()
        elif self.path == "/live":
            self._handle_live()
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not Found")

    def _handle_health(self) -> None:
        """Handle /health endpoint - comprehensive health check."""
        try:
            output_path = os.getenv("OUTPUT_PATH")
            health = HealthChecker.get_overall_health(output_path=output_path)

            if health["healthy"]:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                import json

                self.wfile.write(json.dumps(health).encode())
            else:
                self.send_response(503)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                import json

                self.wfile.write(json.dumps(health).encode())
        except Exception as e:
            logger.error(f"Error in health check: {e}", exc_info=True)
            self.send_response(500)
            self.end_headers()
            self.wfile.write(f'{{"error": "{str(e)}"}}'.encode())

    def _handle_ready(self) -> None:
        """Handle /ready endpoint - readiness probe."""
        try:
            # Check if Ray cluster is initialized and ready
            ray_health = HealthChecker.check_ray_cluster()
            if ray_health.get("healthy", False):
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                import json

                self.wfile.write(json.dumps({"ready": True}).encode())
            else:
                self.send_response(503)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                import json

                self.wfile.write(json.dumps({"ready": False, "reason": ray_health.get("status")}).encode())
        except Exception as e:
            logger.error(f"Error in readiness check: {e}", exc_info=True)
            self.send_response(500)
            self.end_headers()
            self.wfile.write(f'{{"error": "{str(e)}"}}'.encode())

    def _handle_live(self) -> None:
        """Handle /live endpoint - liveness probe."""
        # Simple liveness check - just verify process is running
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        import json

        self.wfile.write(json.dumps({"alive": True}).encode())


class HealthCheckServer:
    """HTTP server for health checks."""

    def __init__(self, port: int = 8080, host: str = "0.0.0.0"):
        """Initialize health check server.

        Args:
            port: Port to listen on
            host: Host to bind to
        """
        self.port = port
        self.host = host
        self.server: Optional[HTTPServer] = None
        self.thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start health check server in background thread."""
        self.server = HTTPServer((self.host, self.port), HealthCheckHandler)

        def run_server() -> None:
            logger.info(f"Health check server starting on {self.host}:{self.port}")
            self.server.serve_forever()

        self.thread = threading.Thread(target=run_server, daemon=True)
        self.thread.start()
        logger.info("Health check server started")

    def stop(self) -> None:
        """Stop health check server."""
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            logger.info("Health check server stopped")

    def wait(self) -> None:
        """Wait for server thread to finish."""
        if self.thread:
            self.thread.join()


def start_health_server(port: int = 8080, host: str = "0.0.0.0") -> HealthCheckServer:
    """Start health check server.

    Args:
        port: Port to listen on
        host: Host to bind to

    Returns:
        HealthCheckServer instance
    """
    server = HealthCheckServer(port=port, host=host)
    server.start()

    # Register signal handlers for graceful shutdown
    def signal_handler(signum: int, frame: object) -> None:
        logger.info(f"Received signal {signum}, shutting down health server")
        server.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    return server


if __name__ == "__main__":
    # Standalone health check server
    logging.basicConfig(level=logging.INFO)
    port = int(os.getenv("HEALTH_CHECK_PORT", "8080"))
    server = start_health_server(port=port)
    try:
        server.wait()
    except KeyboardInterrupt:
        server.stop()

