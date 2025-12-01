"""Command-line interface for the pipeline."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from pipeline.health import HealthChecker
from pipeline.server import start_health_server

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration.

    Args:
        verbose: Enable verbose logging
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def health_check(args: argparse.Namespace) -> int:
    """Run health check.

    Args:
        args: Command line arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    output_path = args.output_path if hasattr(args, "output_path") else None
    health = HealthChecker.get_overall_health(output_path=output_path)

    if health["healthy"]:
        logger.info("Pipeline is healthy")
        return 0
    else:
        logger.error("Pipeline is unhealthy")
        import json

        logger.error(f"Health check details: {json.dumps(health, indent=2)}")
        return 1


def start_server(args: argparse.Namespace) -> int:
    """Start health check server.

    Args:
        args: Command line arguments

    Returns:
        Exit code
    """
    port = args.port if hasattr(args, "port") else 8080
    host = args.host if hasattr(args, "host") else "0.0.0.0"

    logger.info(f"Starting health check server on {host}:{port}")
    server = start_health_server(port=port, host=host)

    try:
        server.wait()
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
        server.stop()
        return 0


def main() -> int:
    """Main CLI entry point.

    Returns:
        Exit code
    """
    parser = argparse.ArgumentParser(
        description="Multimodal Data Pipeline CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Health check command
    health_parser = subparsers.add_parser("health", help="Check pipeline health")
    health_parser.add_argument(
        "--output-path",
        type=str,
        help="Output path for disk space check",
    )

    # Server command
    server_parser = subparsers.add_parser("server", help="Start health check server")
    server_parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to listen on (default: 8080)",
    )
    server_parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )

    args = parser.parse_args()

    setup_logging(verbose=args.verbose)

    if args.command == "health":
        return health_check(args)
    elif args.command == "server":
        return start_server(args)
    elif args.command is None:
        parser.print_help()
        return 1
    else:
        parser.print_help()
        return 1


def health_check_cli() -> int:
    """CLI entry point for health check command."""
    parser = argparse.ArgumentParser(description="Check pipeline health")
    parser.add_argument("--output-path", type=str, help="Output path for disk space check")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()
    setup_logging(verbose=args.verbose)
    return health_check(args)


def start_server_cli() -> int:
    """CLI entry point for server command."""
    parser = argparse.ArgumentParser(description="Start health check server")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()
    setup_logging(verbose=args.verbose)
    return start_server(args)


if __name__ == "__main__":
    sys.exit(main())

