"""Performance profiling utilities for pipeline stages.

Provides cProfile and py-spy integration for identifying bottlenecks.
Critical for GR00T: Large-scale processing requires performance optimization.
"""

from __future__ import annotations

import cProfile
import logging
import pstats
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class StageProfiler:
    """Profiler for individual pipeline stages.

    Uses cProfile to collect detailed performance statistics for each stage.
    """

    def __init__(
        self,
        output_dir: str = "profiles",
        enable_profiling: bool = True,
        sort_by: str = "cumulative",
    ):
        """Initialize stage profiler.

        Args:
            output_dir: Directory to save profile reports
            enable_profiling: Whether to enable profiling
            sort_by: Sort key for profile stats (cumulative, time, calls)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.enable_profiling = enable_profiling
        self.sort_by = sort_by
        self.profiles: Dict[str, cProfile.Profile] = {}

    @contextmanager
    def profile_stage(self, stage_name: str):
        """Context manager for profiling a pipeline stage.

        Args:
            stage_name: Name of the stage being profiled

        Yields:
            Profile context
        """
        if not self.enable_profiling:
            yield
            return

        profiler = cProfile.Profile()
        profiler.enable()

        start_time = time.time()
        try:
            yield profiler
        finally:
            profiler.disable()
            end_time = time.time()

            # Save profile
            self.profiles[stage_name] = profiler
            self._save_profile(stage_name, profiler, end_time - start_time)

    def _save_profile(
        self, stage_name: str, profiler: cProfile.Profile, duration: float
    ) -> None:
        """Save profile statistics to file.

        Args:
            stage_name: Name of the stage
            profiler: cProfile.Profile instance
            duration: Stage duration in seconds
        """
        try:
            # Save stats to file
            stats_file = self.output_dir / f"{stage_name}_profile.txt"
            with open(stats_file, "w") as f:
                stats = pstats.Stats(profiler, stream=f)
                stats.sort_stats(self.sort_by)
                stats.print_stats(50)  # Top 50 functions
                f.write(f"\n\nStage Duration: {duration:.2f} seconds\n")

            # Save binary profile for later analysis
            binary_file = self.output_dir / f"{stage_name}_profile.prof"
            profiler.dump_stats(str(binary_file))

            logger.info(
                f"Saved profile for {stage_name} to {stats_file} "
                f"(duration: {duration:.2f}s)"
            )
        except (IOError, OSError) as e:
            logger.error(f"Failed to save profile for {stage_name}: {e}")

    def get_profile_summary(self, stage_name: str) -> Dict[str, Any]:
        """Get summary statistics for a stage profile.

        Args:
            stage_name: Name of the stage

        Returns:
            Dictionary with profile summary
        """
        if stage_name not in self.profiles:
            return {}

        profiler = self.profiles[stage_name]
        stats = pstats.Stats(profiler)

        # Get top functions by cumulative time
        stats.sort_stats("cumulative")
        top_functions = []
        for func, (cc, nc, tt, ct, callers) in stats.stats.items():
            top_functions.append(
                {
                    "function": f"{func[0]}:{func[1]}({func[2]})",
                    "cumulative_time": ct,
                    "total_time": tt,
                    "call_count": nc,
                }
            )
            if len(top_functions) >= 10:
                break

        return {
            "stage_name": stage_name,
            "total_calls": stats.total_calls,
            "total_time": stats.total_tt,
            "top_functions": top_functions,
        }

    def generate_report(self, output_file: Optional[str] = None) -> str:
        """Generate comprehensive profiling report.

        Args:
            output_file: Path to save report (None = auto-generate)

        Returns:
            Path to generated report
        """
        if not self.profiles:
            logger.warning("No profiles to report")
            return ""

        if output_file is None:
            output_file = str(self.output_dir / "profiling_report.txt")

        try:
            with open(output_file, "w") as f:
                f.write("Pipeline Performance Profiling Report\n")
                f.write("=" * 60 + "\n\n")

                for stage_name in sorted(self.profiles.keys()):
                    summary = self.get_profile_summary(stage_name)
                    f.write(f"Stage: {stage_name}\n")
                    f.write("-" * 60 + "\n")
                    f.write(f"Total Calls: {summary.get('total_calls', 0)}\n")
                    f.write(f"Total Time: {summary.get('total_time', 0):.2f}s\n")
                    f.write("\nTop Functions:\n")
                    for func_info in summary.get("top_functions", [])[:5]:
                        f.write(
                            f"  {func_info['function']}: "
                            f"{func_info['cumulative_time']:.2f}s "
                            f"({func_info['call_count']} calls)\n"
                        )
                    f.write("\n")

            logger.info(f"Generated profiling report: {output_file}")
            return output_file
        except (IOError, OSError) as e:
            logger.error(f"Failed to generate profiling report: {e}")
            return ""


class PerformanceMonitor:
    """Monitor performance metrics for pipeline execution.

    Tracks timing, memory usage, and GPU utilization.
    """

    def __init__(self, enabled: bool = True):
        """Initialize performance monitor.

        Args:
            enabled: Whether monitoring is enabled
        """
        self.enabled = enabled
        self.metrics: Dict[str, Dict[str, Any]] = {}

    @contextmanager
    def monitor_stage(self, stage_name: str):
        """Context manager for monitoring a pipeline stage.

        Args:
            stage_name: Name of the stage being monitored

        Yields:
            Monitor context
        """
        if not self.enabled:
            yield
            return

        start_time = time.time()
        start_memory = self._get_memory_usage()

        try:
            yield
        finally:
            end_time = time.time()
            end_memory = self._get_memory_usage()

            self.metrics[stage_name] = {
                "duration": end_time - start_time,
                "memory_delta": end_memory - start_memory,
                "memory_end": end_memory,
            }

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB.

        Returns:
            Memory usage in megabytes
        """
        try:
            import psutil

            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # Convert to MB
        except ImportError:
            logger.debug("psutil not available, skipping memory monitoring")
            return 0.0

    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary.

        Returns:
            Dictionary with performance summary
        """
        if not self.metrics:
            return {}

        total_duration = sum(m["duration"] for m in self.metrics.values())
        max_memory = max(m.get("memory_end", 0) for m in self.metrics.values())

        return {
            "total_duration": total_duration,
            "max_memory_mb": max_memory,
            "stages": self.metrics,
        }


def create_profiler(
    output_dir: str = "profiles", enable_profiling: bool = True
) -> StageProfiler:
    """Create a stage profiler instance.

    Args:
        output_dir: Directory to save profiles
        enable_profiling: Whether to enable profiling

    Returns:
        StageProfiler instance
    """
    return StageProfiler(output_dir=output_dir, enable_profiling=enable_profiling)


def create_monitor(enabled: bool = True) -> PerformanceMonitor:
    """Create a performance monitor instance.

    Args:
        enabled: Whether monitoring is enabled

    Returns:
        PerformanceMonitor instance
    """
    return PerformanceMonitor(enabled=enabled)

