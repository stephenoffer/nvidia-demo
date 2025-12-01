"""Simulation data renderer for visualization and demo purposes.

Supports both headless rendering for large-scale ETL and interactive
rendering for laptop-scale demos.

Uses matplotlib for visualization and OpenCV for video output.
See: https://matplotlib.org/ and https://opencv.org/
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np  # https://numpy.org/

logger = logging.getLogger(__name__)


class SimulationRenderer:
    """Renderer for simulation trajectories.

    Can operate in two modes:
    - Headless: Fast rendering for large-scale data processing
    - Interactive: Visual rendering for demos and debugging
    """

    def __init__(
        self,
        mode: str = "headless",
        resolution: Tuple[int, int] = (1280, 720),
        fps: int = 30,
        enable_gui: bool = False,
    ):
        """Initialize simulation renderer.

        Args:
            mode: Rendering mode ('headless' or 'interactive')
            resolution: Output resolution (width, height)
            fps: Frames per second for video output
            enable_gui: Whether to enable GUI (only for interactive mode)
        """
        self.mode = mode
        self.resolution = resolution
        self.fps = fps
        self.enable_gui = enable_gui and mode == "interactive"

        # Initialize rendering backend based on mode
        self._init_backend()

    def _init_backend(self) -> None:
        """Initialize rendering backend."""
        if self.mode == "headless":
            # Use lightweight rendering for scale
            try:
                import matplotlib

                matplotlib.use("Agg")  # Non-interactive backend
                self.backend = "matplotlib"
            except ImportError:
                logger.warning("Matplotlib not available, using basic rendering")
                self.backend = "basic"
        else:
            # Interactive mode - can use more advanced rendering
            try:
                import matplotlib.pyplot  # noqa: F401

                self.backend = "matplotlib"
            except ImportError:
                self.backend = "basic"

    def render_trajectory(
        self,
        trajectory: Dict[str, Any],
        output_path: Optional[str] = None,
        show: bool = False,
    ) -> np.ndarray:
        """Render a single trajectory.

        Args:
            trajectory: Trajectory data with joint positions, observations, etc.
            output_path: Optional path to save rendered image/video
            show: Whether to display (only in interactive mode)

        Returns:
            Rendered frames as numpy array
        """
        if self.backend == "matplotlib":
            return self._render_matplotlib(trajectory, output_path, show)
        else:
            return self._render_basic(trajectory, output_path, show)

    def _render_matplotlib(
        self,
        trajectory: Dict[str, Any],
        output_path: Optional[str],
        show: bool,
    ) -> np.ndarray:
        """Render using matplotlib."""
        import matplotlib.pyplot as plt  # https://matplotlib.org/

        # Extract trajectory data
        joint_positions = trajectory.get("joint_positions", [])
        if isinstance(joint_positions, list):
            joint_positions = np.array(joint_positions)

        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle("Robot Trajectory Visualization", fontsize=14)

        frames = []
        num_steps = len(joint_positions) if len(joint_positions.shape) > 1 else 1

        for step in range(num_steps):
            # Clear axes
            for ax in axes.flat:
                ax.clear()

            # Plot joint positions over time
            if num_steps > 1:
                axes[0, 0].plot(joint_positions[: step + 1, :])
                axes[0, 0].set_title("Joint Positions")
                axes[0, 0].set_xlabel("Time Step")
                axes[0, 0].set_ylabel("Joint Angle")
                axes[0, 0].grid(True)

            # Plot current joint positions
            if num_steps == 1:
                current_pos = joint_positions
            else:
                current_pos = joint_positions[step]

            axes[0, 1].bar(range(len(current_pos)), current_pos)
            axes[0, 1].set_title("Current Joint Positions")
            axes[0, 1].set_xlabel("Joint Index")
            axes[0, 1].set_ylabel("Angle")

            # Plot trajectory in 2D (if base pose available)
            if "base_pose" in trajectory:
                base_poses = trajectory["base_pose"]
                if isinstance(base_poses, list):
                    base_poses = np.array(base_poses)

                if len(base_poses.shape) > 1 and step > 0:
                    axes[1, 0].plot(base_poses[: step + 1, 0], base_poses[: step + 1, 1], "b-")
                    axes[1, 0].plot(base_poses[step, 0], base_poses[step, 1], "ro", markersize=10)
                    axes[1, 0].set_title("Robot Trajectory (Top View)")
                    axes[1, 0].set_xlabel("X Position")
                    axes[1, 0].set_ylabel("Y Position")
                    axes[1, 0].grid(True)
                    axes[1, 0].axis("equal")

            # Plot rewards (if available)
            if "rewards" in trajectory:
                rewards = trajectory["rewards"]
                if isinstance(rewards, list):
                    rewards = np.array(rewards)

                if len(rewards.shape) > 0 and step > 0:
                    axes[1, 1].plot(rewards[: step + 1])
                    axes[1, 1].set_title("Reward Signal")
                    axes[1, 1].set_xlabel("Time Step")
                    axes[1, 1].set_ylabel("Reward")
                    axes[1, 1].grid(True)

            # Render frame
            fig.canvas.draw()
            # Get frame as numpy array
            buf = fig.canvas.buffer_rgba()
            frame = np.asarray(buf)
            # Convert RGBA to RGB
            frame = frame[:, :, :3]
            frames.append(frame)

        plt.close(fig)

        # Save if requested
        if output_path:
            self._save_frames(frames, output_path)

        # Show if requested and in interactive mode
        if show and self.enable_gui:
            self._show_interactive(frames)

        return np.array(frames)

    def _render_basic(
        self,
        trajectory: Dict[str, Any],
        output_path: Optional[str],
        show: bool,
    ) -> np.ndarray:
        """Basic rendering fallback."""
        # Create simple visualization
        joint_positions = trajectory.get("joint_positions", [])
        if isinstance(joint_positions, list):
            joint_positions = np.array(joint_positions)

        # Simple frame generation
        frames = []
        num_steps = len(joint_positions) if len(joint_positions.shape) > 1 else 1

        for step in range(num_steps):
            # Create a simple visualization frame
            frame = np.zeros((self.resolution[1], self.resolution[0], 3), dtype=np.uint8)

            # Draw joint positions as bars
            if num_steps == 1:
                current_pos = joint_positions
            else:
                current_pos = joint_positions[step]

            # Normalize joint positions to frame height
            if len(current_pos) > 0:
                normalized = (current_pos - current_pos.min()) / (
                    current_pos.max() - current_pos.min() + 1e-8
                )
                bar_width = self.resolution[0] // len(current_pos)

                for i, val in enumerate(normalized):
                    bar_height = int(val * self.resolution[1] * 0.8)
                    x_start = i * bar_width
                    x_end = min((i + 1) * bar_width, self.resolution[0])
                    frame[self.resolution[1] - bar_height :, x_start:x_end] = [255, 255, 255]

            frames.append(frame)

        if output_path:
            self._save_frames(frames, output_path)

        return np.array(frames)

    def _save_frames(self, frames: List[np.ndarray], output_path: str) -> None:
        """Save frames to file.

        Args:
            frames: List of frame arrays
            output_path: Output file path
        """
        output_path = Path(output_path)

        if output_path.suffix in [".mp4", ".avi", ".mov"]:
            # Save as video
            import cv2  # https://opencv.org/

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(
                str(output_path), fourcc, self.fps, (self.resolution[0], self.resolution[1])
            )

            for frame in frames:
                # Resize frame if needed
                if frame.shape[:2] != (self.resolution[1], self.resolution[0]):
                    frame = cv2.resize(frame, self.resolution)
                out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            out.release()
        else:
            # Save as image sequence
            output_path.parent.mkdir(parents=True, exist_ok=True)
            import cv2  # https://opencv.org/

            for i, frame in enumerate(frames):
                frame_path = output_path.parent / f"{output_path.stem}_{i:04d}.png"
                cv2.imwrite(str(frame_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def _show_interactive(self, frames: List[np.ndarray]) -> None:
        """Show frames interactively (for demos).

        Args:
            frames: List of frame arrays
        """
        import matplotlib.animation as animation  # https://matplotlib.org/
        import matplotlib.pyplot as plt  # https://matplotlib.org/

        fig, ax = plt.subplots()
        im = ax.imshow(frames[0])
        ax.axis("off")

        def animate(frame_idx):
            im.set_array(frames[frame_idx % len(frames)])
            return [im]

        animation.FuncAnimation(
            fig, animate, frames=len(frames), interval=1000 // self.fps, blit=True
        )

        plt.show()
