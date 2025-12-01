"""Synthetic dataset generation for testing and demos."""

from pipeline.synthetic.generator import SyntheticDataGenerator
from pipeline.synthetic.robotics import RoboticsTrajectoryGenerator
from pipeline.synthetic.video import SyntheticVideoGenerator

__all__ = [
    "SyntheticDataGenerator",
    "RoboticsTrajectoryGenerator",
    "SyntheticVideoGenerator",
]
