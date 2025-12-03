"""GR00T Training Functions.

Core training loop, evaluation, and Ray Train integration.
"""

import logging
import time
from typing import Any, Dict, Optional

import ray
import torch
import torch.distributed as dist
import torch.nn as nn
from ray.data import Dataset
from ray.train import Checkpoint, get_context
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP

from pipeline.training.groot.config import GrootTrainingConfig
from pipeline.training.groot.data import get_training_dataset, preprocess_batch, validate_batch
from pipeline.training.groot.losses import compute_diffusion_loss
from pipeline.training.groot.model import GrootVLA, TransformerBlock
from pipeline.training.groot.schedulers import get_learning_rate_scheduler

# GPU memory utilities
try:
    from pipeline.utils.gpu.memory import get_gpu_memory_info
except ImportError:
    def get_gpu_memory_info(device_id=None):
        return {"total": 0, "allocated": 0, "reserved": 0, "free": 0}

logger = logging.getLogger(__name__)





