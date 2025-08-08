"""Minimal utilities for initializing PyTorch distributed training.
This is optional and uses environment variables commonly provided by launchers
(e.g., torchrun, Kubernetes, or SLURM) to initialize the process group.
"""
from __future__ import annotations

import os
from typing import Optional

import torch
import torch.distributed as dist


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def init_from_env(backend: str = "nccl", timeout_seconds: int = 1800) -> bool:
    """Initialize torch.distributed from environment variables.

    Recognized environment variables:
    - RANK, WORLD_SIZE, LOCAL_RANK
    - MASTER_ADDR, MASTER_PORT
    - NCCL_* for NCCL behavior

    Returns True if successfully initialized, False otherwise.
    """
    if not torch.cuda.is_available():
        return False
    if not dist.is_available():
        return False
    if is_distributed():
        return True

    required = ["MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE"]
    if not all(os.environ.get(k) for k in required):
        return False

    # Default to environment-provided local rank for device assignment
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", 0)))
    torch.cuda.set_device(local_rank % torch.cuda.device_count())

    try:
        from datetime import timedelta
        dist.init_process_group(
            backend=backend,
            timeout=timedelta(seconds=timeout_seconds),
        )
        return True
    except Exception:
        return False


def get_world_size() -> int:
    if is_distributed():
        return dist.get_world_size()
    return 1


def get_rank() -> int:
    if is_distributed():
        return dist.get_rank()
    return 0


def barrier():
    if is_distributed():
        dist.barrier()
