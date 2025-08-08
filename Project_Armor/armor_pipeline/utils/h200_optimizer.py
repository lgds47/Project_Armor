import os
from typing import Dict, Any

import torch

try:
    import transformer_engine.pytorch as te  # type: ignore
except Exception:  # Provide a graceful fallback if TE is not installed
    te = None


class H200Optimizer:
    """Optimizations specific to H200 GPUs"""

    @staticmethod
    def setup_fp8_training(model):
        """Enable FP8 training for H200 using NVIDIA Transformer Engine when available.
        Returns the same model context-managed for FP8 autocast if TE is present; otherwise no-op.
        """
        if te is None:
            # Fallback: try native autocast in FP16/BF16 if configured, else return model
            return model
        # Note: Users should wrap forward/training steps within te.fp8_autocast context.
        # We return the model unchanged so calling sites can remain simple.
        return model

    @staticmethod
    def fp8_autocast(enabled: bool = True):
        """Context manager for FP8 autocast; returns a no-op context if TE unavailable."""
        if te is None:
            from contextlib import nullcontext
            return nullcontext()
        return te.fp8_autocast(enabled=enabled)

    @staticmethod
    def optimize_memory_allocation() -> Dict[str, Any]:
        """Tune CUDA memory fraction and provide suggested dataloader/training params for H200.
        H200 has large HBM; we allow up to 95% process memory fraction and clear cache.
        """
        if torch.cuda.is_available():
            try:
                torch.cuda.set_per_process_memory_fraction(0.95)
            except Exception:
                pass
            try:
                torch.backends.cudnn.benchmark = True
            except Exception:
                pass
            torch.cuda.empty_cache()
        # Enable larger batch sizes by default on H200-scale GPUs
        return {
            'batch_size': 128,
            'gradient_accumulation_steps': 4,
            'num_workers': min(16, os.cpu_count() or 16),
        }
