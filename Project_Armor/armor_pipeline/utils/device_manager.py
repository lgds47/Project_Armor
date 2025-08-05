"""
Device Manager module for Project Armor.

This module provides functionality to detect and manage hardware devices
for optimal performance based on available resources.
"""

import torch
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

class DeviceManager:
    """
    Manages device selection and configuration based on available hardware.
    
    This class provides utilities to detect available hardware resources
    and determine optimal settings for model training and inference.
    """
    
    @staticmethod
    def get_optimal_device() -> Tuple[torch.device, int]:
        """
        Determine the optimal device and batch size based on available hardware.
        
        Returns:
            Tuple containing:
                - torch.device: The optimal device (CUDA or CPU)
                - int: Recommended batch size for the device
        """
        if torch.cuda.is_available():
            # Check available memory
            mem = torch.cuda.get_device_properties(0).total_memory
            if mem < 16 * 1024**3:  # Less than 16GB
                print("Warning: Low GPU memory, reducing batch size")
                return torch.device('cuda'), 4  # Smaller batch
            return torch.device('cuda'), 8
        else:
            print("Warning: No GPU available, using CPU (will be slow)")
            return torch.device('cpu'), 2  # Tiny batch for CPU
            
    @staticmethod
    def get_ensemble_optimal_device(num_models: int = 3) -> Tuple[torch.device, int]:
        """
        Determine the optimal device and per-model batch size for ensemble models.
        
        This method adjusts memory calculations for multiple simultaneous models
        and provides appropriate batch size recommendations. It accounts for the
        increased memory requirements of running multiple models in parallel and
        includes fallback logic for insufficient memory scenarios.
        
        Args:
            num_models: Number of models in the ensemble (default: 3)
            
        Returns:
            Tuple containing:
                - torch.device: The optimal device (CUDA or CPU)
                - int: Recommended batch size per model
        """
        if torch.cuda.is_available():
            # Get device properties
            props = torch.cuda.get_device_properties(0)
            total_memory = props.total_memory
            
            # Get current memory usage
            memory_stats = DeviceManager.get_memory_stats()
            current_allocated = 0
            if memory_stats and 0 in memory_stats:
                current_allocated = memory_stats[0]['allocated'] * (1024**3)  # Convert GB to bytes
            
            # Calculate available memory (with safety margin)
            available_memory = total_memory - current_allocated
            safety_margin = 1 * (1024**3)  # 1GB safety margin
            usable_memory = max(0, available_memory - safety_margin)
            
            # Estimate memory per model (based on typical model sizes)
            # Assuming each model requires approximately 2-4GB depending on size
            base_model_memory = 3 * (1024**3)  # 3GB per model
            
            # Calculate total memory needed for ensemble
            ensemble_memory = base_model_memory * num_models
            
            # Determine batch size based on available memory and ensemble size
            if usable_memory >= ensemble_memory:
                # Enough memory for full ensemble with default batch sizes
                if total_memory >= 32 * (1024**3):  # High-end GPU (>=32GB)
                    return torch.device('cuda'), 6
                elif total_memory >= 16 * (1024**3):  # Mid-range GPU (>=16GB)
                    return torch.device('cuda'), 4
                else:  # Low-end GPU (<16GB)
                    return torch.device('cuda'), 2
            elif usable_memory >= ensemble_memory / 2:
                # Limited memory - reduce batch size
                logger.warning(f"Limited GPU memory for {num_models} models, reducing batch size")
                return torch.device('cuda'), 1
            else:
                # Very limited memory - fallback to sequential processing
                logger.warning(f"Insufficient GPU memory for {num_models} models ensemble, using minimal batch size")
                return torch.device('cuda'), 1
        else:
            # CPU fallback
            logger.warning("No GPU available, using CPU for ensemble (will be slow)")
            return torch.device('cpu'), 1  # Minimal batch size for CPU with ensemble
    
    @staticmethod
    def get_device_info() -> dict:
        """
        Get detailed information about the available devices.
        
        Returns:
            Dictionary containing device information
        """
        info = {
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "devices": []
        }
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                info["devices"].append({
                    "name": props.name,
                    "total_memory_gb": props.total_memory / (1024**3),
                    "compute_capability": f"{props.major}.{props.minor}",
                    "multi_processor_count": props.multi_processor_count
                })
        
        return info
    
    @staticmethod
    def get_memory_stats() -> dict:
        """
        Get current memory statistics for CUDA devices.
        
        Returns:
            Dictionary containing memory statistics or empty dict if CUDA is not available
        """
        if not torch.cuda.is_available():
            return {}
        
        stats = {}
        for i in range(torch.cuda.device_count()):
            stats[i] = {
                "allocated": torch.cuda.memory_allocated(i) / (1024**3),  # GB
                "reserved": torch.cuda.memory_reserved(i) / (1024**3),    # GB
                "max_allocated": torch.cuda.max_memory_allocated(i) / (1024**3)  # GB
            }
        
        return stats