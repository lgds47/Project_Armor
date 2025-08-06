#!/usr/bin/env python3
"""
Test script for DeviceManager memory calculations.

This script tests the memory calculation logic in the DeviceManager class
to ensure that batch sizes are calculated correctly based on available memory.
"""

import sys
from pathlib import Path
import torch
import unittest
from unittest.mock import patch, MagicMock

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import DeviceManager
from armor_pipeline.utils.device_manager import DeviceManager


class TestDeviceManagerMemory(unittest.TestCase):
    """Test cases for DeviceManager memory calculations."""

    @patch('torch.cuda.is_available')
    @patch('torch.cuda.get_device_properties')
    @patch('armor_pipeline.utils.device_manager.DeviceManager.get_memory_stats')
    def test_ensemble_memory_calculation(self, mock_get_memory_stats, mock_get_device_props, mock_cuda_available):
        """Test ensemble memory calculation with different scenarios."""
        # Mock CUDA availability
        mock_cuda_available.return_value = True
        
        # Create test scenarios with different GPU memory sizes and model counts
        scenarios = [
            # (total_memory_gb, allocated_memory_gb, num_models, expected_batch_size)
            (8, 1, 2, 1),    # Low-end GPU with 2 models
            (16, 2, 3, 2),   # Mid-range GPU with 3 models
            (32, 4, 4, 3),   # High-end GPU with 4 models
            (64, 8, 5, 6),   # Very high-end GPU with 5 models
        ]
        
        for total_memory_gb, allocated_memory_gb, num_models, expected_batch_size in scenarios:
            # Convert GB to bytes
            total_memory = total_memory_gb * (1024**3)
            allocated_memory = allocated_memory_gb * (1024**3)
            
            # Mock device properties
            mock_props = MagicMock()
            mock_props.total_memory = total_memory
            mock_get_device_props.return_value = mock_props
            
            # Mock memory stats
            mock_get_memory_stats.return_value = {0: {'allocated': allocated_memory_gb}}
            
            # Call the method
            device, batch_size = DeviceManager.get_ensemble_optimal_device(num_models)
            
            # Verify results
            self.assertEqual(device, torch.device('cuda'))
            self.assertEqual(batch_size, expected_batch_size, 
                            f"Failed with {total_memory_gb}GB GPU, {allocated_memory_gb}GB allocated, {num_models} models")
            
            # Print results for debugging
            print(f"GPU: {total_memory_gb}GB, Allocated: {allocated_memory_gb}GB, Models: {num_models}, "
                  f"Batch Size: {batch_size} (Expected: {expected_batch_size})")
    
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.get_device_properties')
    @patch('armor_pipeline.utils.device_manager.DeviceManager.get_memory_stats')
    def test_insufficient_memory(self, mock_get_memory_stats, mock_get_device_props, mock_cuda_available):
        """Test behavior when there's insufficient memory for the ensemble."""
        # Mock CUDA availability
        mock_cuda_available.return_value = True
        
        # Mock device properties (8GB GPU with 7GB already allocated)
        mock_props = MagicMock()
        mock_props.total_memory = 8 * (1024**3)
        mock_get_device_props.return_value = mock_props
        
        # Mock memory stats (7GB allocated)
        mock_get_memory_stats.return_value = {0: {'allocated': 7}}
        
        # Call the method with 3 models (which would require at least 6GB for base memory)
        device, batch_size = DeviceManager.get_ensemble_optimal_device(3)
        
        # Verify results - should fall back to minimal batch size
        self.assertEqual(device, torch.device('cuda'))
        self.assertEqual(batch_size, 1)
        
        print(f"Insufficient memory test: GPU: 8GB, Allocated: 7GB, Models: 3, "
              f"Batch Size: {batch_size} (Expected: 1)")


if __name__ == "__main__":
    print("Testing DeviceManager memory calculations...")
    unittest.main()