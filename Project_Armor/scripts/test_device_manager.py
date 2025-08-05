#!/usr/bin/env python3
"""
Test script for DeviceManager class.

This script tests the functionality of the DeviceManager class to ensure that
it correctly detects available hardware and recommends appropriate device and batch size.
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import DeviceManager
from armor_pipeline.utils.device_manager import DeviceManager


def test_device_detection():
    """Test device detection functionality."""
    print("\nTesting device detection...")
    
    # Get device and batch size
    device, batch_size = DeviceManager.get_optimal_device()
    
    print(f"Detected device: {device}")
    print(f"Recommended batch size: {batch_size}")
    
    # Check if device is as expected
    if device.type == 'cuda':
        print("CUDA device detected")
        
        # Get device properties
        import torch
        props = torch.cuda.get_device_properties(0)
        print(f"Device name: {props.name}")
        print(f"Total memory: {props.total_memory / (1024**3):.2f} GB")
        print(f"Compute capability: {props.major}.{props.minor}")
        
        # Check if batch size is appropriate for the GPU memory
        mem_gb = props.total_memory / (1024**3)
        if mem_gb < 16 and batch_size > 4:
            print("WARNING: Batch size may be too large for available GPU memory")
        elif mem_gb >= 16 and batch_size < 8:
            print("WARNING: Batch size may be too small for available GPU memory")
    else:
        print("CPU device detected")
        
        # Check if batch size is appropriate for CPU
        if batch_size > 2:
            print("WARNING: Batch size may be too large for CPU")


def test_device_info():
    """Test device info functionality."""
    print("\nTesting device info...")
    
    # Get device info
    info = DeviceManager.get_device_info()
    
    print(f"CUDA available: {info['cuda_available']}")
    print(f"Device count: {info['device_count']}")
    
    # Print device details
    for i, device in enumerate(info['devices']):
        print(f"\nDevice {i}:")
        print(f"  Name: {device['name']}")
        print(f"  Total memory: {device['total_memory_gb']:.2f} GB")
        print(f"  Compute capability: {device['compute_capability']}")
        print(f"  Multi-processor count: {device['multi_processor_count']}")


def test_memory_stats():
    """Test memory statistics functionality."""
    print("\nTesting memory statistics...")
    
    # Get memory stats
    stats = DeviceManager.get_memory_stats()
    
    if not stats:
        print("No CUDA devices available, skipping memory stats test")
        return
    
    # Print memory stats
    for device_id, device_stats in stats.items():
        print(f"\nDevice {device_id}:")
        print(f"  Allocated memory: {device_stats['allocated']:.2f} GB")
        print(f"  Reserved memory: {device_stats['reserved']:.2f} GB")
        print(f"  Max allocated memory: {device_stats['max_allocated']:.2f} GB")


def test_ensemble_device_selection():
    """Test ensemble device selection functionality."""
    print("\nTesting ensemble device selection...")
    
    # Test with different ensemble sizes
    ensemble_sizes = [2, 3, 5, 8]
    
    for num_models in ensemble_sizes:
        print(f"\nTesting with {num_models} models in ensemble:")
        
        # Get device and batch size for ensemble
        device, batch_size = DeviceManager.get_ensemble_optimal_device(num_models)
        
        print(f"  Recommended device: {device}")
        print(f"  Recommended batch size per model: {batch_size}")
        
        # Calculate total batch size across all models
        total_batch = batch_size * num_models
        print(f"  Total effective batch size: {total_batch}")
        
        # Check if device is as expected
        if device.type == 'cuda':
            # Get device properties
            import torch
            props = torch.cuda.get_device_properties(0)
            mem_gb = props.total_memory / (1024**3)
            
            print(f"  GPU memory: {mem_gb:.2f} GB")
            
            # Get current memory usage
            stats = DeviceManager.get_memory_stats()
            if stats and 0 in stats:
                print(f"  Current allocated memory: {stats[0]['allocated']:.2f} GB")
                print(f"  Current reserved memory: {stats[0]['reserved']:.2f} GB")
            
            # Estimate memory requirements
            estimated_memory = 3 * num_models  # 3GB per model (estimated)
            print(f"  Estimated memory requirement: {estimated_memory:.2f} GB")
            
            # Check if recommendations are reasonable
            if mem_gb < estimated_memory and batch_size > 1:
                print("  WARNING: Batch size may be too large for available GPU memory with this ensemble size")
            elif mem_gb >= 32 and batch_size < 4 and num_models <= 3:
                print("  WARNING: Batch size may be too small for available GPU memory with this ensemble size")
        else:
            print("  CPU device selected for ensemble")
            
            # Check if batch size is appropriate for CPU with ensemble
            if batch_size > 1:
                print("  WARNING: Batch size may be too large for CPU with ensemble")


def main():
    """Run tests."""
    print("Testing DeviceManager class...")
    
    # Test device detection
    test_device_detection()
    
    # Test device info
    test_device_info()
    
    # Test memory stats
    test_memory_stats()
    
    # Test ensemble device selection
    test_ensemble_device_selection()
    
    print("\nAll tests completed.")


if __name__ == "__main__":
    main()