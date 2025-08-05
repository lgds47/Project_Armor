#!/usr/bin/env python3
"""
Test script for CheckpointManager class.

This script tests the functionality of the CheckpointManager class to ensure that
it correctly manages checkpoint paths and directories.
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import CheckpointManager
from armor_pipeline.models.registry import CheckpointManager


def test_checkpoint_manager():
    """Test basic functionality of CheckpointManager."""
    print("\nTesting CheckpointManager...")
    
    # Create a temporary directory for testing
    temp_dir = Path("./temp_checkpoints")
    
    # Create a CheckpointManager instance
    checkpoint_manager = CheckpointManager(base_dir=temp_dir)
    
    # Check that the base directory was created
    print(f"Base directory exists: {temp_dir.exists()}")
    
    # Test getting checkpoint paths for different models
    models = ["model1", "model2", "model3"]
    
    for model_name in models:
        # Get best checkpoint path
        best_path = checkpoint_manager.get_checkpoint_path(model_name)
        print(f"\nModel: {model_name}")
        print(f"Best checkpoint path: {best_path}")
        
        # Check that the model directory was created
        model_dir = temp_dir / model_name
        print(f"Model directory exists: {model_dir.exists()}")
        
        # Get epoch checkpoint paths
        for epoch in [1, 10, 100]:
            epoch_path = checkpoint_manager.get_checkpoint_path(model_name, epoch)
            print(f"Epoch {epoch} checkpoint path: {epoch_path}")
    
    # Create a dummy checkpoint file
    dummy_model = "dummy_model"
    dummy_path = checkpoint_manager.get_checkpoint_path(dummy_model)
    with open(dummy_path, 'w') as f:
        f.write("dummy checkpoint content")
    
    print(f"\nCreated dummy checkpoint at: {dummy_path}")
    print(f"Dummy checkpoint exists: {dummy_path.exists()}")
    
    # Clean up
    try:
        import shutil
        shutil.rmtree(temp_dir)
        print(f"\nRemoved temporary directory: {temp_dir}")
    except Exception as e:
        print(f"\nError removing temporary directory: {e}")


def test_with_model_names():
    """Test CheckpointManager with realistic model names."""
    print("\nTesting CheckpointManager with realistic model names...")
    
    # Create a temporary directory for testing
    temp_dir = Path("./temp_checkpoints_realistic")
    
    # Create a CheckpointManager instance
    checkpoint_manager = CheckpointManager(base_dir=temp_dir)
    
    # Test with realistic model names
    model_names = [
        "yolov8m_edge",
        "micro_defect_specialist",
        "edge_defect_specialist",
        "presentation_checker",
        "yolo_nano_bubbles"
    ]
    
    for model_name in model_names:
        # Get best checkpoint path
        best_path = checkpoint_manager.get_checkpoint_path(model_name)
        print(f"\nModel: {model_name}")
        print(f"Best checkpoint path: {best_path}")
        
        # Get epoch checkpoint path
        epoch_path = checkpoint_manager.get_checkpoint_path(model_name, 50)
        print(f"Epoch 50 checkpoint path: {epoch_path}")
    
    # Clean up
    try:
        import shutil
        shutil.rmtree(temp_dir)
        print(f"\nRemoved temporary directory: {temp_dir}")
    except Exception as e:
        print(f"\nError removing temporary directory: {e}")


def main():
    """Run tests."""
    print("Testing CheckpointManager class...")
    
    # Test basic functionality
    test_checkpoint_manager()
    
    # Test with realistic model names
    test_with_model_names()
    
    print("\nAll tests completed.")


if __name__ == "__main__":
    main()