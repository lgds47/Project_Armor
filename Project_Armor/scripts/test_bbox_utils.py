#!/usr/bin/env python3
"""
Test script for bbox_utils.py, focusing on the validate_boxes function.

This script tests that the validate_boxes function correctly handles different input types
and does not modify the original input unexpectedly.
"""

import sys
from pathlib import Path
import numpy as np
import torch

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the function to test
from armor_pipeline.utils.bbox_utils import validate_boxes


def test_validate_boxes_numpy():
    """Test validate_boxes with NumPy arrays."""
    print("\nTesting validate_boxes with NumPy arrays...")
    
    # Create a test array with some invalid boxes
    boxes = np.array([
        [10, 10, 20, 20],  # Valid box
        [30, 30, 30, 40],  # Zero width
        [50, 50, 60, 50],  # Zero height
        [70, 70, 65, 65]   # Negative dimensions
    ])
    
    # Make a copy of the original boxes
    original_boxes = boxes.copy()
    
    # Call validate_boxes
    validated_boxes = validate_boxes(boxes, min_size=2.0)
    
    # Check that the original boxes were not modified
    if np.array_equal(boxes, original_boxes):
        print("✓ Original NumPy array was not modified")
    else:
        print("✗ Original NumPy array was modified unexpectedly")
        print(f"Original:\n{original_boxes}")
        print(f"Modified:\n{boxes}")
    
    # Check that the validated boxes were fixed correctly
    expected = np.array([
        [10, 10, 20, 20],      # Unchanged
        [30, 30, 32, 40],      # Width fixed to min_size
        [50, 50, 60, 52],      # Height fixed to min_size
        [70, 70, 72, 72]       # Both dimensions fixed to min_size
    ])
    
    if np.array_equal(validated_boxes, expected):
        print("✓ Validation correctly fixed the boxes")
    else:
        print("✗ Validation did not fix the boxes correctly")
        print(f"Expected:\n{expected}")
        print(f"Got:\n{validated_boxes}")


def test_validate_boxes_torch():
    """Test validate_boxes with PyTorch tensors."""
    print("\nTesting validate_boxes with PyTorch tensors...")
    
    # Create a test tensor with some invalid boxes
    boxes = torch.tensor([
        [10.0, 10.0, 20.0, 20.0],  # Valid box
        [30.0, 30.0, 30.0, 40.0],  # Zero width
        [50.0, 50.0, 60.0, 50.0],  # Zero height
        [70.0, 70.0, 65.0, 65.0]   # Negative dimensions
    ])
    
    # Make a copy of the original boxes
    original_boxes = boxes.clone()
    
    # Call validate_boxes
    validated_boxes = validate_boxes(boxes, min_size=2.0)
    
    # Check that the original boxes were not modified
    if torch.equal(boxes, original_boxes):
        print("✓ Original PyTorch tensor was not modified")
    else:
        print("✗ Original PyTorch tensor was modified unexpectedly")
        print(f"Original:\n{original_boxes}")
        print(f"Modified:\n{boxes}")
    
    # Check that the validated boxes were fixed correctly
    expected = torch.tensor([
        [10.0, 10.0, 20.0, 20.0],  # Unchanged
        [30.0, 30.0, 32.0, 40.0],  # Width fixed to min_size
        [50.0, 50.0, 60.0, 52.0],  # Height fixed to min_size
        [70.0, 70.0, 72.0, 72.0]   # Both dimensions fixed to min_size
    ])
    
    if torch.equal(validated_boxes, expected):
        print("✓ Validation correctly fixed the boxes")
    else:
        print("✗ Validation did not fix the boxes correctly")
        print(f"Expected:\n{expected}")
        print(f"Got:\n{validated_boxes}")


def test_validate_boxes_with_img_size():
    """Test validate_boxes with image size constraints."""
    print("\nTesting validate_boxes with image size constraints...")
    
    # Create a test array with some boxes that exceed image boundaries
    boxes = np.array([
        [10, 10, 20, 20],      # Within bounds
        [-5, 10, 20, 30],      # Negative x1
        [10, -5, 30, 20],      # Negative y1
        [80, 10, 110, 30],     # x2 exceeds width
        [10, 80, 30, 110]      # y2 exceeds height
    ])
    
    # Make a copy of the original boxes
    original_boxes = boxes.copy()
    
    # Call validate_boxes with image size
    img_size = (100, 100)  # width, height
    validated_boxes = validate_boxes(boxes, min_size=2.0, img_size=img_size)
    
    # Check that the original boxes were not modified
    if np.array_equal(boxes, original_boxes):
        print("✓ Original array was not modified")
    else:
        print("✗ Original array was modified unexpectedly")
        print(f"Original:\n{original_boxes}")
        print(f"Modified:\n{boxes}")
    
    # Check that the validated boxes were clipped correctly
    expected = np.array([
        [10, 10, 20, 20],      # Unchanged
        [0, 10, 20, 30],       # x1 clipped to 0
        [10, 0, 30, 20],       # y1 clipped to 0
        [80, 10, 100, 30],     # x2 clipped to width
        [10, 80, 30, 100]      # y2 clipped to height
    ])
    
    if np.array_equal(validated_boxes, expected):
        print("✓ Validation correctly clipped the boxes to image boundaries")
    else:
        print("✗ Validation did not clip the boxes correctly")
        print(f"Expected:\n{expected}")
        print(f"Got:\n{validated_boxes}")


def main():
    """Run all tests."""
    print("Testing bbox_utils.validate_boxes function...")
    
    # Test with NumPy arrays
    test_validate_boxes_numpy()
    
    # Test with PyTorch tensors
    test_validate_boxes_torch()
    
    # Test with image size constraints
    test_validate_boxes_with_img_size()
    
    print("\nAll tests completed.")


if __name__ == "__main__":
    main()