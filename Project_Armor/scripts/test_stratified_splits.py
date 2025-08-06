#!/usr/bin/env python3
"""
Test script for the create_stratified_splits function with oversampling.

This script creates a synthetic dataset with rare classes and tests that
the oversampling logic correctly increases the number of samples for rare classes.
"""

import sys
from pathlib import Path
import numpy as np
from collections import Counter

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the function to test and required classes
from armor_pipeline.data.dataset import create_stratified_splits
from armor_pipeline.data.parser import Annotation, Defect, DefectType, Point


def create_synthetic_dataset(num_samples=100):
    """Create a synthetic dataset with rare and common defect types."""
    # Define defect types with varying frequencies
    defect_types = {
        "common_defect_1": 30,     # Common defect (30 samples)
        "common_defect_2": 25,     # Common defect (25 samples)
        "medium_defect": 15,       # Medium frequency defect (15 samples)
        "rare_defect_1": 5,        # Rare defect (5 samples)
        "rare_defect_2": 3,        # Very rare defect (3 samples)
        "ultra_rare_defect": 1     # Ultra-rare defect (1 sample)
    }
    
    # Create annotations with defects
    annotations = []
    defect_count = 0
    
    for defect_name, count in defect_types.items():
        for i in range(count):
            # Create a defect
            defect = Defect(
                defect_type=DefectType.BBOX,
                name=defect_name,
                x_min=10,
                y_min=10,
                x_max=50,
                y_max=50
            )
            
            # Create an annotation with this defect
            annotation = Annotation(
                image_path=Path(f"dummy_image_{defect_count}.jpg"),
                defects=[defect]
            )
            
            annotations.append(annotation)
            defect_count += 1
    
    return annotations


def count_defects_by_type(data):
    """Count the number of samples for each defect type."""
    defect_counts = {}
    for sample in data:
        for defect in sample.defects:
            defect_counts[defect.name] = defect_counts.get(defect.name, 0) + 1
    return defect_counts


def test_oversampling():
    """Test that rare classes are oversampled correctly."""
    print("\nTesting oversampling of rare classes...")
    
    # Create a synthetic dataset
    data = create_synthetic_dataset()
    
    # Count defects before oversampling
    original_counts = count_defects_by_type(data)
    print("Original defect counts:")
    for defect_name, count in sorted(original_counts.items()):
        print(f"  {defect_name}: {count}")
    
    # Apply stratified splitting with oversampling
    train_data, val_data, test_data = create_stratified_splits(data, test_size=0.2, val_size=0.1)
    
    # Count defects in the combined dataset after oversampling
    all_data = train_data + val_data + test_data
    new_counts = count_defects_by_type(all_data)
    
    print("\nDefect counts after oversampling and splitting:")
    for defect_name, count in sorted(new_counts.items()):
        original = original_counts.get(defect_name, 0)
        print(f"  {defect_name}: {count} (was {original})")
    
    # Verify that rare classes have been oversampled
    min_samples = 20  # Same as in the implementation
    success = True
    
    for defect_name, original_count in original_counts.items():
        new_count = new_counts.get(defect_name, 0)
        
        if original_count < min_samples:
            # This should have been oversampled
            if new_count < original_count:
                print(f"✗ Error: {defect_name} has fewer samples after oversampling ({new_count} < {original_count})")
                success = False
            elif new_count < min_samples:
                print(f"⚠ Warning: {defect_name} still has fewer than {min_samples} samples ({new_count})")
            else:
                print(f"✓ {defect_name} was correctly oversampled from {original_count} to {new_count}")
        else:
            # This should not have been oversampled significantly
            if new_count > original_count * 1.5:
                print(f"⚠ Warning: {defect_name} was oversampled more than expected ({new_count} > {original_count * 1.5})")
            else:
                print(f"✓ {defect_name} was not significantly oversampled ({new_count} samples)")
    
    # Check distribution in train/val/test splits
    print("\nDistribution in splits:")
    train_counts = count_defects_by_type(train_data)
    val_counts = count_defects_by_type(val_data)
    test_counts = count_defects_by_type(test_data)
    
    for defect_name in sorted(new_counts.keys()):
        train = train_counts.get(defect_name, 0)
        val = val_counts.get(defect_name, 0)
        test = test_counts.get(defect_name, 0)
        total = train + val + test
        
        print(f"  {defect_name}: train={train} ({train/total:.1%}), val={val} ({val/total:.1%}), test={test} ({test/total:.1%})")
        
        # Check that each split has at least one sample of each defect type
        if train == 0 or val == 0 or test == 0:
            print(f"⚠ Warning: {defect_name} is missing from at least one split")
            success = False
    
    if success:
        print("\n✓ All rare classes were correctly oversampled and distributed across splits")
    else:
        print("\n✗ There were issues with the oversampling or distribution")
    
    return success


def main():
    """Run all tests."""
    print("Testing create_stratified_splits function with oversampling...")
    
    # Test oversampling
    success = test_oversampling()
    
    print("\nAll tests completed.")
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())