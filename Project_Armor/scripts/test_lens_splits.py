#!/usr/bin/env python3
"""
Test script to verify that create_stratified_splits() prevents the same lens from appearing in different splits.

This script creates a synthetic dataset with multiple annotations per lens and verifies that
all annotations for the same lens are kept together in the same split.
"""

import sys
from pathlib import Path
import random

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the function to test and required classes
from armor_pipeline.data.dataset import create_stratified_splits
from armor_pipeline.data.parser import Annotation, Defect, DefectType


def create_synthetic_dataset(num_lenses=50, annotations_per_lens=3, defect_types=None):
    """
    Create a synthetic dataset with multiple annotations per lens.
    
    Args:
        num_lenses: Number of unique lenses to create
        annotations_per_lens: Number of annotations per lens
        defect_types: List of defect types to use (if None, uses default list)
        
    Returns:
        List of Annotation objects
    """
    if defect_types is None:
        defect_types = [
            "bubble", "scratch", "tear", "debris", "edge_chip",
            "folded", "inverted", "missing", "multiple"
        ]
    
    annotations = []
    
    # Create annotations for each lens
    for lens_idx in range(num_lenses):
        lens_id = f"lens_{lens_idx:04d}"
        
        # Assign 1-3 defect types to this lens
        num_defects = random.randint(1, 3)
        lens_defect_types = random.sample(defect_types, num_defects)
        
        # Create multiple annotations for this lens
        for ann_idx in range(annotations_per_lens):
            # Create a defect for each defect type
            defects = []
            for defect_type in lens_defect_types:
                defect = Defect(
                    defect_type=DefectType.BBOX,
                    name=defect_type,
                    x_min=random.randint(10, 100),
                    y_min=random.randint(10, 100),
                    x_max=random.randint(101, 200),
                    y_max=random.randint(101, 200)
                )
                defects.append(defect)
            
            # Create an annotation with these defects
            annotation = Annotation(
                image_path=Path(f"{lens_id}_view_{ann_idx}.jpg"),
                defects=defects
            )
            
            annotations.append(annotation)
    
    return annotations


def check_lens_separation(train_data, val_data, test_data=None):
    """
    Check that no lens appears in more than one split.
    
    Args:
        train_data: List of annotations in the training set
        val_data: List of annotations in the validation set
        test_data: List of annotations in the test set (optional)
        
    Returns:
        True if no lens appears in more than one split, False otherwise
    """
    # Extract lens IDs from each split
    train_lens_ids = {ann.image_path.stem.split('_view_')[0] for ann in train_data}
    val_lens_ids = {ann.image_path.stem.split('_view_')[0] for ann in val_data}
    test_lens_ids = set()
    if test_data:
        test_lens_ids = {ann.image_path.stem.split('_view_')[0] for ann in test_data}
    
    # Check for overlaps
    train_val_overlap = train_lens_ids.intersection(val_lens_ids)
    train_test_overlap = train_lens_ids.intersection(test_lens_ids)
    val_test_overlap = val_lens_ids.intersection(test_lens_ids)
    
    # Print results
    print(f"Train set: {len(train_data)} annotations, {len(train_lens_ids)} unique lenses")
    print(f"Val set: {len(val_data)} annotations, {len(val_lens_ids)} unique lenses")
    if test_data:
        print(f"Test set: {len(test_data)} annotations, {len(test_lens_ids)} unique lenses")
    
    if train_val_overlap:
        print(f"❌ ERROR: {len(train_val_overlap)} lenses appear in both train and val sets")
        print(f"Overlapping lenses: {train_val_overlap}")
        return False
    
    if train_test_overlap:
        print(f"❌ ERROR: {len(train_test_overlap)} lenses appear in both train and test sets")
        print(f"Overlapping lenses: {train_test_overlap}")
        return False
    
    if val_test_overlap:
        print(f"❌ ERROR: {len(val_test_overlap)} lenses appear in both val and test sets")
        print(f"Overlapping lenses: {val_test_overlap}")
        return False
    
    print("✅ SUCCESS: No lens appears in more than one split")
    return True


def test_lens_separation():
    """Test that create_stratified_splits() keeps all annotations for the same lens in the same split."""
    print("\nTesting lens separation in create_stratified_splits()...")
    
    # Create a synthetic dataset with multiple annotations per lens
    data = create_synthetic_dataset(num_lenses=50, annotations_per_lens=3)
    print(f"Created synthetic dataset with {len(data)} annotations for {50} lenses")
    
    # Test train/val split (no test set)
    print("\nTesting train/val split (no test set)...")
    train_data, val_data = create_stratified_splits(data, test_size=0, val_size=0.2)
    check_lens_separation(train_data, val_data)
    
    # Test train/test split (no val set)
    print("\nTesting train/test split (no val set)...")
    train_data, test_data = create_stratified_splits(data, test_size=0.2, val_size=0)
    check_lens_separation(train_data, test_data)
    
    # Test train/val/test split
    print("\nTesting train/val/test split...")
    train_data, val_data, test_data = create_stratified_splits(data, test_size=0.2, val_size=0.1)
    check_lens_separation(train_data, val_data, test_data)


def main():
    """Run tests."""
    print("Testing create_stratified_splits() lens separation...")
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Test lens separation
    test_lens_separation()
    
    print("\nAll tests completed.")


if __name__ == "__main__":
    main()