#!/usr/bin/env python3
"""
Test script for S3ImageLoader.

This script tests the functionality of the S3ImageLoader class for loading images from S3.
It includes options for testing with real S3 credentials and mock testing for environments
without S3 access.
"""

import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import unittest
from unittest.mock import patch, MagicMock

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the S3ImageLoader class
from armor_pipeline.data.s3_image_loader import S3ImageLoader
from armor_pipeline.data.dataset import ContactLensDataset


def test_s3_image_loader_real():
    """Test the S3ImageLoader with real S3 credentials."""
    print("\nTesting S3ImageLoader with real S3 credentials...")
    
    # Check if S3 credentials are available
    if not os.getenv('S3_ACCESS_KEY') or not os.getenv('S3_SECRET_KEY'):
        print("⚠️ S3 credentials not found in environment variables.")
        print("Set S3_ACCESS_KEY and S3_SECRET_KEY to test with real S3.")
        print("Skipping real S3 test.")
        return
    
    # Create a temporary directory for testing
    temp_dir = Path("./temp_s3_cache")
    temp_dir.mkdir(exist_ok=True)
    
    try:
        # Initialize S3ImageLoader
        s3_loader = S3ImageLoader(cache_dir=temp_dir)
        
        # List images in the bucket
        print("Listing images in S3 bucket...")
        images = s3_loader.get_image_list()
        
        if not images:
            print("No images found in S3 bucket.")
            return
        
        print(f"Found {len(images)} images in S3 bucket.")
        print(f"First 5 images: {images[:5]}")
        
        # Load the first image
        print(f"Loading image: {images[0]}")
        image = s3_loader.load_image(images[0])
        
        if image is None:
            print("Failed to load image from S3.")
            return
        
        print(f"Successfully loaded image with shape: {image.shape}")
        
        # Display the image
        plt.figure(figsize=(10, 10))
        plt.imshow(image[:, :, ::-1])  # Convert BGR to RGB
        plt.title(f"Image from S3: {Path(images[0]).name}")
        plt.axis('off')
        plt.savefig(temp_dir / "s3_image_test.png")
        print(f"Saved image preview to {temp_dir / 's3_image_test.png'}")
        
        # Test cache functionality
        print("Testing cache functionality...")
        cache_files = list(temp_dir.glob("*"))
        print(f"Cache directory contains {len(cache_files)} files")
        
        # Load the same image again (should use cache)
        print("Loading the same image again (should use cache)...")
        image2 = s3_loader.load_image(images[0])
        if image2 is not None:
            print("Successfully loaded image from cache")
        
        # Test prefetch functionality
        if len(images) >= 3:
            print("Testing prefetch functionality...")
            s3_loader.prefetch_images(images[1:3])
            cache_files_after = list(temp_dir.glob("*"))
            print(f"Cache directory now contains {len(cache_files_after)} files")
        
        # Test clear cache functionality
        print("Testing clear cache functionality...")
        s3_loader.clear_cache()
        cache_files_after_clear = list(temp_dir.glob("*"))
        print(f"Cache directory contains {len(cache_files_after_clear)} files after clearing")
        
    finally:
        # Clean up
        import shutil
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            print(f"Removed temporary directory: {temp_dir}")


class MockResponse:
    """Mock S3 response for testing."""
    
    def __init__(self, content):
        self.body = MagicMock()
        self.body.read.return_value = content


def test_s3_image_loader_mock():
    """Test the S3ImageLoader with mock S3 responses."""
    print("\nTesting S3ImageLoader with mock S3 responses...")
    
    # Create a temporary directory for testing
    temp_dir = Path("./temp_s3_cache")
    temp_dir.mkdir(exist_ok=True)
    
    try:
        # Create a mock image
        mock_image = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_image[25:75, 25:75, 0] = 255  # Red square
        
        # Import cv2 here to avoid issues if it's not installed
        import cv2
        _, mock_image_bytes = cv2.imencode('.png', mock_image)
        mock_image_bytes = mock_image_bytes.tobytes()
        
        # Create mock S3 client
        mock_s3_client = MagicMock()
        mock_s3_client.get_object.return_value = MockResponse(mock_image_bytes)
        mock_s3_client.get_paginator.return_value.paginate.return_value = [
            {'Contents': [{'Key': 'test_image1.png'}, {'Key': 'test_image2.jpg'}]}
        ]
        
        # Patch boto3.client to return our mock
        with patch('boto3.client', return_value=mock_s3_client):
            # Initialize S3ImageLoader
            s3_loader = S3ImageLoader(cache_dir=temp_dir)
            
            # Test get_image_list
            images = s3_loader.get_image_list()
            print(f"Mock S3 returned {len(images)} images: {images}")
            
            # Test load_image
            image = s3_loader.load_image('test_image1.png')
            if image is not None:
                print(f"Successfully loaded mock image with shape: {image.shape}")
                
                # Display the image
                plt.figure(figsize=(5, 5))
                plt.imshow(image)
                plt.title("Mock Image from S3")
                plt.axis('off')
                plt.savefig(temp_dir / "mock_s3_image_test.png")
                print(f"Saved mock image preview to {temp_dir / 'mock_s3_image_test.png'}")
            else:
                print("Failed to load mock image")
    
    finally:
        # Clean up
        import shutil
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            print(f"Removed temporary directory: {temp_dir}")


def test_contact_lens_dataset_with_s3():
    """Test the ContactLensDataset with S3 image loading."""
    print("\nTesting ContactLensDataset with S3 image loading...")
    
    # Check if S3 credentials are available
    if not os.getenv('S3_ACCESS_KEY') or not os.getenv('S3_SECRET_KEY'):
        print("⚠️ S3 credentials not found in environment variables.")
        print("Set S3_ACCESS_KEY and S3_SECRET_KEY to test with real S3.")
        print("Skipping ContactLensDataset with S3 test.")
        return
    
    # Create a mock annotation with S3 path
    from armor_pipeline.data.parser import Annotation, Defect, DefectType
    
    # Get a list of images from S3
    s3_loader = S3ImageLoader()
    images = s3_loader.get_image_list()
    
    if not images:
        print("No images found in S3 bucket.")
        return
    
    # Create a mock annotation with S3 path
    s3_path = f"s3://{images[0]}"
    mock_annotation = Annotation(
        image_path=Path(s3_path),
        defects=[
            Defect(
                defect_type=DefectType.BBOX,
                name="test_defect",
                x_min=10,
                y_min=10,
                x_max=50,
                y_max=50
            )
        ]
    )
    
    # Patch the exists method to return True for S3 paths
    original_exists = Path.exists
    
    def mock_exists(self):
        if str(self).startswith('s3://'):
            return True
        return original_exists(self)
    
    try:
        # Apply the patch
        Path.exists = mock_exists
        
        # Create a ContactLensDataset with S3 support
        dataset = ContactLensDataset(
            annotations=[mock_annotation],
            class_mapping={"test_defect": 1},
            use_s3=True
        )
        
        print(f"Created ContactLensDataset with {len(dataset)} samples")
        
        # Try to get an item from the dataset
        try:
            image, target = dataset[0]
            print(f"Successfully loaded image with shape: {image.shape}")
            print(f"Target: {target}")
        except Exception as e:
            print(f"Error loading image from dataset: {e}")
    
    finally:
        # Restore the original exists method
        Path.exists = original_exists


def main():
    """Run all tests."""
    print("Testing S3ImageLoader functionality...")
    
    # Test with real S3 credentials if available
    test_s3_image_loader_real()
    
    # Test with mock S3 responses
    test_s3_image_loader_mock()
    
    # Test ContactLensDataset with S3 support
    test_contact_lens_dataset_with_s3()
    
    print("\nAll tests completed.")


if __name__ == "__main__":
    main()