#!/usr/bin/env python3
"""
Test script to verify that ContactLensDataset raises a ValueError when expected_num_classes
doesn't match the actual number of classes.
"""

import sys
from pathlib import Path
import unittest
from unittest.mock import MagicMock

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the class to test
from armor_pipeline.data.dataset import ContactLensDataset


class TestClassMismatch(unittest.TestCase):
    """Test class mismatch validation in ContactLensDataset."""

    def test_class_mismatch(self):
        """Test that a ValueError is raised when expected_num_classes doesn't match num_classes."""
        # Create a mock annotation
        mock_annotation = MagicMock()
        mock_annotation.defects = []
        mock_annotation.image_path = MagicMock()
        mock_annotation.image_path.exists.return_value = True

        # Create a class mapping with 3 classes
        class_mapping = {"class1": 1, "class2": 2, "class3": 3}

        # Create a dataset with the class mapping
        dataset = ContactLensDataset(
            annotations=[mock_annotation],
            class_mapping=class_mapping
        )

        # Verify that num_classes is correctly calculated
        self.assertEqual(dataset.num_classes, 4)  # 3 classes + 1 for background

        # Set expected_num_classes to a different value
        dataset.expected_num_classes = 5

        # Verify that a ValueError is raised with the correct message
        with self.assertRaises(ValueError) as context:
            # Access any attribute to trigger __getattribute__, which will check the validation
            # This is a bit of a hack, but it's the simplest way to test the validation
            dataset.transform

        # Verify the error message
        self.assertIn("Class mismatch: dataset has 4 classes, model expects 5", str(context.exception))

    def test_no_expected_num_classes(self):
        """Test that no error is raised when expected_num_classes is not set."""
        # Create a mock annotation
        mock_annotation = MagicMock()
        mock_annotation.defects = []
        mock_annotation.image_path = MagicMock()
        mock_annotation.image_path.exists.return_value = True

        # Create a class mapping with 3 classes
        class_mapping = {"class1": 1, "class2": 2, "class3": 3}

        # Create a dataset with the class mapping
        dataset = ContactLensDataset(
            annotations=[mock_annotation],
            class_mapping=class_mapping
        )

        # Verify that num_classes is correctly calculated
        self.assertEqual(dataset.num_classes, 4)  # 3 classes + 1 for background

        # Verify that no error is raised when expected_num_classes is not set
        try:
            # Access any attribute to trigger __getattribute__
            dataset.transform
        except ValueError:
            self.fail("ValueError was raised unexpectedly!")

    def test_matching_expected_num_classes(self):
        """Test that no error is raised when expected_num_classes matches num_classes."""
        # Create a mock annotation
        mock_annotation = MagicMock()
        mock_annotation.defects = []
        mock_annotation.image_path = MagicMock()
        mock_annotation.image_path.exists.return_value = True

        # Create a class mapping with 3 classes
        class_mapping = {"class1": 1, "class2": 2, "class3": 3}

        # Create a dataset with the class mapping
        dataset = ContactLensDataset(
            annotations=[mock_annotation],
            class_mapping=class_mapping
        )

        # Verify that num_classes is correctly calculated
        self.assertEqual(dataset.num_classes, 4)  # 3 classes + 1 for background

        # Set expected_num_classes to the same value as num_classes
        dataset.expected_num_classes = 4

        # Verify that no error is raised when expected_num_classes matches num_classes
        try:
            # Access any attribute to trigger __getattribute__
            dataset.transform
        except ValueError:
            self.fail("ValueError was raised unexpectedly!")


if __name__ == "__main__":
    unittest.main()