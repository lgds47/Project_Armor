"""
Unit tests for dataset module
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import os
import shutil

from armor_pipeline.data.dataset import ContactLensDataset
from armor_pipeline.data.parser import Annotation, Defect, DefectType, Point


class TestContactLensDataset:
    """Test ContactLensDataset class functionality"""
    
    @pytest.fixture
    def sample_dataset(self):
        """Create a sample dataset for testing"""
        # Create a minimal dataset with empty annotations
        dataset = ContactLensDataset(
            annotations=[],
            transform=None,
            mode="test",
            use_polygons=False,
            class_mapping={"test_defect": 1}
        )
        return dataset
    
    def test_ensure_coco_format_with_xyxy(self, sample_dataset):
        """Test ensure_coco_format with (x1, y1, x2, y2) format"""
        # Test with valid (x1, y1, x2, y2) format
        bbox = [10, 20, 50, 80]  # x1, y1, x2, y2
        coco_bbox = sample_dataset.ensure_coco_format(bbox)
        
        # Expected result: [x, y, w, h] = [10, 20, 40, 60]
        assert coco_bbox == [10, 20, 40, 60]
        assert len(coco_bbox) == 4
        assert coco_bbox[2] > 0  # width > 0
        assert coco_bbox[3] > 0  # height > 0
    
    def test_ensure_coco_format_with_coco(self, sample_dataset):
        """Test ensure_coco_format with (x, y, w, h) format"""
        # Test with valid COCO format (x, y, w, h)
        bbox = [10, 20, 30, 40]  # x, y, w, h
        coco_bbox = sample_dataset.ensure_coco_format(bbox)
        
        # Should return the same bbox since it's already in COCO format
        assert coco_bbox == [10, 20, 30, 40]
        assert len(coco_bbox) == 4
        assert coco_bbox[2] > 0  # width > 0
        assert coco_bbox[3] > 0  # height > 0
    
    def test_ensure_coco_format_with_invalid_bbox(self, sample_dataset):
        """Test ensure_coco_format with invalid bbox"""
        # Test with invalid bbox (less than 4 values)
        with pytest.raises(ValueError):
            sample_dataset.ensure_coco_format([10, 20, 30])
        
        # Test with invalid bbox (more than 4 values)
        with pytest.raises(ValueError):
            sample_dataset.ensure_coco_format([10, 20, 30, 40, 50])
    
    def test_ensure_coco_format_with_zero_dimensions(self, sample_dataset):
        """Test ensure_coco_format with zero dimensions"""
        # Test with zero width
        bbox = [10, 20, 10, 50]  # x1, y1, x2, y2 (zero width)
        coco_bbox = sample_dataset.ensure_coco_format(bbox)
        
        # Should fix the width to be at least 0.1
        assert coco_bbox[2] >= 0.1
        
        # Test with zero height
        bbox = [10, 20, 50, 20]  # x1, y1, x2, y2 (zero height)
        coco_bbox = sample_dataset.ensure_coco_format(bbox)
        
        # Should fix the height to be at least 0.1
        assert coco_bbox[3] >= 0.1
    
    def test_ensure_coco_format_with_negative_dimensions(self, sample_dataset):
        """Test ensure_coco_format with negative dimensions"""
        # Test with negative width (x2 < x1)
        bbox = [50, 20, 10, 60]  # x1, y1, x2, y2 (negative width)
        
        # This should be detected as COCO format since x2 < x1
        coco_bbox = sample_dataset.ensure_coco_format(bbox)
        assert coco_bbox == [50, 20, 10, 60]
        
        # Test with negative height (y2 < y1)
        bbox = [10, 60, 50, 20]  # x1, y1, x2, y2 (negative height)
        
        # This should be detected as COCO format since y2 < y1
        coco_bbox = sample_dataset.ensure_coco_format(bbox)
        assert coco_bbox == [10, 60, 50, 20]
    
    def test_ensure_coco_format_with_edge_cases(self, sample_dataset):
        """Test ensure_coco_format with edge cases"""
        # Test with very small values
        bbox = [0.1, 0.2, 0.3, 0.4]  # x1, y1, x2, y2
        coco_bbox = sample_dataset.ensure_coco_format(bbox)
        
        # Should be converted to COCO format
        assert coco_bbox == [0.1, 0.2, 0.2, 0.2]
        
        # Test with very large values
        bbox = [1000, 2000, 3000, 4000]  # x1, y1, x2, y2
        coco_bbox = sample_dataset.ensure_coco_format(bbox)
        
        # Should be converted to COCO format
        assert coco_bbox == [1000, 2000, 2000, 2000]
        
        # Test with float values
        bbox = [10.5, 20.5, 30.5, 40.5]  # x1, y1, x2, y2
        coco_bbox = sample_dataset.ensure_coco_format(bbox)
        
        # Should be converted to COCO format
        assert coco_bbox == [10.5, 20.5, 20.0, 20.0]