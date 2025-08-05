"""
Unit tests for bounding box utility functions
"""

import pytest
import numpy as np
import torch
from armor_pipeline.utils.bbox_utils import convert_bbox_format, detect_bbox_format, validate_boxes


class TestBBoxUtils:
    """Test bounding box utility functions"""

    def test_convert_bbox_format_single_box(self):
        """Test converting a single box between formats"""
        # Test converting from xyxy to COCO
        xyxy_box = [10, 20, 50, 80]  # x1, y1, x2, y2
        coco_box = convert_bbox_format(xyxy_box, source_format='xyxy', target_format='coco')
        assert coco_box == [10, 20, 40, 60]  # x, y, w, h

        # Test converting from COCO to xyxy
        coco_box = [10, 20, 40, 60]  # x, y, w, h
        xyxy_box = convert_bbox_format(coco_box, source_format='coco', target_format='xyxy')
        assert xyxy_box == [10, 20, 50, 80]  # x1, y1, x2, y2

        # Test auto-detection of format (xyxy)
        xyxy_box = [10, 20, 50, 80]  # x1, y1, x2, y2
        coco_box = convert_bbox_format(xyxy_box, source_format=None, target_format='coco')
        assert coco_box == [10, 20, 40, 60]  # x, y, w, h

        # Test auto-detection of format (COCO)
        coco_box = [10, 20, 5, 10]  # x, y, w, h (small width/height)
        xyxy_box = convert_bbox_format(coco_box, source_format=None, target_format='xyxy')
        assert xyxy_box == [10, 20, 15, 30]  # x1, y1, x2, y2

    def test_convert_bbox_format_batch(self):
        """Test converting a batch of boxes between formats"""
        # Test with NumPy array
        xyxy_boxes = np.array([
            [10, 20, 50, 80],
            [100, 200, 300, 400]
        ])
        coco_boxes = convert_bbox_format(xyxy_boxes, source_format='xyxy', target_format='coco')
        assert isinstance(coco_boxes, np.ndarray)
        assert coco_boxes.shape == (2, 4)
        assert np.array_equal(coco_boxes, np.array([
            [10, 20, 40, 60],
            [100, 200, 200, 200]
        ]))

        # Test with PyTorch tensor
        xyxy_boxes = torch.tensor([
            [10, 20, 50, 80],
            [100, 200, 300, 400]
        ], dtype=torch.float32)
        coco_boxes = convert_bbox_format(xyxy_boxes, source_format='xyxy', target_format='coco')
        assert isinstance(coco_boxes, torch.Tensor)
        assert coco_boxes.shape == (2, 4)
        assert torch.all(coco_boxes == torch.tensor([
            [10, 20, 40, 60],
            [100, 200, 200, 200]
        ], dtype=torch.float32))

    def test_convert_bbox_format_edge_cases(self):
        """Test converting boxes with edge cases"""
        # Test with empty list
        empty_list = []
        result = convert_bbox_format(empty_list, source_format='xyxy', target_format='coco')
        assert result == []

        # Test with empty NumPy array
        empty_array = np.array([])
        result = convert_bbox_format(empty_array, source_format='xyxy', target_format='coco')
        assert result.size == 0

        # Test with empty PyTorch tensor
        empty_tensor = torch.tensor([])
        result = convert_bbox_format(empty_tensor, source_format='xyxy', target_format='coco')
        assert result.size(0) == 0

        # Test with invalid input format
        with pytest.raises(ValueError):
            convert_bbox_format([10, 20, 50, 80], source_format='invalid', target_format='coco')

        # Test with invalid target format
        with pytest.raises(ValueError):
            convert_bbox_format([10, 20, 50, 80], source_format='xyxy', target_format='invalid')

        # Test with invalid box shape
        with pytest.raises(ValueError):
            convert_bbox_format([10, 20, 50], source_format='xyxy', target_format='coco')

    def test_detect_bbox_format(self):
        """Test detecting bounding box format"""
        # Test with xyxy format
        xyxy_box = np.array([[10, 20, 50, 80]])
        assert detect_bbox_format(xyxy_box) == 'xyxy'

        # Test with COCO format (small width/height)
        coco_box = np.array([[10, 20, 5, 10]])
        assert detect_bbox_format(coco_box) == 'coco'

        # Test with COCO format (negative width/height)
        coco_box = np.array([[10, 20, -5, 10]])
        assert detect_bbox_format(coco_box) == 'coco'

        # Test with PyTorch tensor
        xyxy_box = torch.tensor([[10, 20, 50, 80]])
        assert detect_bbox_format(xyxy_box) == 'xyxy'

        # Test with empty array
        empty_array = np.array([])
        assert detect_bbox_format(empty_array) == 'xyxy'  # Default for empty arrays

    def test_validate_boxes(self):
        """Test validating and fixing bounding boxes"""
        # Test with valid boxes
        valid_boxes = np.array([
            [10, 20, 50, 80],
            [100, 200, 300, 400]
        ])
        result = validate_boxes(valid_boxes, format='xyxy')
        assert np.array_equal(result, valid_boxes)

        # Test with invalid boxes (zero width/height)
        invalid_boxes = np.array([
            [10, 20, 10, 80],  # Zero width
            [100, 200, 300, 200]  # Zero height
        ])
        result = validate_boxes(invalid_boxes, format='xyxy', min_size=5.0)
        # Should fix the boxes to have minimum width/height
        assert result[0, 2] == 15  # x2 = x1 + min_size
        assert result[1, 3] == 205  # y2 = y1 + min_size

        # Test with image boundaries
        boxes = np.array([
            [10, 20, 50, 80],
            [100, 200, 1000, 1000]  # Exceeds image boundaries
        ])
        result = validate_boxes(boxes, format='xyxy', img_size=(500, 500))
        # Should clip the boxes to image boundaries
        assert result[1, 2] <= 500  # x2 <= img_width
        assert result[1, 3] <= 500  # y2 <= img_height

        # Test with COCO format
        coco_boxes = np.array([
            [10, 20, 40, 60],
            [100, 200, 0, 0]  # Zero width/height
        ])
        result = validate_boxes(coco_boxes, format='coco', min_size=5.0)
        # Should fix the boxes to have minimum width/height
        assert result[1, 2] >= 5.0  # w >= min_size
        assert result[1, 3] >= 5.0  # h >= min_size

    def test_convert_bbox_format_preserves_type(self):
        """Test that convert_bbox_format preserves input type"""
        # Test with list
        list_box = [10, 20, 50, 80]
        result = convert_bbox_format(list_box, source_format='xyxy', target_format='coco')
        assert isinstance(result, list)

        # Test with tuple
        tuple_box = (10, 20, 50, 80)
        result = convert_bbox_format(tuple_box, source_format='xyxy', target_format='coco')
        assert isinstance(result, tuple)

        # Test with NumPy array
        np_box = np.array([10, 20, 50, 80])
        result = convert_bbox_format(np_box, source_format='xyxy', target_format='coco')
        assert isinstance(result, np.ndarray)

        # Test with PyTorch tensor
        torch_box = torch.tensor([10, 20, 50, 80])
        result = convert_bbox_format(torch_box, source_format='xyxy', target_format='coco')
        assert isinstance(result, torch.Tensor)

    def test_inplace_conversion(self):
        """Test inplace conversion for PyTorch tensors"""
        # Test with PyTorch tensor
        torch_box = torch.tensor([[10, 20, 50, 80]], dtype=torch.float32)
        original_data_ptr = torch_box.data_ptr()
        
        # Non-inplace conversion should create a new tensor
        result = convert_bbox_format(torch_box, source_format='xyxy', target_format='coco', inplace=False)
        assert result.data_ptr() != original_data_ptr
        
        # Inplace conversion should modify the original tensor
        result = convert_bbox_format(torch_box, source_format='xyxy', target_format='coco', inplace=True)
        assert result.data_ptr() == original_data_ptr