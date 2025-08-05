"""
Bounding Box Utility Functions for Project Armor.

This module provides utility functions for handling bounding box coordinates,
including format detection and conversion between different formats.
"""

from typing import Union, Tuple, List, Optional, Any
import numpy as np
import torch


def convert_bbox_format(
    boxes: Union[np.ndarray, torch.Tensor, List[float], Tuple[float, ...]], 
    source_format: Optional[str] = None,
    target_format: str = 'xyxy',
    inplace: bool = False
) -> Union[np.ndarray, torch.Tensor, List[float]]:
    """
    Convert bounding boxes between different coordinate formats.
    
    This function can:
    1. Automatically detect the source format if not specified
    2. Convert between COCO (x, y, w, h) and xyxy (x1, y1, x2, y2) formats
    3. Handle both single boxes and batches of boxes
    4. Work with NumPy arrays, PyTorch tensors, or Python lists/tuples
    
    Args:
        boxes: Bounding boxes to convert. Can be:
            - A single box as a list/tuple of 4 values
            - A batch of boxes as a 2D NumPy array or PyTorch tensor of shape (N, 4)
        source_format: Format of the input boxes. If None, it will be auto-detected.
            Valid values: 'xyxy', 'coco', None
        target_format: Format to convert to. Valid values: 'xyxy', 'coco'
        inplace: If True and input is a tensor, the operation will be done in-place
            when possible. Only applies to PyTorch tensors.
            
    Returns:
        Converted boxes in the same type as the input (list, numpy array, or tensor)
        
    Raises:
        ValueError: If the input format is invalid or cannot be detected
        TypeError: If the input type is not supported
    """
    # Handle empty input
    if isinstance(boxes, (list, tuple)) and len(boxes) == 0:
        return []
    if isinstance(boxes, (np.ndarray, torch.Tensor)) and boxes.size == 0:
        return boxes
    
    # Convert single box to batch format for consistent processing
    is_single_box = False
    original_type = type(boxes)
    
    if isinstance(boxes, (list, tuple)):
        if len(boxes) == 4 and all(isinstance(x, (int, float)) for x in boxes):
            # Single box as list/tuple
            is_single_box = True
            boxes = np.array([boxes])
        else:
            # List of boxes
            boxes = np.array(boxes)
    
    # Ensure we're working with a 2D array/tensor
    if isinstance(boxes, np.ndarray) and boxes.ndim == 1 and len(boxes) == 4:
        is_single_box = True
        boxes = boxes.reshape(1, 4)
    elif isinstance(boxes, torch.Tensor) and boxes.dim() == 1 and len(boxes) == 4:
        is_single_box = True
        boxes = boxes.reshape(1, 4)
    
    # Validate input shape
    if not isinstance(boxes, (np.ndarray, torch.Tensor)):
        raise TypeError(f"Unsupported box type: {type(boxes)}. Expected list, tuple, numpy.ndarray, or torch.Tensor.")
    
    if boxes.shape[-1] != 4:
        raise ValueError(f"Boxes must have 4 values per box, got {boxes.shape[-1]}")
    
    # Clone tensor if not inplace
    if isinstance(boxes, torch.Tensor) and not inplace:
        boxes = boxes.clone()
    
    # Auto-detect source format if not specified
    if source_format is None:
        source_format = detect_bbox_format(boxes)
    
    # If source and target formats are the same, return the input
    if source_format == target_format:
        # Return single box if input was a single box
        if is_single_box:
            if isinstance(boxes, np.ndarray):
                if original_type in (list, tuple):
                    return original_type(boxes[0].tolist())
                return boxes[0]
            else:  # torch.Tensor
                if original_type in (list, tuple):
                    return original_type(boxes[0].tolist())
                return boxes[0]
        return boxes
    
    # Convert between formats
    if source_format == 'coco' and target_format == 'xyxy':
        # Convert COCO (x, y, w, h) to xyxy (x1, y1, x2, y2)
        if isinstance(boxes, np.ndarray):
            boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
            boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        else:  # torch.Tensor
            boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
            boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
    
    elif source_format == 'xyxy' and target_format == 'coco':
        # Convert xyxy (x1, y1, x2, y2) to COCO (x, y, w, h)
        if isinstance(boxes, np.ndarray):
            boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
            boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
        else:  # torch.Tensor
            boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
            boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
    
    else:
        raise ValueError(f"Unsupported conversion: {source_format} to {target_format}")
    
    # Return single box if input was a single box
    if is_single_box:
        if isinstance(boxes, np.ndarray):
            if original_type in (list, tuple):
                return original_type(boxes[0].tolist())
            return boxes[0]
        else:  # torch.Tensor
            if original_type in (list, tuple):
                return original_type(boxes[0].tolist())
            return boxes[0]
    
    return boxes


def detect_bbox_format(boxes: Union[np.ndarray, torch.Tensor]) -> str:
    """
    Detect the format of bounding boxes using multiple heuristics.
    
    This function uses several heuristics to determine if boxes are in:
    - COCO format (x, y, w, h)
    - xyxy format (x1, y1, x2, y2)
    
    Args:
        boxes: Bounding boxes to detect format for. Can be:
            - A single box as a 1D array/tensor of 4 values
            - A batch of boxes as a 2D array/tensor of shape (N, 4)
            
    Returns:
        String indicating the detected format: 'coco' or 'xyxy'
        
    Raises:
        ValueError: If the format cannot be reliably detected
    """
    # Handle empty input
    if isinstance(boxes, (np.ndarray, torch.Tensor)) and boxes.size == 0:
        return 'xyxy'  # Default to xyxy for empty arrays
    
    # Ensure we're working with a 2D array/tensor
    if isinstance(boxes, np.ndarray) and boxes.ndim == 1 and len(boxes) == 4:
        boxes = boxes.reshape(1, 4)
    elif isinstance(boxes, torch.Tensor) and boxes.dim() == 1 and len(boxes) == 4:
        boxes = boxes.reshape(1, 4)
    
    # Apply multiple heuristics to detect format
    coco_score = 0
    xyxy_score = 0
    
    # Heuristic 1: In COCO format, width/height are typically smaller than coordinates
    if isinstance(boxes, np.ndarray):
        if np.any(boxes[:, 2] < boxes[:, 0]) or np.any(boxes[:, 3] < boxes[:, 1]):
            coco_score += 2
        else:
            xyxy_score += 1
    else:  # torch.Tensor
        if torch.any(boxes[:, 2] < boxes[:, 0]) or torch.any(boxes[:, 3] < boxes[:, 1]):
            coco_score += 2
        else:
            xyxy_score += 1
    
    # Heuristic 2: In COCO format, width is typically less than twice the x-coordinate
    if isinstance(boxes, np.ndarray):
        if np.mean(boxes[:, 2]) < np.mean(boxes[:, 0] * 2):
            coco_score += 1
        else:
            xyxy_score += 1
    else:  # torch.Tensor
        if torch.mean(boxes[:, 2]) < torch.mean(boxes[:, 0] * 2):
            coco_score += 1
        else:
            xyxy_score += 1
    
    # Heuristic 3: In xyxy format, x2 > x1 and y2 > y1
    if isinstance(boxes, np.ndarray):
        if np.all(boxes[:, 2] > boxes[:, 0]) and np.all(boxes[:, 3] > boxes[:, 1]):
            xyxy_score += 2
    else:  # torch.Tensor
        if torch.all(boxes[:, 2] > boxes[:, 0]) and torch.all(boxes[:, 3] > boxes[:, 1]):
            xyxy_score += 2
    
    # Make a decision based on scores
    if coco_score > xyxy_score:
        return 'coco'
    elif xyxy_score > coco_score:
        return 'xyxy'
    else:
        # If scores are tied, default to xyxy as it's more common
        return 'xyxy'


def validate_boxes(
    boxes: Union[np.ndarray, torch.Tensor], 
    format: str = 'xyxy',
    min_size: float = 1.0,
    img_size: Optional[Tuple[int, int]] = None
) -> Union[np.ndarray, torch.Tensor]:
    """
    Validate and fix bounding boxes.
    
    This function:
    1. Ensures boxes have valid dimensions (width/height > 0)
    2. Optionally clips boxes to image boundaries
    3. Ensures minimum box size
    
    Args:
        boxes: Bounding boxes to validate. Can be:
            - A batch of boxes as a 2D NumPy array or PyTorch tensor of shape (N, 4)
        format: Format of the input boxes: 'xyxy' or 'coco'
        min_size: Minimum width/height for boxes
        img_size: Optional tuple of (width, height) to clip boxes to image boundaries
            
    Returns:
        Validated boxes in the same type as the input
    """
    if boxes.size == 0:
        return boxes
    
    # Convert to xyxy for validation if needed
    if format == 'coco':
        boxes = convert_bbox_format(boxes, 'coco', 'xyxy')
    
    # Ensure x2 > x1 and y2 > y1
    if isinstance(boxes, np.ndarray):
        boxes[:, 2] = np.maximum(boxes[:, 0] + min_size, boxes[:, 2])
        boxes[:, 3] = np.maximum(boxes[:, 1] + min_size, boxes[:, 3])
    else:  # torch.Tensor
        boxes[:, 2] = torch.maximum(boxes[:, 0] + min_size, boxes[:, 2])
        boxes[:, 3] = torch.maximum(boxes[:, 1] + min_size, boxes[:, 3])
    
    # Clip to image boundaries if img_size is provided
    if img_size is not None:
        img_w, img_h = img_size
        if isinstance(boxes, np.ndarray):
            boxes[:, 0] = np.clip(boxes[:, 0], 0, img_w - min_size)
            boxes[:, 1] = np.clip(boxes[:, 1], 0, img_h - min_size)
            boxes[:, 2] = np.clip(boxes[:, 2], boxes[:, 0] + min_size, img_w)
            boxes[:, 3] = np.clip(boxes[:, 3], boxes[:, 1] + min_size, img_h)
        else:  # torch.Tensor
            boxes[:, 0] = torch.clamp(boxes[:, 0], 0, img_w - min_size)
            boxes[:, 1] = torch.clamp(boxes[:, 1], 0, img_h - min_size)
            boxes[:, 2] = torch.clamp(boxes[:, 2], boxes[:, 0] + min_size, img_w)
            boxes[:, 3] = torch.clamp(boxes[:, 3], boxes[:, 1] + min_size, img_h)
    
    # Convert back to original format if needed
    if format == 'coco':
        boxes = convert_bbox_format(boxes, 'xyxy', 'coco')
    
    return boxes