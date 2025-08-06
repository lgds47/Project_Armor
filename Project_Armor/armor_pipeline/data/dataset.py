"""
Data Module for J&J Contact Lens Defect Detection
Handles image loading, augmentation, and dataset creation
"""

import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import albumentations as A
from albumentations.pytorch import ToTensorV2
import json
from PIL import Image
from sklearn.model_selection import train_test_split
from skmultilearn.model_selection import iterative_train_test_split

from armor_pipeline.data.parser import Annotation, Defect, DefectType
from armor_pipeline.utils.bbox_utils import convert_bbox_format


def create_stratified_splits(data, test_size=0.2, val_size=0.1):
    """
    Create stratified train/val/test splits ensuring rare classes appear in all splits.
    Ensures that annotations from the same lens (image) are kept together in the same split.
    
    Args:
        data: List of data samples (annotations)
        test_size: Proportion of data to use for testing
        val_size: Proportion of data to use for validation
        
    Returns:
        Tuple of (train_data, val_data, test_data) if both test_size > 0 and val_size > 0
        Tuple of (train_data, test_data) if test_size > 0 and val_size = 0
        Tuple of (train_data, val_data) if test_size = 0 and val_size > 0
        Tuple of (data, []) if both test_size = 0 and val_size = 0
    """
    # First, handle ultra-rare classes with oversampling
    defect_counts = {}
    for sample in data:
        for defect in sample.defects:
            defect_counts[defect.name] = defect_counts.get(defect.name, 0) + 1
    
    # Oversample rare classes BEFORE splitting
    min_samples = 20  # Minimum per class
    augmented_data = list(data)
    for defect_name, count in defect_counts.items():
        if count < min_samples:
            # Find all samples with this defect
            rare_samples = [s for s in data if any(d.name == defect_name for d in s.defects)]
            # Oversample to reach minimum
            num_copies = (min_samples - count) // len(rare_samples) + 1
            for _ in range(num_copies):
                augmented_data.extend(rare_samples)
    
    data = augmented_data
    
    # Group annotations by lens (image) identifier
    lens_groups = {}
    for i, sample in enumerate(data):
        # Use image_path.stem as the lens identifier
        lens_id = sample.image_path.stem
        if lens_id not in lens_groups:
            lens_groups[lens_id] = []
        lens_groups[lens_id].append((i, sample))
    
    # Extract all labels at the lens level
    all_labels = []
    lens_label_matrix = []
    
    for lens_id, samples in lens_groups.items():
        lens_labels = []
        
        # Collect all defect labels from all annotations for this lens
        for _, sample in samples:
            if hasattr(sample, 'defects') and sample.defects:
                for defect in sample.defects:
                    lens_labels.append(defect.name)
        
        # Create a binary label vector for this lens
        unique_lens_labels = list(set(lens_labels))
        all_labels.extend(unique_lens_labels)
        
        # We'll build the label matrix after getting all unique labels
        lens_label_matrix.append((lens_id, unique_lens_labels))
    
    # Get unique labels across all lenses
    unique_labels = sorted(list(set(all_labels)))
    label_to_idx = {label: i for i, label in enumerate(unique_labels)}
    
    # Build the multi-label matrix (lenses x labels)
    lens_ids = list(lens_groups.keys())
    X = np.array(range(len(lens_ids))).reshape(-1, 1)
    y = np.zeros((len(lens_ids), len(unique_labels)), dtype=int)
    
    for i, (lens_id, labels) in enumerate(lens_label_matrix):
        for label in labels:
            y[i, label_to_idx[label]] = 1
    
    # First split: train+val vs test
    if test_size > 0:
        train_val_idx, test_idx, y_train_val, y_test = iterative_train_test_split(
            X, y, test_size=test_size
        )
        
        # Convert back to lens indices
        train_val_lens_indices = train_val_idx.flatten()
        test_lens_indices = test_idx.flatten()
        
        # Get the lens IDs for each split
        train_val_lens_ids = [lens_ids[i] for i in train_val_lens_indices]
        test_lens_ids = [lens_ids[i] for i in test_lens_indices]
        
        # Create test set by collecting all annotations for test lenses
        test_data = []
        for lens_id in test_lens_ids:
            for _, sample in lens_groups[lens_id]:
                test_data.append(sample)
        
        # Second split: train vs val (if needed)
        if val_size > 0:
            # Adjust val_size relative to the train+val set
            relative_val_size = val_size / (1 - test_size)
            
            # Create a new lens groups dictionary for train+val lenses
            train_val_lens_groups = {lens_id: lens_groups[lens_id] for lens_id in train_val_lens_ids}
            
            # Extract all labels at the lens level for train+val
            train_val_all_labels = []
            train_val_lens_label_matrix = []
            
            for lens_id, samples in train_val_lens_groups.items():
                lens_labels = []
                
                # Collect all defect labels from all annotations for this lens
                for _, sample in samples:
                    if hasattr(sample, 'defects') and sample.defects:
                        for defect in sample.defects:
                            lens_labels.append(defect.name)
                
                # Create a binary label vector for this lens
                unique_lens_labels = list(set(lens_labels))
                train_val_all_labels.extend(unique_lens_labels)
                
                # We'll build the label matrix after getting all unique labels
                train_val_lens_label_matrix.append((lens_id, unique_lens_labels))
            
            # Get unique labels across all train+val lenses
            train_val_unique_labels = sorted(list(set(train_val_all_labels)))
            train_val_label_to_idx = {label: i for i, label in enumerate(train_val_unique_labels)}
            
            # Build the multi-label matrix for train+val lenses
            train_val_lens_ids = list(train_val_lens_groups.keys())
            X_train_val = np.array(range(len(train_val_lens_ids))).reshape(-1, 1)
            y_train_val_new = np.zeros((len(train_val_lens_ids), len(train_val_unique_labels)), dtype=int)
            
            for i, (lens_id, labels) in enumerate(train_val_lens_label_matrix):
                for label in labels:
                    if label in train_val_label_to_idx:
                        y_train_val_new[i, train_val_label_to_idx[label]] = 1
            
            # Split train+val into train and val
            train_idx, val_idx, y_train, y_val = iterative_train_test_split(
                X_train_val, y_train_val_new, test_size=relative_val_size
            )
            
            # Convert back to lens indices
            train_lens_indices = train_idx.flatten()
            val_lens_indices = val_idx.flatten()
            
            # Get the lens IDs for each split
            train_lens_ids = [train_val_lens_ids[i] for i in train_lens_indices]
            val_lens_ids = [train_val_lens_ids[i] for i in val_lens_indices]
            
            # Create train and val sets by collecting all annotations for each lens
            train_data = []
            for lens_id in train_lens_ids:
                for _, sample in lens_groups[lens_id]:
                    train_data.append(sample)
                    
            val_data = []
            for lens_id in val_lens_ids:
                for _, sample in lens_groups[lens_id]:
                    val_data.append(sample)
            
            return train_data, val_data, test_data
        else:
            # No validation set needed
            train_data = []
            for lens_id in train_val_lens_ids:
                for _, sample in lens_groups[lens_id]:
                    train_data.append(sample)
            return train_data, test_data
    else:
        # No test set needed, just train/val split
        if val_size > 0:
            train_idx, val_idx, y_train, y_val = iterative_train_test_split(
                X, y, test_size=val_size
            )
            
            # Convert back to lens indices
            train_lens_indices = train_idx.flatten()
            val_lens_indices = val_idx.flatten()
            
            # Get the lens IDs for each split
            train_lens_ids = [lens_ids[i] for i in train_lens_indices]
            val_lens_ids = [lens_ids[i] for i in val_lens_indices]
            
            # Create train and val sets by collecting all annotations for each lens
            train_data = []
            for lens_id in train_lens_ids:
                for _, sample in lens_groups[lens_id]:
                    train_data.append(sample)
                    
            val_data = []
            for lens_id in val_lens_ids:
                for _, sample in lens_groups[lens_id]:
                    val_data.append(sample)
            
            return train_data, val_data
        else:
            # No splitting needed
            return data, []


class ContactLensDataset(Dataset):
    """PyTorch Dataset for contact lens defect detection"""

    def __init__(
        self,
        annotations: List[Annotation],
        transform: Optional[A.Compose] = None,
        mode: str = "train",
        use_polygons: bool = False,
        class_mapping: Optional[Dict[str, int]] = None,
        mapping_file: Optional[Path] = None,
        allow_new_classes: bool = False
    ):
        """
        Args:
            annotations: List of parsed annotations
            transform: Albumentations transform pipeline
            mode: 'train', 'val', or 'test'
            use_polygons: If True, return polygon masks for segmentation
            class_mapping: Dict mapping defect names to class IDs
            mapping_file: Path to JSON file for saving/loading class mapping
            allow_new_classes: If True, allow new classes not in existing mapping
        """
        self.annotations = annotations
        self.transform = transform
        self.mode = mode
        self.use_polygons = use_polygons
        self.mapping_file = mapping_file
        self.allow_new_classes = allow_new_classes

        # Build class mapping if not provided
        if class_mapping is None:
            self.class_mapping = self._build_class_mapping()
        else:
            self.class_mapping = class_mapping

        self.num_classes = len(self.class_mapping) + 1  # +1 for background

        # Validate if model expects different num_classes
        if hasattr(self, 'expected_num_classes') and self.expected_num_classes != self.num_classes:
            raise ValueError(f"Class mismatch: dataset has {self.num_classes} classes, model expects {self.expected_num_classes}")

        # Filter valid annotations (those with existing images)
        self.valid_annotations = [
            ann for ann in annotations
            if ann.image_path.exists()
        ]

        print(f"Dataset initialized with {len(self.valid_annotations)} valid images")
        print(f"Class mapping: {self.class_mapping}")

    def _build_class_mapping(self) -> Dict[str, int]:
        """
        Build mapping from defect names to class IDs with validation and progress indicators.
        
        This method:
        1. Loads existing mapping from JSON file if available
        2. Validates that no new classes appear without explicit handling
        3. Saves the mapping to a JSON file for reproducibility
        4. Shows progress indicators for large datasets
        5. Returns the mapping dictionary
        
        Returns:
            Dict mapping defect names to class IDs
            
        Raises:
            ValueError: If new classes are found and allow_new_classes is False
        """
        import logging
        from tqdm import tqdm
        
        # Get logger
        logger = logging.getLogger("armor_pipeline.dataset")
        
        logger.info("Building class mapping...")
        
        # Get unique defect names from annotations with progress indicator
        unique_names = set()
        defect_counts = {}
        
        logger.info("Analyzing defect classes in annotations...")
        for ann in tqdm(self.annotations, desc="Analyzing defect classes", unit="annotation"):
            for defect in ann.defects:
                unique_names.add(defect.name)
                defect_counts[defect.name] = defect_counts.get(defect.name, 0) + 1
        
        logger.info(f"Found {len(unique_names)} unique defect classes in annotations")
        
        # Log class distribution
        logger.info("Defect class distribution in annotations:")
        for name, count in sorted(defect_counts.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {name}: {count} instances")
        
        # Try to load existing mapping if mapping_file is provided
        existing_mapping = {}
        if self.mapping_file is not None and self.mapping_file.exists():
            try:
                with open(self.mapping_file, 'r') as f:
                    existing_mapping = json.load(f)
                logger.info(f"Loaded existing class mapping from {self.mapping_file}")
            except json.JSONDecodeError:
                logger.warning(f"Could not parse mapping file {self.mapping_file}. Creating new mapping.")
        
        # Check for new classes if we have an existing mapping
        if existing_mapping:
            new_classes = unique_names - set(existing_mapping.keys())
            if new_classes and not self.allow_new_classes:
                error_msg = (
                    f"Found {len(new_classes)} new classes not in existing mapping: {new_classes}. "
                    f"Set allow_new_classes=True to allow new classes."
                )
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            # If we allow new classes, add them to the mapping
            if new_classes and self.allow_new_classes:
                logger.warning(f"Adding {len(new_classes)} new classes to mapping: {new_classes}")
                # Find the highest class ID in the existing mapping
                max_id = max(existing_mapping.values())
                # Add new classes with consecutive IDs
                sorted_new_classes = sorted(list(new_classes))
                for idx, name in enumerate(sorted_new_classes):
                    existing_mapping[name] = max_id + idx + 1
                
            # Use the existing mapping (possibly updated with new classes)
            class_mapping = existing_mapping
        else:
            # Create a new mapping from scratch
            # Sort for consistency
            logger.info("Creating new class mapping from scratch")
            sorted_names = sorted(list(unique_names))
            class_mapping = {name: idx + 1 for idx, name in enumerate(sorted_names)}
        
        # Save mapping to file if mapping_file is provided
        if self.mapping_file is not None:
            # Create directory if it doesn't exist
            self.mapping_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.mapping_file, 'w') as f:
                json.dump(class_mapping, f, indent=2, sort_keys=True)
            logger.info(f"Saved class mapping to {self.mapping_file}")
        
        # Validate class mapping
        logger.info("Validating class mapping...")
        missing_classes = unique_names - set(class_mapping.keys())
        if missing_classes:
            logger.warning(f"The following classes are in the annotations but not in the mapping: {missing_classes}")
            
        unused_classes = set(class_mapping.keys()) - unique_names
        if unused_classes:
            logger.warning(f"The following classes are in the mapping but not in the annotations: {unused_classes}")
            
        # Check for classes with few samples
        min_samples_warning = 5
        classes_with_few_samples = [
            name for name, count in defect_counts.items()
            if count < min_samples_warning
        ]
        
        if classes_with_few_samples:
            logger.warning(f"The following classes have fewer than {min_samples_warning} samples: {', '.join(classes_with_few_samples)}")
            
        logger.info(f"Class mapping validation completed. Mapping contains {len(class_mapping)} classes.")
        
        return class_mapping

    def __len__(self) -> int:
        return len(self.valid_annotations)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get item for training/inference
        
        This method:
        1. Loads the image
        2. Extracts bounding boxes from defects
        3. Validates and converts bounding boxes to COCO format (x, y, w, h)
        4. Applies transformations if specified
        5. Prepares the target dictionary with boxes, labels, areas, etc.
        
        Args:
            idx: Index of the annotation to retrieve
            
        Returns:
            Tuple of (image, target) where:
                - image is a tensor of shape [C, H, W]
                - target is a dictionary containing boxes, labels, etc.
        """
        ann = self.valid_annotations[idx]

        # Load image
        image = self._load_image(ann.image_path)

        # Prepare targets
        bboxes = []
        labels = []
        sample_weights = []
        masks = [] if self.use_polygons else None

        for defect in ann.defects:
            # Get bbox from defect - always returns (x1, y1, x2, y2) format
            # per parser.py:94-113
            bbox = defect.to_bbox()
            
            # Calculate tightness ratio for linear defects
            if defect.defect_type == DefectType.POLYLINE:
                polyline_area = self._calculate_polyline_area(defect.points, width=2)  # 2px width
                bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                tightness = polyline_area / max(bbox_area, 1e-6)
                
                # Skip if bbox is >90% empty space (would confuse model)
                if tightness < 0.1:
                    continue
                    
            # For polygons, use actual polygon area vs bbox area as sample weight
            sample_weight = 1.0
            if defect.defect_type in [DefectType.POLYGON, DefectType.POLYLINE]:
                actual_area = self._calculate_defect_area(defect)
                bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                sample_weight = actual_area / max(bbox_area, 1e-6)  # Weight by fill ratio
            
            try:
                # Explicitly convert from xyxy to COCO format
                coco_bbox = convert_bbox_format(bbox, source_format='xyxy', target_format='coco')
                
                # Extract coordinates for further processing
                x1, y1, w, h = coco_bbox
                
                # Skip invalid bboxes with zero or negative dimensions
                if w <= 0 or h <= 0:
                    continue
                    
                # Ensure coordinates are within image bounds
                # This prevents issues with bounding boxes that extend beyond the image
                img_h, img_w = image.shape[:2]
                x1 = max(0, min(x1, img_w - 1))  # Clip x1 to image width
                y1 = max(0, min(y1, img_h - 1))  # Clip y1 to image height
                w = min(w, img_w - x1)           # Adjust width to stay within image
                h = min(h, img_h - y1)           # Adjust height to stay within image
                
                # Add the validated and converted bbox to our list
                bboxes.append([x1, y1, w, h])
                # Add the label for this defect
                labels.append(self.class_mapping.get(defect.name, 0))
                # Add the sample weight
                sample_weights.append(sample_weight)
                
                # Get polygon mask if needed
                if self.use_polygons:
                    mask = self._create_mask(defect, image.shape[:2])
                    masks.append(mask)
            except (ValueError, TypeError) as e:
                # Skip invalid bboxes
                continue

        # Apply augmentations
        if self.transform:
            if self.use_polygons and masks:
                transformed = self.transform(
                    image=image,
                    bboxes=bboxes,
                    masks=masks,
                    labels=labels,
                    sample_weights=sample_weights  # Include sample weights
                )
                image = transformed['image']
                bboxes = transformed['bboxes']
                masks = transformed['masks']
                labels = transformed['labels']
                sample_weights = transformed['sample_weights']
            else:
                transformed = self.transform(
                    image=image,
                    bboxes=bboxes,
                    labels=labels,
                    sample_weights=sample_weights  # Include sample weights
                )
                image = transformed['image']
                bboxes = transformed['bboxes']
                labels = transformed['labels']
                sample_weights = transformed['sample_weights']

        # Convert to tensors
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        # Step 7: Prepare tensors for model input
        boxes_tensor = torch.tensor(bboxes, dtype=torch.float32)
        
        # Step 8: Calculate areas from COCO format boxes
        # This is important for:
        # - COCO evaluation metrics
        # - NMS (Non-Maximum Suppression) during inference
        # - Handling small vs large objects differently
        areas = []
        for box in bboxes:
            x, y, w, h = box
            # Double-check that width and height are positive
            # This is a safety measure in case any invalid boxes slipped through
            w = max(0, w)
            h = max(0, h)
            # Area in COCO format is simply width * height
            areas.append(w * h)
        
        # Step 9: Build the final target dictionary
        # This follows the format expected by detection models like Faster R-CNN
        target = {
            'boxes': boxes_tensor,                                  # Bounding boxes in COCO format
            'labels': torch.tensor(labels, dtype=torch.int64),      # Class labels
            'image_id': torch.tensor([idx]),                        # Image identifier
            'area': torch.tensor(areas, dtype=torch.float32),       # Box areas for COCO metrics
            'iscrowd': torch.zeros(len(bboxes), dtype=torch.int64), # Crowd flag (0 for individual objects)
            'image_path': str(ann.image_path),                      # Original image path for reference
            'sample_weights': torch.tensor(sample_weights, dtype=torch.float32)  # Use in loss function
        }

        if self.use_polygons and masks:
            target['masks'] = torch.stack([torch.from_numpy(m) for m in masks])

        return image, target

    def _load_image(self, image_path: Path) -> np.ndarray:
        """Load and preprocess image"""
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Crop to upper half (2048x2048)
        image = image[:2048, :]

        return image

    def _create_mask(self, defect: Defect, shape: Tuple[int, int]) -> np.ndarray:
        """Create binary mask for defect"""
        mask = np.zeros(shape, dtype=np.uint8)

        # Get polygon points
        polygon = defect.to_polygon()
        pts = np.array(polygon, dtype=np.int32)

        # Fill polygon
        cv2.fillPoly(mask, [pts], 1)

        return mask
        
    def _calculate_polyline_area(self, points: List[Point], width: int = 2) -> float:
        """
        Calculate the area of a polyline with a given width.
        
        Args:
            points: List of points defining the polyline
            width: Width of the polyline in pixels
            
        Returns:
            Area of the polyline in pixels
        """
        # Create a mask with the same size as a typical image
        mask = np.zeros((2048, 2048), dtype=np.uint8)
        
        # Convert points to numpy array
        pts = np.array([(p.x, p.y) for p in points], dtype=np.int32)
        
        # Draw the polyline with the specified width
        cv2.polylines(mask, [pts], False, 1, thickness=width)
        
        # Count non-zero pixels
        return float(np.count_nonzero(mask))
    
    def _calculate_defect_area(self, defect: Defect) -> float:
        """
        Calculate the area of any defect type.
        
        Args:
            defect: Defect object
            
        Returns:
            Area of the defect in pixels
        """
        if defect.defect_type == DefectType.POLYLINE:
            # For polylines, calculate area with a width of 2 pixels
            return self._calculate_polyline_area(defect.points, width=2)
        
        elif defect.defect_type == DefectType.POLYGON:
            # For polygons, create a mask and count non-zero pixels
            mask = self._create_mask(defect, (2048, 2048))
            return float(np.count_nonzero(mask))
        
        elif defect.defect_type == DefectType.ELLIPSE:
            # For ellipses, calculate area using the formula: π * rx * ry
            return np.pi * defect.rx * defect.ry
        
        elif defect.defect_type == DefectType.BBOX:
            # For bboxes, calculate area as width * height
            return (defect.x_max - defect.x_min) * (defect.y_max - defect.y_min)
        
        # Default fallback: calculate bbox area
        bbox = defect.to_bbox()
        return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

        
    def get_class_names(self) -> List[str]:
        """Get list of class names in order"""
        names = ['background']
        for name, idx in sorted(self.class_mapping.items(), key=lambda x: x[1]):
            names.append(name)
        return names


def get_transform(mode: str = "train", img_size: int = 1024) -> A.Compose:
    """Get augmentation pipeline for training/validation"""

    if mode == "train":
        transform = A.Compose([
            A.Resize(img_size, img_size),
            # Photometric augmentations OK for grayscale
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
            A.RandomGamma(gamma_limit=(90, 110), p=0.2),
            # Geometric augmentations that preserve measurements
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            # NO rotation - changes defect measurements!
            # NO scaling - changes pixel sizes!
            A.OneOf([
                A.GaussNoise(p=1),
                A.GaussianBlur(blur_limit=3, p=1),  # Mild blur only
                # NO MotionBlur - creates artificial linear defects
            ], p=0.2),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ], bbox_params=A.BboxParams(
            format='coco',
            label_fields=['labels', 'sample_weights'],
            min_visibility=0.3
        ))
    else:
        transform = A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ], bbox_params=A.BboxParams(
            format='coco',
            label_fields=['labels', 'sample_weights']
        ))

    return transform


def create_dataloaders(
    train_annotations: List[Annotation],
    val_annotations: List[Annotation],
    batch_size: int = 4,
    num_workers: int = 4,
    img_size: int = 1024,
    use_polygons: bool = False,
    mapping_file: Optional[Path] = None,
    class_mapping: Optional[Dict[str, int]] = None,
    allow_new_classes: bool = False
) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
    """
    Create train and validation dataloaders
    
    Args:
        train_annotations: List of annotations for training
        val_annotations: List of annotations for validation
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for dataloaders
        img_size: Size of images after resizing
        use_polygons: If True, return polygon masks for segmentation
        mapping_file: Path to JSON file for saving/loading class mapping
        class_mapping: Optional pre-loaded class mapping to use
        allow_new_classes: If True, allow new classes not in existing mapping
        
    Returns:
        Tuple of (train_loader, val_loader, class_mapping)
    """
    # Use provided class_mapping if available
    # Otherwise, build unified class mapping from all annotations if mapping_file is not provided
    # Or let ContactLensDataset handle the mapping from file
    if class_mapping is None and mapping_file is None:
        all_annotations = train_annotations + val_annotations
        class_mapping = {}
        for ann in all_annotations:
            for defect in ann.defects:
                if defect.name not in class_mapping:
                    class_mapping[defect.name] = len(class_mapping) + 1

    # Create datasets
    train_dataset = ContactLensDataset(
        annotations=train_annotations,
        transform=get_transform("train", img_size),
        mode="train",
        use_polygons=use_polygons,
        class_mapping=class_mapping,
        mapping_file=mapping_file,
        allow_new_classes=allow_new_classes
    )

    # Use the same class mapping for validation dataset
    # If mapping_file was provided, train_dataset.class_mapping will have been loaded/created
    val_dataset = ContactLensDataset(
        annotations=val_annotations,
        transform=get_transform("val", img_size),
        mode="val",
        use_polygons=use_polygons,
        class_mapping=train_dataset.class_mapping,
        mapping_file=None  # Don't save mapping again from validation dataset
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    return train_loader, val_loader, class_mapping


def collate_fn(batch):
    """Custom collate function for variable number of objects per image"""
    images = []
    targets = []

    for image, target in batch:
        images.append(image)
        targets.append(target)

    # Stack images
    images = torch.stack(images)

    return images, targets


class DataModule:
    """High-level data module for the entire pipeline"""

    def __init__(
        self,
        data_root: Path = None,
        batch_size: int = 4,
        num_workers: int = 4,
        img_size: int = 1024,
        train_split: float = 0.8,
        use_polygons: bool = False,
        mapping_file: Optional[Path] = None,
        allow_new_classes: bool = False,
        load_existing_mapping: bool = False,
        save_mapping: bool = True
    ):
        """
        Initialize the data module.
        
        Args:
            data_root: Root directory for the project
            batch_size: Batch size for dataloaders
            num_workers: Number of workers for dataloaders
            img_size: Size of images after resizing
            train_split: Proportion of data to use for training
            use_polygons: If True, return polygon masks for segmentation
            mapping_file: Path to JSON file for saving/loading class mapping
            allow_new_classes: If True, allow new classes not in existing mapping
            load_existing_mapping: If True, load class mapping from mapping_file
            save_mapping: If True, save class mapping to mapping_file after setup
        """
        from armor_pipeline.utils.project_config import get_project_config
        import logging
        
        self.logger = logging.getLogger("armor_pipeline.data")
        self.project_config = get_project_config(base_path=data_root if data_root else None)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = img_size
        self.train_split = train_split
        self.use_polygons = use_polygons
        self.allow_new_classes = allow_new_classes
        self.load_existing_mapping = load_existing_mapping
        self.save_mapping = save_mapping

        self.images_dir = self.project_config.image_path
        self.annotations_dir = self.project_config.annotation_path
        
        # Set up mapping file path
        if mapping_file is None:
            # Default to a file in the data directory
            self.mapping_file = self.project_config.data_path / "mappings" / "class_mapping.json"
        else:
            self.mapping_file = mapping_file

        self.class_mapping = None
        self.train_loader = None
        self.val_loader = None
        self.all_annotations = []

    def setup(self):
        """
        Parse annotations and create dataloaders with validation and progress indicators.
        
        This method:
        1. Parses all annotations from the annotations directory
        2. Splits the annotations into training and validation sets
        3. Creates dataloaders for training and validation
        4. Handles class mapping with reproducibility and validation
        5. Optionally loads existing class mapping or saves new mapping
        6. Shows progress indicators for large datasets
        7. Validates dataset integrity and consistency
        
        Returns:
            Tuple of (train_loader, val_loader)
            
        Raises:
            FileNotFoundError: If annotations or images directories don't exist
            ValueError: If class mapping is inconsistent
        """
        from armor_pipeline.data.parser import parse_all_annotations
        from tqdm import tqdm
        import time

        # Start timing
        start_time = time.time()
        self.logger.info("Setting up data module...")
        
        # Verify directories exist
        if not self.annotations_dir.exists():
            error_msg = f"Annotations directory not found: {self.annotations_dir}"
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)
            
        if not self.images_dir.exists():
            error_msg = f"Images directory not found: {self.images_dir}"
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)
            
        self.logger.info(f"Using annotations directory: {self.annotations_dir}")
        self.logger.info(f"Using images directory: {self.images_dir}")

        # Parse all annotations with progress tracking
        self.logger.info("Parsing annotations...")
        parse_start_time = time.time()
        self.all_annotations = parse_all_annotations(
            self.annotations_dir,
            self.images_dir,
            crop_to_upper_half=True
        )
        parse_time = time.time() - parse_start_time
        self.logger.info(f"Parsed {len(self.all_annotations)} annotations in {parse_time:.2f} seconds")

        # Use stratified splitting to ensure rare classes appear in all splits
        # We're only doing train/val split here (no test set), so we set test_size=0
        self.logger.info("Splitting dataset into train and validation sets...")
        split_start_time = time.time()
        train_annotations, val_annotations = create_stratified_splits(
            self.all_annotations, 
            test_size=0, 
            val_size=1-self.train_split
        )
        split_time = time.time() - split_start_time
        
        self.logger.info(f"Split dataset in {split_time:.2f} seconds")
        self.logger.info(f"Train samples: {len(train_annotations)}")
        self.logger.info(f"Val samples: {len(val_annotations)}")
        
        # Validate split distribution
        if len(train_annotations) == 0 or len(val_annotations) == 0:
            error_msg = f"Invalid split: train={len(train_annotations)}, val={len(val_annotations)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
            
        # Check if split ratio is close to requested
        actual_train_ratio = len(train_annotations) / (len(train_annotations) + len(val_annotations))
        if abs(actual_train_ratio - self.train_split) > 0.1:  # Allow 10% deviation
            self.logger.warning(
                f"Actual train split ratio ({actual_train_ratio:.2f}) differs significantly "
                f"from requested ratio ({self.train_split:.2f})"
            )
        
        # Load existing class mapping if requested
        loaded_mapping = None
        if self.load_existing_mapping and self.mapping_file.exists():
            try:
                self.logger.info(f"Loading existing class mapping from {self.mapping_file}...")
                loaded_mapping = self.load_class_mapping()
                self.logger.info(f"Loaded existing class mapping with {len(loaded_mapping)} classes")
            except Exception as e:
                self.logger.warning(f"Failed to load class mapping from {self.mapping_file}: {e}")
                self.logger.warning("Will create new class mapping instead")

        # Create dataloaders with class mapping handling
        self.logger.info("Creating dataloaders...")
        dataloader_start_time = time.time()
        self.train_loader, self.val_loader, self.class_mapping = create_dataloaders(
            train_annotations,
            val_annotations,
            self.batch_size,
            self.num_workers,
            self.img_size,
            self.use_polygons,
            mapping_file=self.mapping_file if not loaded_mapping else None,
            class_mapping=loaded_mapping,
            allow_new_classes=self.allow_new_classes
        )
        dataloader_time = time.time() - dataloader_start_time
        
        # Log class mapping information
        self.logger.info(f"Created dataloaders in {dataloader_time:.2f} seconds")
        self.logger.info(f"Using class mapping with {len(self.class_mapping)} classes")
        self.logger.info(f"Class mapping file: {self.mapping_file}")
        
        # Save class mapping with metadata if requested
        if self.save_mapping:
            try:
                self.logger.info("Saving class mapping with metadata...")
                saved_path = self.save_class_mapping()
                self.logger.info(f"Saved class mapping with metadata to {saved_path}")
            except Exception as e:
                self.logger.error(f"Failed to save class mapping to {self.mapping_file}: {e}")
                
        # Log total setup time
        total_time = time.time() - start_time
        self.logger.info(f"Data module setup completed in {total_time:.2f} seconds")

        return self.train_loader, self.val_loader

    def get_class_names(self) -> List[str]:
        """Get ordered list of class names"""
        names = ['background']
        for name, idx in sorted(self.class_mapping.items(), key=lambda x: x[1]):
            names.append(name)
        return names
        
    def save_class_mapping(self, output_path: Optional[Path] = None) -> Path:
        """
        Save the class mapping to a JSON file with metadata.
        
        This method:
        1. Counts the number of samples per class in the dataset
        2. Creates a JSON structure with the class mapping, metadata, and samples per class
        3. Saves this structure to a JSON file in the project config directory
        
        Args:
            output_path: Path to save the class mapping JSON file. If None, uses self.mapping_file
            
        Returns:
            Path to the saved class mapping file
            
        Raises:
            ValueError: If class_mapping is None (setup() hasn't been called)
            IOError: If the file cannot be written
        """
        import datetime
        
        if self.class_mapping is None:
            raise ValueError("Class mapping is not available. Call setup() first.")
            
        # Use provided output_path or default to self.mapping_file
        output_path = output_path or self.mapping_file
        
        # Create directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Count samples per class
        samples_per_class = {}
        for annotation in self.all_annotations:
            for defect in annotation.defects:
                class_name = defect.name
                if class_name in self.class_mapping:
                    samples_per_class[class_name] = samples_per_class.get(class_name, 0) + 1
        
        # Create JSON structure
        mapping_data = {
            "metadata": {
                "creation_timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "dataset_name": "contact_lens_defects",
                "total_samples": len(self.all_annotations)
            },
            "class_mapping": self.class_mapping,
            "samples_per_class": samples_per_class
        }
        
        # Save to file
        try:
            with open(output_path, 'w') as f:
                json.dump(mapping_data, f, indent=2, sort_keys=True)
            self.logger.info(f"Saved class mapping with metadata to {output_path}")
            return output_path
        except IOError as e:
            self.logger.error(f"Failed to save class mapping to {output_path}: {e}")
            raise
            
    def load_class_mapping(self, input_path: Optional[Path] = None) -> Dict[str, int]:
        """
        Load class mapping from a JSON file.
        
        This method:
        1. Loads the class mapping from the JSON file
        2. Validates the loaded mapping
        3. Returns the mapping or raises appropriate errors
        
        Args:
            input_path: Path to the class mapping JSON file. If None, uses self.mapping_file
            
        Returns:
            Dict mapping defect names to class IDs
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            json.JSONDecodeError: If the file is not valid JSON
            KeyError: If the file doesn't have the expected structure
        """
        # Use provided input_path or default to self.mapping_file
        input_path = input_path or self.mapping_file
        
        if not input_path.exists():
            raise FileNotFoundError(f"Class mapping file not found: {input_path}")
            
        try:
            with open(input_path, 'r') as f:
                mapping_data = json.load(f)
                
            # Validate structure
            if "class_mapping" not in mapping_data:
                raise KeyError(f"Invalid class mapping file: 'class_mapping' key not found in {input_path}")
                
            class_mapping = mapping_data["class_mapping"]
            
            # Log metadata if available
            if "metadata" in mapping_data:
                metadata = mapping_data["metadata"]
                self.logger.info(f"Loaded class mapping created on {metadata.get('creation_timestamp', 'unknown date')}")
                self.logger.info(f"Dataset: {metadata.get('dataset_name', 'unknown')}, Total samples: {metadata.get('total_samples', 'unknown')}")
                
            # Log samples per class if available
            if "samples_per_class" in mapping_data:
                samples_per_class = mapping_data["samples_per_class"]
                self.logger.info(f"Loaded mapping with {len(samples_per_class)} classes")
                for class_name, count in sorted(samples_per_class.items(), key=lambda x: x[1], reverse=True):
                    self.logger.debug(f"Class {class_name}: {count} samples")
            
            # Store the loaded mapping
            self.class_mapping = class_mapping
            return class_mapping
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse class mapping file {input_path}: {e}")
            raise
        except KeyError as e:
            self.logger.error(f"Invalid class mapping file structure: {e}")
            raise