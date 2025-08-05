"""
Model Module for J&J Contact Lens Defect Detection
Advanced plug-and-play architecture with domain-specific optimizations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json
import yaml
from dataclasses import dataclass
import numpy as np
import cv2
import time
import logging
from collections import OrderedDict, defaultdict
from datetime import datetime

# Import detection frameworks
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

try:
    import torchvision
    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False


@dataclass
class ModelConfig:
    """Configuration for a detection model"""
    name: str
    model_type: str  # 'yolov8', 'faster_rcnn', 'micro_defect_specialist'
    backbone: str
    num_classes: int
    input_size: int
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.5
    pretrained: bool = True
    checkpoint_path: Optional[str] = None
    # Advanced configurations
    use_synthetic_pretraining: bool = True
    use_jj_loss: bool = True
    ensemble_member: bool = False
    specialization: Optional[str] = None  # 'edge', 'center', 'presentation'
    extra_params: Dict[str, Any] = None


# ==================== Advanced Loss Functions ====================

class JJAwareLoss(nn.Module):
    """Loss function incorporating J&J business rules"""

    def __init__(self, pixel_threshold: float = 5.0, device: str = 'cuda'):
        super().__init__()
        self.pixel_threshold = pixel_threshold
        self.device = device

        # J&J specific weights
        self.always_fail_weight = 10.0
        self.near_threshold_weight = 5.0
        self.size_weights = {
            'micro': 3.0,    # < 10 pixels
            'small': 2.0,    # 10-50 pixels
            'medium': 1.5,   # 50-200 pixels
            'large': 1.0     # > 200 pixels
        }

    def forward(self, predictions: Dict, targets: List[Dict]) -> Dict[str, torch.Tensor]:
        """Compute J&J aware losses"""
        losses = {}

        # Extract predictions
        pred_boxes = predictions.get('boxes', [])
        pred_scores = predictions.get('scores', [])
        pred_labels = predictions.get('labels', [])

        # 1. Size-aware detection loss
        size_weights = self._compute_size_weights(targets)
        losses['loss_detection'] = self._focal_loss_weighted(
            pred_scores, targets, size_weights
        )

        # 2. Threshold-aware regression loss
        losses['loss_threshold'] = self._threshold_aware_regression(
            pred_boxes, targets
        )

        # 3. Critical defect priority loss
        losses['loss_critical'] = self._critical_defect_loss(
            pred_labels, targets
        )

        # 4. Measurement precision loss (for near-threshold cases)
        losses['loss_precision'] = self._measurement_precision_loss(
            pred_boxes, targets
        )

        return losses

    def _compute_size_weights(self, targets: List[Dict]) -> torch.Tensor:
        """Weight smaller defects more heavily"""
        weights = []
        for target in targets:
            boxes = target['boxes']
            for box in boxes:
                area = (box[2] - box[0]) * (box[3] - box[1])
                if area < 100:  # < 10x10 pixels
                    weights.append(self.size_weights['micro'])
                elif area < 2500:  # < 50x50 pixels
                    weights.append(self.size_weights['small'])
                elif area < 40000:  # < 200x200 pixels
                    weights.append(self.size_weights['medium'])
                else:
                    weights.append(self.size_weights['large'])

        return torch.tensor(weights, device=self.device)

    def _threshold_aware_regression(self, pred_boxes: torch.Tensor,
                                  targets: List[Dict]) -> torch.Tensor:
        """Asymmetric loss around J&J pixel thresholds"""
        if len(pred_boxes) == 0:
            return torch.tensor(0.0, device=self.device)

        total_loss = 0.0
        for pred, target in zip(pred_boxes, targets):
            gt_boxes = target['boxes']

            # Compute sizes
            pred_sizes = torch.max(pred[:, 2:] - pred[:, :2], dim=1)[0]
            gt_sizes = torch.max(gt_boxes[:, 2:] - gt_boxes[:, :2], dim=1)[0]

            # Asymmetric penalties
            errors = pred_sizes - gt_sizes

            # Critical: False accepts (missing defects > 5px)
            false_accept = (gt_sizes > self.pixel_threshold) & (pred_sizes < self.pixel_threshold)

            # Bad but not critical: False rejects
            false_reject = (gt_sizes < self.pixel_threshold) & (pred_sizes > self.pixel_threshold)

            # Near threshold: Both within 4-6 pixels
            near_threshold = (torch.abs(gt_sizes - self.pixel_threshold) < 1) | \
                           (torch.abs(pred_sizes - self.pixel_threshold) < 1)

            loss = torch.where(
                false_accept,
                self.always_fail_weight * errors**2,
                torch.where(
                    false_reject,
                    2.0 * errors**2,
                    torch.where(
                        near_threshold,
                        self.near_threshold_weight * errors**2,
                        errors**2
                    )
                )
            )

            total_loss += loss.mean()

        return total_loss

    def _critical_defect_loss(self, pred_labels: torch.Tensor,
                            targets: List[Dict]) -> torch.Tensor:
        """Heavy penalty for missing always-fail defects"""
        # Define critical defect class IDs (would be configured)
        critical_classes = [10, 11, 12, 13]  # folded, inverted, missing, multiple

        loss = 0.0
        for target in targets:
            gt_labels = target['labels']
            critical_mask = torch.isin(gt_labels, torch.tensor(critical_classes))

            if critical_mask.any():
                # Must detect ALL critical defects
                loss += self.always_fail_weight * (1 - critical_mask.float()).sum()

        return loss

    def _measurement_precision_loss(self, pred_boxes: torch.Tensor,
                                  targets: List[Dict]) -> torch.Tensor:
        """Extra precision for near-threshold measurements"""
        # Focus on getting exact measurements for 4-6 pixel defects
        precision_loss = 0.0

        for pred, target in zip(pred_boxes, targets):
            gt_boxes = target['boxes']
            gt_sizes = torch.max(gt_boxes[:, 2:] - gt_boxes[:, :2], dim=1)[0]

            # Near threshold mask
            near_mask = (gt_sizes >= 4) & (gt_sizes <= 6)

            if near_mask.any():
                # Use IoU loss for precise localization
                iou = self._box_iou(pred[near_mask], gt_boxes[near_mask])
                precision_loss += (1 - iou).mean()

        return precision_loss

    def _box_iou(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """Compute IoU between boxes"""
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

        lt = torch.max(boxes1[:, :2], boxes2[:, :2])
        rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])

        wh = (rb - lt).clamp(min=0)
        inter = wh[:, 0] * wh[:, 1]

        return inter / (area1 + area2 - inter)

    def _focal_loss_weighted(self, predictions: torch.Tensor,
                           targets: List[Dict],
                           weights: torch.Tensor) -> torch.Tensor:
        """Weighted focal loss for imbalanced detection"""
        # Simplified focal loss implementation
        alpha = 0.25
        gamma = 2.0

        # This would be properly implemented with the detection head outputs
        # For now, return placeholder
        return torch.tensor(0.0, device=self.device)


# ==================== Micro-Defect Architecture Components ====================

class MicroDefectBackbone(nn.Module):
    """Specialized backbone for tiny defect detection"""

    def __init__(self, grayscale: bool = True):
        super().__init__()
        in_channels = 1 if grayscale else 3

        # Modified stem for high-resolution preservation
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Minimal downsampling
            nn.MaxPool2d(2, stride=2)  # Only 2x downsample
        )

        # Parallel processing at multiple scales
        self.micro_branch = nn.ModuleList([
            self._make_dilated_block(64, 128, dilation=1),
            self._make_dilated_block(128, 128, dilation=2),
            self._make_dilated_block(128, 128, dilation=4),
        ])

        # High-resolution feature pyramid
        self.fpn = nn.ModuleDict({
            'P1': nn.Conv2d(128, 256, 1),  # 1/2 resolution
            'P2': nn.Conv2d(256, 256, 1),  # 1/4 resolution
            'P3': nn.Conv2d(512, 256, 1),  # 1/8 resolution
        })

    def _make_dilated_block(self, in_channels: int, out_channels: int,
                           dilation: int) -> nn.Module:
        """Dilated convolution block for multi-scale processing"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3,
                     padding=dilation, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3,
                     padding=dilation, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # High-resolution processing
        x = self.stem(x)

        # Multi-scale micro defect features
        micro_features = x
        feature_maps = {}

        for i, block in enumerate(self.micro_branch):
            micro_features = block(micro_features)
            if i == 0:
                feature_maps['feat_s2'] = micro_features
            elif i == len(self.micro_branch) - 1:
                feature_maps['feat_s4'] = F.max_pool2d(micro_features, 2)
                feature_maps['feat_s8'] = F.max_pool2d(feature_maps['feat_s4'], 2)

        # Build FPN
        fpn_features = OrderedDict()
        fpn_features['0'] = self.fpn['P1'](feature_maps['feat_s2'])
        fpn_features['1'] = self.fpn['P2'](feature_maps['feat_s4'])
        fpn_features['2'] = self.fpn['P3'](feature_maps['feat_s8'])

        return fpn_features


# ==================== Synthetic Pre-training ====================

class SyntheticDefectGenerator:
    """Generate synthetic defects for pre-training"""

    def __init__(self, base_images_path: Path):
        self.base_images_path = base_images_path
        self.rng = np.random.RandomState(42)

    def generate_synthetic_dataset(self, num_samples: int = 10000) -> List[Dict]:
        """Generate synthetic defect dataset"""
        synthetic_data = []

        defect_generators = {
            'edge_tear': self._generate_edge_tear,
            'center_debris': self._generate_debris,
            'bubble': self._generate_bubble,
            'scratch': self._generate_scratch,
            'edge_chip': self._generate_edge_chip
        }

        for i in range(num_samples):
            # Load random clean lens image
            base_image = self._load_random_clean_lens()

            # Apply random defects
            num_defects = self.rng.randint(0, 4)
            annotations = []

            for _ in range(num_defects):
                defect_type = self.rng.choice(list(defect_generators.keys()))
                defect_image, annotation = defect_generators[defect_type](base_image)
                annotations.append(annotation)
                base_image = defect_image

            synthetic_data.append({
                'image': base_image,
                'annotations': annotations
            })

        return synthetic_data

    def _generate_edge_tear(self, image: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Generate synthetic edge tear"""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        radius = min(h, w) // 2 - 100

        # Random angle for tear location
        angle = self.rng.uniform(0, 2 * np.pi)

        # Tear parameters
        tear_length = self.rng.randint(5, 50)  # Match J&J thresholds
        tear_width = self.rng.randint(1, 3)

        # Create tear
        start_x = int(center[0] + radius * np.cos(angle))
        start_y = int(center[1] + radius * np.sin(angle))

        end_x = int(start_x + tear_length * np.cos(angle))
        end_y = int(start_y + tear_length * np.sin(angle))

        # Draw tear
        cv2.line(image, (start_x, start_y), (end_x, end_y), 0, tear_width)

        # Create annotation
        annotation = {
            'type': 'edge_tear',
            'bbox': [min(start_x, end_x), min(start_y, end_y),
                    max(start_x, end_x), max(start_y, end_y)],
            'depth': tear_length
        }

        return image, annotation

    def _generate_debris(self, image: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Generate synthetic debris"""
        h, w = image.shape[:2]

        # Random debris size (around J&J threshold)
        size = self.rng.randint(3, 20)

        # Random location (avoiding edges)
        x = self.rng.randint(size, w - size)
        y = self.rng.randint(size, h - size)

        # Random irregular shape
        num_points = self.rng.randint(5, 10)
        angles = np.sort(self.rng.uniform(0, 2 * np.pi, num_points))

        points = []
        for angle in angles:
            r = self.rng.uniform(size * 0.3, size * 0.7)
            px = int(x + r * np.cos(angle))
            py = int(y + r * np.sin(angle))
            points.append([px, py])

        points = np.array(points, dtype=np.int32)

        # Draw debris
        cv2.fillPoly(image, [points], 0)

        # Create annotation
        annotation = {
            'type': 'debris',
            'bbox': [points[:, 0].min(), points[:, 1].min(),
                    points[:, 0].max(), points[:, 1].max()],
            'longest_dimension': size
        }

        return image, annotation

    def _generate_bubble(self, image: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Generate synthetic bubble"""
        # Implementation for bubble generation
        # Similar pattern to debris but circular
        pass

    def _generate_scratch(self, image: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Generate synthetic scratch using Bezier curves"""
        # Implementation for scratch generation
        pass

    def _generate_edge_chip(self, image: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Generate synthetic edge chip"""
        # Implementation for edge chip generation
        pass

    def _load_random_clean_lens(self) -> np.ndarray:
        """Load a random clean lens image"""
        # This would load from your clean lens dataset
        # For now, create a synthetic clean lens
        size = 2048
        image = np.ones((size, size), dtype=np.uint8) * 200

        # Add lens circle
        center = (size // 2, size // 2)
        radius = int(size * 0.4)  # 1580/2048 ≈ 0.77, leaving margin
        cv2.circle(image, center, radius, 150, -1)

        # Add some realistic texture
        noise = np.random.normal(0, 5, (size, size))
        image = np.clip(image + noise, 0, 255).astype(np.uint8)

        return image


# ==================== NEW: Tag Data Processing ====================

class TagDataProcessor:
    """Process tag/metadata according to Edge Algorithm requirements"""

    def __init__(self):
        self.tag_thresholds = {
            'manufacturing_date': 30,  # days
            'batch_variance': 0.05,
            'temperature_deviation': 2.0,
            # Add other tag-specific thresholds
        }

    def process_tag_data(self, image_detections: Dict, tag_data: Dict) -> Dict:
        """Enhance detections with tag data analysis"""
        # Extract tag features
        tag_features = self._extract_tag_features(tag_data)

        # Modify detection confidence based on tag data
        if 'batch_risk_score' in tag_features:
            risk_factor = tag_features['batch_risk_score']
            # Lower confidence threshold for high-risk batches
            image_detections['adjusted_threshold'] = max(
                0.2,
                image_detections.get('confidence_threshold', 0.3) - (risk_factor * 0.1)
            )

        # Add tag-based automatic failures
        tag_failures = self._check_tag_failures(tag_data)
        if tag_failures:
            image_detections['tag_failures'] = tag_failures
            image_detections['pass_fail'] = 'FAIL'

        return image_detections

    def _extract_tag_features(self, tag_data: Dict) -> Dict:
        """Convert tag data to features"""
        features = {}

        # Manufacturing age
        if 'manufacturing_date' in tag_data:
            age_days = (datetime.now() - tag_data['manufacturing_date']).days
            features['age_risk'] = min(1.0, age_days / 30.0)

        # Batch statistics
        if 'batch_stats' in tag_data:
            features['batch_risk_score'] = tag_data['batch_stats'].get('defect_rate', 0)

        return features

    def _check_tag_failures(self, tag_data: Dict) -> List[str]:
        """Check for automatic failures based on tag data"""
        failures = []

        for key, threshold in self.tag_thresholds.items():
            if key in tag_data and tag_data[key] > threshold:
                failures.append(f"Tag failure: {key} exceeds threshold")

        return failures


# ==================== NEW: Enhanced Data Splitting ====================

class StratifiedDataSplitter:
    """Advanced splitting strategy for extreme imbalance"""

    def __init__(self, min_samples_per_class: Dict[str, int]):
        self.min_samples_per_class = min_samples_per_class

    def create_splits(self, dataset, labels, split_ratios=(0.7, 0.15, 0.15)):
        """Create balanced splits with guaranteed minimum representation"""

        # Group samples by class
        class_indices = defaultdict(list)
        for idx, label_tensor in enumerate(labels):
            # Handle multi-label case
            unique_labels = label_tensor.unique().tolist()
            for label in unique_labels:
                class_indices[label].append(idx)

        # Analyze class distribution
        class_counts = {k: len(v) for k, v in class_indices.items()}
        print("Class distribution:", class_counts)

        # Split strategy based on class frequency
        train_indices, val_indices, test_indices = [], [], []

        for class_id, indices in class_indices.items():
            n_samples = len(indices)

            if n_samples < 10:  # Ultra-rare classes
                # Put ALL in train, synthesize for val/test
                train_indices.extend(indices)
                print(f"Class {class_id}: Ultra-rare, all {n_samples} samples in train")

            elif n_samples < 100:  # Rare classes
                # 80% train, 10% val, 10% test
                n_train = int(0.8 * n_samples)
                n_val = max(1, int(0.1 * n_samples))
                n_test = max(1, n_samples - n_train - n_val)

                shuffled = np.random.permutation(indices)
                train_indices.extend(shuffled[:n_train])
                val_indices.extend(shuffled[n_train:n_train+n_val])
                test_indices.extend(shuffled[n_train+n_val:])

            elif n_samples < 1000:  # Medium frequency
                # Standard 70/15/15 split
                n_train = int(0.7 * n_samples)
                n_val = int(0.15 * n_samples)
                n_test = n_samples - n_train - n_val

                shuffled = np.random.permutation(indices)
                train_indices.extend(shuffled[:n_train])
                val_indices.extend(shuffled[n_train:n_train+n_val])
                test_indices.extend(shuffled[n_train+n_val:])

            else:  # High frequency (e.g., bubbles)
                # Downsample to reasonable size
                max_samples = self.min_samples_per_class.get(class_id, 5000)
                downsampled = np.random.choice(indices, min(n_samples, max_samples), replace=False)

                n_train = int(0.7 * len(downsampled))
                n_val = int(0.15 * len(downsampled))

                train_indices.extend(downsampled[:n_train])
                val_indices.extend(downsampled[n_train:n_train+n_val])
                test_indices.extend(downsampled[n_train+n_val:])

        return train_indices, val_indices, test_indices


# ==================== NEW: Augmentation Strategy ====================

class AugmentationStrategy:
    """Class-specific augmentation for rare defects"""

    def __init__(self):
        self.rare_defect_augmentations = {
            'folded': self._augment_folded,
            'inverted': self._augment_inverted,
            'edge_not_closed': self._augment_edge_defect
        }

    def augment_rare_classes(self, image, annotations, class_counts):
        """Apply aggressive augmentation for rare classes"""
        augmented_samples = []

        for ann in annotations:
            class_name = ann['class_name']
            class_count = class_counts.get(ann['class_id'], 0)

            # More augmentations for rarer classes
            if class_count < 100:
                n_augmentations = 10
            elif class_count < 500:
                n_augmentations = 5
            else:
                n_augmentations = 1

            # Apply class-specific augmentations
            if class_name in self.rare_defect_augmentations:
                aug_func = self.rare_defect_augmentations[class_name]
                for _ in range(n_augmentations):
                    aug_img, aug_ann = aug_func(image, ann)
                    augmented_samples.append((aug_img, aug_ann))

        return augmented_samples

    def _augment_folded(self, image, annotation):
        """Specific augmentation for folded defects"""
        # Implement folded-specific augmentation
        return image, annotation

    def _augment_inverted(self, image, annotation):
        """Specific augmentation for inverted defects"""
        # Implement inverted-specific augmentation
        return image, annotation

    def _augment_edge_defect(self, image, annotation):
        """Specific augmentation for edge defects"""
        # Implement edge-specific augmentation
        return image, annotation


# ==================== Existing: JJBalancedSampler ====================

class JJBalancedSampler:
    """Balanced sampling strategy for extreme class imbalance"""

    def __init__(self, dataset, min_samples_per_class: int = 50):
        self.dataset = dataset
        self.min_samples_per_class = min_samples_per_class

        # Analyze class distribution
        self.class_indices = self._build_class_indices()
        self.class_weights = self._compute_class_weights()

    def _build_class_indices(self) -> Dict[int, List[int]]:
        """Build mapping of class to sample indices"""
        class_indices = {}
        for idx, sample in enumerate(self.dataset):
            labels = sample['labels']
            for label in labels.unique():
                label_int = label.item()
                if label_int not in class_indices:
                    class_indices[label_int] = []
                class_indices[label_int].append(idx)
        return class_indices

    def _compute_class_weights(self) -> Dict[int, float]:
        """Compute inverse frequency weights"""
        total_samples = len(self.dataset)
        weights = {}

        for class_id, indices in self.class_indices.items():
            count = len(indices)
            # Use sqrt to moderate extreme weights
            weights[class_id] = np.sqrt(total_samples / (count + 1))

        # Special handling for rare defects
        rare_classes = {
            5: 5.0,   # edge_not_closed (always fail)
            10: 5.0,  # folded
            11: 5.0,  # inverted
            12: 5.0,  # missing
            13: 5.0   # multiple
        }

        for class_id, weight in rare_classes.items():
            if class_id in weights:
                weights[class_id] = max(weights[class_id], weight)

        return weights

    def get_balanced_batch_sampler(self, batch_size: int):
        """Get a balanced batch sampler"""
        # Ensure minimum representation
        samples_per_class = max(
            batch_size // len(self.class_indices),
            self.min_samples_per_class
        )

        # Build balanced batches
        batches = []
        for class_id, indices in self.class_indices.items():
            # Oversample rare classes
            if len(indices) < samples_per_class:
                sampled = np.random.choice(indices, samples_per_class, replace=True)
            else:
                sampled = np.random.choice(indices, samples_per_class, replace=False)
            batches.extend(sampled)

        # Shuffle within epoch
        np.random.shuffle(batches)

        return torch.utils.data.BatchSampler(
            torch.utils.data.SubsetRandomSampler(batches),
            batch_size,
            drop_last=True
        )

    def get_class_weights_tensor(self) -> torch.Tensor:
        """Get class weights as tensor for loss function"""
        num_classes = max(self.class_weights.keys()) + 1
        weights = torch.ones(num_classes)
        for class_id, weight in self.class_weights.items():
            weights[class_id] = weight
        return weights


# ==================== Base Detector with J&J Enhancements ====================

class BaseDetector(ABC):
    """Enhanced base class for all detectors"""

    def __init__(self, config: ModelConfig):
        from armor_pipeline.utils.device_manager import DeviceManager
        
        self.config = config
        self.model = None
        self.device, self.batch_size = DeviceManager.get_optimal_device()

        # J&J specific components
        if config.use_jj_loss:
            self.criterion = JJAwareLoss(device=self.device)

        # Synthetic pre-training
        if config.use_synthetic_pretraining:
            self.synthetic_generator = SyntheticDefectGenerator(
                Path('./synthetic_base_images')
            )

    @abstractmethod
    def build_model(self):
        """Build the model architecture"""
        pass

    def pretrain_synthetic(self, num_epochs: int = 10):
        """Pre-train on synthetic defects"""
        print("Generating synthetic defect dataset...")
        synthetic_data = self.synthetic_generator.generate_synthetic_dataset(10000)

        # Create synthetic dataloader
        # This would be properly implemented with your DataModule

        print(f"Pre-training on {len(synthetic_data)} synthetic samples...")
        # Training loop implementation

    @abstractmethod
    def train_step(self, images: torch.Tensor, targets: List[Dict]) -> Dict[str, float]:
        """Single training step"""
        pass

    @abstractmethod
    def predict(self, images: torch.Tensor) -> List[Dict]:
        """Run inference on images"""
        pass

    @abstractmethod
    def save_checkpoint(self, path: Path):
        """Save model checkpoint"""
        pass

    @abstractmethod
    def load_checkpoint(self, path: Path):
        """Load model checkpoint"""
        pass

    def to(self, device):
        """Move model to device"""
        self.device = device
        if self.model:
            self.model = self.model.to(device)
        if hasattr(self, 'criterion'):
            self.criterion = self.criterion.to(device)
        return self


# ==================== Enhanced Detector Implementations ====================

class MicroDefectSpecialist(BaseDetector):
    """Specialized model for micro defects using custom backbone"""

    def build_model(self):
        """Build micro-defect specialized model"""
        # Use custom backbone
        backbone = MicroDefectBackbone(grayscale=True)

        # Create Faster R-CNN with custom backbone
        self.model = torchvision.models.detection.FasterRCNN(
            backbone,
            num_classes=self.config.num_classes,
            min_size=2048,  # Keep high resolution
            max_size=2048,
            # Anchor sizes for tiny defects
            rpn_anchor_generator=self._get_micro_anchor_generator(),
            # More proposals for small objects
            rpn_pre_nms_top_n_train=4000,
            rpn_pre_nms_top_n_test=2000,
            rpn_post_nms_top_n_train=2000,
            rpn_post_nms_top_n_test=1000,
        )

    def _get_micro_anchor_generator(self):
        """Anchor generator for micro defects"""
        from torchvision.models.detection.anchor_utils import AnchorGenerator

        # Tiny anchor sizes for 2-50 pixel defects
        anchor_sizes = (
            (4, 8, 16),      # P1 level
            (16, 32, 64),    # P2 level
            (64, 128, 256)   # P3 level
        )
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)

        return AnchorGenerator(anchor_sizes, aspect_ratios)

    def train_step(self, images: torch.Tensor, targets: List[Dict]) -> Dict[str, float]:
        """Training step with J&J aware loss"""
        self.model.train()

        # Standard detection losses
        detection_losses = self.model(images, targets)

        # Add J&J specific losses
        if self.config.use_jj_loss:
            # Get model predictions for J&J loss computation
            self.model.eval()
            with torch.no_grad():
                predictions = self.model(images)
            self.model.train()

            jj_losses = self.criterion(predictions[0], targets)

            # Combine losses
            for key, value in jj_losses.items():
                detection_losses[key] = value

        # Total loss
        losses = sum(loss for loss in detection_losses.values())
        losses.backward()

        return {k: v.item() for k, v in detection_losses.items()}

    def predict(self, images: torch.Tensor) -> List[Dict]:
        """Inference with micro-defect optimizations"""
        self.model.eval()

        with torch.no_grad():
            predictions = self.model(images)

        # Post-process for J&J requirements
        processed_predictions = []
        for pred in predictions:
            # Filter by confidence
            keep = pred['scores'] >= self.config.confidence_threshold

            # Additional filtering for micro defects
            boxes = pred['boxes'][keep]
            sizes = torch.max(boxes[:, 2:] - boxes[:, :2], dim=1)[0]

            # Keep all defects, but flag near-threshold ones
            near_threshold = (sizes >= 4) & (sizes <= 6)

            processed_pred = {
                'boxes': pred['boxes'][keep],
                'scores': pred['scores'][keep],
                'labels': pred['labels'][keep],
                'near_threshold': near_threshold,
                'sizes': sizes
            }
            processed_predictions.append(processed_pred)

        return processed_predictions

    def save_checkpoint(self, path: Path):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config.__dict__,
            'backbone_type': 'micro_defect'
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: Path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])


class YOLOv8MicroDefect(BaseDetector):
    """YOLOv8 adapted for J&J micro defects"""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        if not YOLO_AVAILABLE:
            raise ImportError("Ultralytics YOLO not available")
        self.build_model()

    def build_model(self):
        """Build YOLOv8 with J&J modifications"""
        # Start with YOLOv8 large for better small object detection
        model_name = 'yolov8l.pt' if 'yolov8l' in self.config.backbone else 'yolov8m.pt'

        if self.config.checkpoint_path:
            self.model = YOLO(self.config.checkpoint_path)
        else:
            self.model = YOLO(model_name)

        # Modify for grayscale input
        if hasattr(self.model.model, 'model') and hasattr(self.model.model.model, '0'):
            first_conv = self.model.model.model[0].conv
            if first_conv.in_channels != 1:
                # Replace first conv layer for grayscale
                new_conv = nn.Conv2d(
                    1, first_conv.out_channels,
                    kernel_size=first_conv.kernel_size,
                    stride=first_conv.stride,
                    padding=first_conv.padding,
                    bias=first_conv.bias is not None
                )
                # Initialize with mean of RGB weights
                with torch.no_grad():
                    new_conv.weight.data = first_conv.weight.data.mean(dim=1, keepdim=True)

                self.model.model.model[0].conv = new_conv

        # Configure for J&J requirements
        self.model.model.nc = self.config.num_classes - 1  # -1 for background

    def train(self, data_yaml_path: str, epochs: int = 100, **kwargs):
        """Train with J&J optimizations"""
        # Override default training arguments for micro defects
        train_args = {
            'imgsz': self.config.input_size,
            'batch': kwargs.get('batch_size', 8),  # Smaller batch for high res
            'device': self.device,
            'mosaic': 0.5,  # Less aggressive mosaic for small objects
            'copy_paste': 0.3,  # Copy-paste augmentation for defects
            'degrees': 15,  # Limited rotation
            'translate': 0.1,  # Limited translation
            'scale': 0.1,  # Limited scaling to preserve pixel measurements
            'conf': 0.3,  # Lower confidence for micro defects
            'iou': 0.5,
            'max_det': 300,  # More detections per image
        }
        train_args.update(kwargs)

        # Synthetic pre-training if enabled
        if self.config.use_synthetic_pretraining and epochs > 20:
            print("Starting synthetic pre-training phase...")
            # First 20% epochs on synthetic data
            synthetic_epochs = int(epochs * 0.2)
            self.pretrain_synthetic(synthetic_epochs)

            # Remaining epochs on real data
            epochs = epochs - synthetic_epochs

        results = self.model.train(data=data_yaml_path, epochs=epochs, **train_args)
        return results

    def predict(self, images: torch.Tensor) -> List[Dict]:
        """YOLOv8 inference optimized for micro defects"""
        # Convert torch tensor to numpy if needed
        if isinstance(images, torch.Tensor):
            # Handle grayscale conversion
            if images.shape[1] == 1:  # [B, 1, H, W]
                images = images.repeat(1, 3, 1, 1)  # Convert to pseudo-RGB

            # Denormalize
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            images = images * std + mean
            images = (images * 255).byte()

            # Convert to numpy
            images_np = images.cpu().numpy().transpose(0, 2, 3, 1)
        else:
            images_np = images

        # Run prediction with micro-defect settings
        results = self.model.predict(
            images_np,
            conf=0.3,  # Lower threshold for micro defects
            iou=self.config.nms_threshold,
            max_det=300,
            device=self.device,
            augment=True  # Test-time augmentation for better small object detection
        )

        # Convert to standard format with J&J enhancements
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                # Calculate defect sizes
                box_coords = boxes.xyxy.cpu()
                sizes = torch.max(
                    box_coords[:, 2:] - box_coords[:, :2],
                    dim=1
                )[0]

                # Flag near-threshold defects
                near_threshold = (sizes >= 4) & (sizes <= 6)

                detection = {
                    'boxes': box_coords,
                    'scores': boxes.conf.cpu(),
                    'labels': boxes.cls.cpu().int() + 1,  # Add 1 for background
                    'sizes': sizes,
                    'near_threshold': near_threshold
                }
            else:
                detection = {
                    'boxes': torch.empty((0, 4)),
                    'scores': torch.empty(0),
                    'labels': torch.empty(0, dtype=torch.int),
                    'sizes': torch.empty(0),
                    'near_threshold': torch.empty(0, dtype=torch.bool)
                }
            detections.append(detection)

        return detections

    def save_checkpoint(self, path: Path):
        """Save YOLOv8 checkpoint"""
        self.model.save(str(path))

    def load_checkpoint(self, path: Path):
        """Load YOLOv8 checkpoint"""
        self.model = YOLO(str(path))


class FasterRCNNDetector(BaseDetector):
    """Enhanced Faster R-CNN with J&J optimizations"""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        if not TORCHVISION_AVAILABLE:
            raise ImportError("torchvision not available")
        self.build_model()
        self.optimizer = None

    def build_model(self):
        """Build Faster R-CNN model with J&J enhancements"""
        # Use micro-defect backbone if specified
        if self.config.backbone == 'micro_defect':
            backbone = MicroDefectBackbone(grayscale=True)
            self.model = torchvision.models.detection.FasterRCNN(
                backbone,
                num_classes=self.config.num_classes,
                min_size=self.config.input_size,
                max_size=self.config.input_size,
                rpn_anchor_generator=self._get_anchor_generator(),
            )
        else:
            # Standard backbone with modifications
            if self.config.pretrained:
                self.model = fasterrcnn_resnet50_fpn_v2(weights='DEFAULT')
            else:
                self.model = fasterrcnn_resnet50_fpn_v2(weights=None)

            # Replace classifier head
            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            self.model.roi_heads.box_predictor = FastRCNNPredictor(
                in_features, self.config.num_classes
            )

            # Modify first conv for grayscale if needed
            if hasattr(self.model.backbone, 'body'):
                first_conv = self.model.backbone.body.conv1
                if first_conv.in_channels != 1:
                    new_conv = nn.Conv2d(
                        1, first_conv.out_channels,
                        kernel_size=first_conv.kernel_size,
                        stride=first_conv.stride,
                        padding=first_conv.padding,
                        bias=False
                    )
                    new_conv.weight.data = first_conv.weight.data.mean(dim=1, keepdim=True)
                    self.model.backbone.body.conv1 = new_conv

        # Load checkpoint if provided
        if self.config.checkpoint_path:
            self.load_checkpoint(Path(self.config.checkpoint_path))

    def _get_anchor_generator(self):
        """Get anchor generator optimized for J&J defects"""
        from torchvision.models.detection.anchor_utils import AnchorGenerator

        # Multi-scale anchors for various defect sizes
        anchor_sizes = ((4, 8, 16, 32),) * 5  # Smaller anchors for micro defects
        aspect_ratios = ((0.5, 1.0, 2.0),) * 5

        return AnchorGenerator(anchor_sizes, aspect_ratios)

    def train_step(self, images: torch.Tensor, targets: List[Dict]) -> Dict[str, float]:
        """Single training step with J&J aware loss"""
        from armor_pipeline.utils.bbox_utils import convert_bbox_format
        
        self.model.train()

        # Convert COCO format to Faster R-CNN format
        converted_targets = []
        for target in targets:
            # Convert bbox to xyxy format (Faster R-CNN format)
            boxes = target['boxes'].clone()
            if len(boxes) > 0 and boxes.shape[1] == 4:
                # Use the utility function to convert to xyxy format
                boxes = convert_bbox_format(boxes, source_format=None, target_format='xyxy', inplace=True)

            converted_target = {
                'boxes': boxes,
                'labels': target['labels']
            }
            converted_targets.append(converted_target)

        # Forward pass
        loss_dict = self.model(images, converted_targets)

        # Add J&J specific losses if enabled
        if self.config.use_jj_loss:
            # Get predictions for J&J loss
            self.model.eval()
            with torch.no_grad():
                predictions = self.model(images)
            self.model.train()

            jj_losses = self.criterion(predictions[0], converted_targets)

            # Weight J&J losses
            for key, value in jj_losses.items():
                loss_dict[f'jj_{key}'] = value * 0.5  # Weight appropriately

        # Calculate total loss
        losses = sum(loss for loss in loss_dict.values())

        # Extract individual losses for logging
        loss_values = {key: loss.item() for key, loss in loss_dict.items()}
        loss_values['loss_total'] = losses.item()

        # Backward pass
        losses.backward()

        return loss_values

    def predict(self, images: torch.Tensor) -> List[Dict]:
        """Run Faster R-CNN inference with J&J optimizations"""
        self.model.eval()

        with torch.no_grad():
            predictions = self.model(images)

        # Filter and enhance predictions
        filtered_predictions = []
        for pred in predictions:
            keep = pred['scores'] >= self.config.confidence_threshold

            boxes = pred['boxes'][keep]
            scores = pred['scores'][keep]
            labels = pred['labels'][keep]

            # Calculate sizes for J&J requirements
            if len(boxes) > 0:
                sizes = torch.max(boxes[:, 2:] - boxes[:, :2], dim=1)[0]
                near_threshold = (sizes >= 4) & (sizes <= 6)
            else:
                sizes = torch.empty(0)
                near_threshold = torch.empty(0, dtype=torch.bool)

            filtered_pred = {
                'boxes': boxes,
                'scores': scores,
                'labels': labels,
                'sizes': sizes,
                'near_threshold': near_threshold
            }
            filtered_predictions.append(filtered_pred)

        return filtered_predictions

    def save_checkpoint(self, path: Path):
        """Save Faster R-CNN checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config.__dict__,
            'use_jj_loss': self.config.use_jj_loss
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: Path):
        """Load Faster R-CNN checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])


# ==================== Model Registry with Ensemble Support ====================

class ModelRegistry:
    """Enhanced registry with ensemble and specialization support"""

    SUPPORTED_MODELS = {
        'yolov8': YOLOv8MicroDefect,
        'yolov8_micro': YOLOv8MicroDefect,
        'faster_rcnn': FasterRCNNDetector,
        'micro_defect_specialist': MicroDefectSpecialist,
    }

    def __init__(self, registry_path: Path):
        self.registry_path = Path(registry_path)
        self.registry = self._load_registry()
        self._ensemble_cache = {}

    def _load_registry(self) -> Dict[str, ModelConfig]:
        """Load model registry from JSON/YAML"""
        if not self.registry_path.exists():
            # Create default registry
            self._create_default_registry()

        if self.registry_path.suffix == '.json':
            with open(self.registry_path) as f:
                data = json.load(f)
        elif self.registry_path.suffix in ['.yaml', '.yml']:
            with open(self.registry_path) as f:
                data = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported registry format: {self.registry_path.suffix}")

        # Convert to ModelConfig objects
        registry = {}
        for name, config_dict in data.items():
            # Handle extra_params properly
            if 'extra_params' not in config_dict:
                config_dict['extra_params'] = {}
            registry[name] = ModelConfig(**config_dict)

        return registry

    def _create_default_registry(self):
        """Create default J&J optimized registry"""
        default_configs = {
            # Edge Algorithm Models (Speed Priority)
            "yolov8m_edge": {
                "name": "yolov8m_edge",
                "model_type": "yolov8",
                "backbone": "yolov8m",
                "num_classes": 15,  # All J&J defect types
                "input_size": 1024,
                "confidence_threshold": 0.3,
                "nms_threshold": 0.5,
                "pretrained": True,
                "use_synthetic_pretraining": True,
                "use_jj_loss": True,
                "ensemble_member": False,
                "specialization": None
            },

            # Back Office Models (Accuracy Priority)
            "micro_defect_specialist": {
                "name": "micro_defect_specialist",
                "model_type": "micro_defect_specialist",
                "backbone": "micro_defect",
                "num_classes": 15,
                "input_size": 2048,
                "confidence_threshold": 0.25,
                "nms_threshold": 0.4,
                "pretrained": False,
                "use_synthetic_pretraining": True,
                "use_jj_loss": True,
                "ensemble_member": True,
                "specialization": "micro"
            },

            "edge_defect_specialist": {
                "name": "edge_defect_specialist",
                "model_type": "faster_rcnn",
                "backbone": "resnet50_fpn",
                "num_classes": 8,  # Only edge defect classes
                "input_size": 2048,
                "confidence_threshold": 0.3,
                "nms_threshold": 0.5,
                "pretrained": True,
                "use_synthetic_pretraining": True,
                "use_jj_loss": True,
                "ensemble_member": True,
                "specialization": "edge"
            },

            "presentation_checker": {
                "name": "presentation_checker",
                "model_type": "yolov8",
                "backbone": "yolov8s",  # Small model for quick checks
                "num_classes": 10,  # Presentation defects only
                "input_size": 512,  # Lower res for speed
                "confidence_threshold": 0.5,
                "nms_threshold": 0.5,
                "pretrained": True,
                "use_synthetic_pretraining": False,
                "use_jj_loss": False,
                "ensemble_member": True,
                "specialization": "presentation"
            },

            "yolo_nano_bubbles": {
                "name": "yolo_nano_bubbles",
                "model_type": "yolov8",
                "backbone": "yolov8n",  # nano model
                "num_classes": 2,  # background + bubble only
                "input_size": 640,  # smaller input for speed
                "confidence_threshold": 0.4,
                "nms_threshold": 0.5,
                "pretrained": True,
                "use_synthetic_pretraining": True,
                "use_jj_loss": False,  # Simple task, standard loss is fine
                "ensemble_member": True,
                "specialization": "bubbles",
                "extra_params": {
                    "target_inference_ms": 3,  # Target 3ms inference
                    "single_class_mode": True
                }
            }
        }

        # Save registry
        self.registry_path.parent.mkdir(exist_ok=True, parents=True)
        with open(self.registry_path, 'w') as f:
            json.dump(default_configs, f, indent=2)

    def save_registry(self):
        """Save registry to file"""
        data = {}
        for name, config in self.registry.items():
            config_dict = config.__dict__.copy()
            # Ensure extra_params is serializable
            if config_dict.get('extra_params') is None:
                config_dict['extra_params'] = {}
            data[name] = config_dict

        if self.registry_path.suffix == '.json':
            with open(self.registry_path, 'w') as f:
                json.dump(data, f, indent=2)
        else:
            with open(self.registry_path, 'w') as f:
                yaml.dump(data, f)

    def register_model(self, config: ModelConfig):
        """Register a new model configuration"""
        self.registry[config.name] = config
        self.save_registry()

    def get_model(self, name: str) -> BaseDetector:
        """Get model instance by name"""
        if name not in self.registry:
            raise ValueError(f"Model {name} not found in registry")

        config = self.registry[name]
        model_class = self.SUPPORTED_MODELS.get(config.model_type)

        if model_class is None:
            raise ValueError(f"Unsupported model type: {config.model_type}")

        return model_class(config)

    def get_ensemble(self, ensemble_name: str = "jj_precision_ensemble") -> 'ModelEnsemble':
        """Get ensemble of specialized models"""
        if ensemble_name in self._ensemble_cache:
            return self._ensemble_cache[ensemble_name]

        # Define ensemble compositions
        ensemble_configs = {
            "jj_precision_ensemble": [
                "presentation_checker",
                "micro_defect_specialist",
                "edge_defect_specialist"
            ],
            "speed_ensemble": [
                "yolov8m_edge",
                "presentation_checker"
            ],
            "hierarchical_ensemble": [
                "yolo_nano_bubbles",
                "presentation_checker",
                "micro_defect_specialist"
            ]
        }

        if ensemble_name not in ensemble_configs:
            raise ValueError(f"Unknown ensemble: {ensemble_name}")

        models = {}
        for model_name in ensemble_configs[ensemble_name]:
            models[model_name] = self.get_model(model_name)

        # Use HierarchicalEnsemble for specific types
        if ensemble_name == "hierarchical_ensemble":
            ensemble = HierarchicalEnsemble(models)
        else:
            ensemble = ModelEnsemble(models)

        self._ensemble_cache[ensemble_name] = ensemble

        return ensemble

    def list_models(self) -> List[str]:
        """List all registered models"""
        return list(self.registry.keys())

    def list_ensembles(self) -> List[str]:
        """List available ensembles"""
        return ["jj_precision_ensemble", "speed_ensemble", "hierarchical_ensemble"]


class ModelEnsemble:
    """Ensemble of specialized models for high precision"""

    def __init__(self, models: Dict[str, BaseDetector]):
        from armor_pipeline.utils.device_manager import DeviceManager
        
        self.models = models
        # Use ensemble-specific device selection that accounts for multiple models
        self.device, self.batch_size = DeviceManager.get_ensemble_optimal_device(num_models=len(models))

        # Move all models to device
        for model in self.models.values():
            model.to(self.device)

    def predict(self, images: torch.Tensor) -> List[Dict]:
        """Run ensemble prediction with intelligent merging"""
        all_predictions = {}

        # 1. Quick presentation check (if available)
        if "presentation_checker" in self.models:
            presentation_results = self.models["presentation_checker"].predict(images)
            # Early exit if presentation is bad
            for i, result in enumerate(presentation_results):
                if self._has_critical_presentation_issue(result):
                    return [{
                        'boxes': torch.empty((0, 4)),
                        'scores': torch.empty(0),
                        'labels': torch.empty(0),
                        'ensemble_decision': 'FAIL_PRESENTATION'
                    }]

        # 2. Run specialized models
        for name, model in self.models.items():
            if name != "presentation_checker":
                predictions = model.predict(images)
                all_predictions[name] = predictions

        # 3. Intelligent merging
        merged_predictions = self._merge_predictions(all_predictions)

        return merged_predictions

    def _has_critical_presentation_issue(self, result: Dict) -> bool:
        """Check for critical presentation issues"""
        critical_labels = [10, 11, 12, 13]  # folded, inverted, missing, multiple
        if 'labels' in result:
            return any(label in critical_labels for label in result['labels'].tolist())
        return False

    def _merge_predictions(self, all_predictions: Dict[str, List[Dict]]) -> List[Dict]:
        """Intelligently merge predictions from multiple models"""
        # Weight predictions based on model specialization
        specialization_weights = {
            'micro': {'micro_defects': 2.0, 'edge_defects': 0.5},
            'edge': {'edge_defects': 2.0, 'center_defects': 0.5},
            'bubbles': {'bubble': 3.0, 'others': 0.3}
        }

        merged = []

        # For each image in the batch
        for i in range(len(next(iter(all_predictions.values())))):
            combined_boxes = []
            combined_scores = []
            combined_labels = []
            original_scores = []  # Store original scores for reference

            # Process predictions from each model
            for model_name, predictions in all_predictions.items():
                if i < len(predictions):
                    pred = predictions[i]
                    model = self.models[model_name]
                    specialization = model.config.specialization

                    # Skip if no specialization or no predictions
                    if not specialization or len(pred['boxes']) == 0:
                        combined_boxes.append(pred['boxes'])
                        combined_scores.append(pred['scores'])
                        combined_labels.append(pred['labels'])
                        original_scores.append(pred['scores'])
                        continue

                    # Get weights for this model's specialization
                    if specialization in specialization_weights:
                        weights = specialization_weights[specialization]

                        # Apply weights to scores based on defect type
                        weighted_scores = []
                        for j, (score, label) in enumerate(zip(pred['scores'], pred['labels'])):
                            label_id = label.item()

                            # Map label ID to defect category
                            # This is a simplified mapping based on available information
                            if specialization == 'micro':
                                # Assume small defects (< 10 pixels) are micro_defects
                                # Others are edge_defects or center_defects
                                if 'sizes' in pred and pred['sizes'][j] < 10:
                                    weight = weights.get('micro_defects', 1.0)
                                else:
                                    weight = weights.get('edge_defects', 1.0)
                            elif specialization == 'edge':
                                # Assume edge defects have specific label range
                                # This is a guess - adjust based on actual label mapping
                                if 1 <= label_id <= 8:  # Edge defect specialist has 8 classes
                                    weight = weights.get('edge_defects', 1.0)
                                else:
                                    weight = weights.get('center_defects', 1.0)
                            elif specialization == 'bubbles':
                                # Bubble specialist only detects bubbles
                                if label_id == 1:  # Assuming label 1 is bubble
                                    weight = weights.get('bubble', 1.0)
                                else:
                                    weight = weights.get('others', 1.0)
                            else:
                                # Default weight if no specific mapping
                                weight = 1.0

                            weighted_scores.append(score * weight)

                        # Create tensor of weighted scores
                        weighted_scores = torch.tensor(weighted_scores,
                                                      device=pred['scores'].device,
                                                      dtype=pred['scores'].dtype)

                        combined_boxes.append(pred['boxes'])
                        combined_scores.append(weighted_scores)
                        combined_labels.append(pred['labels'])
                        original_scores.append(pred['scores'])
                    else:
                        # No weights for this specialization, use original scores
                        combined_boxes.append(pred['boxes'])
                        combined_scores.append(pred['scores'])
                        combined_labels.append(pred['labels'])
                        original_scores.append(pred['scores'])

            if combined_boxes:
                merged_boxes = torch.cat(combined_boxes, dim=0)
                merged_scores = torch.cat(combined_scores, dim=0)
                merged_labels = torch.cat(combined_labels, dim=0)
                merged_original_scores = torch.cat(original_scores, dim=0)

                # Apply NMS using weighted scores
                keep = torchvision.ops.nms(merged_boxes, merged_scores, 0.5)

                merged.append({
                    'boxes': merged_boxes[keep],
                    'scores': merged_scores[keep],
                    'labels': merged_labels[keep],
                    'original_scores': merged_original_scores[keep],
                    'ensemble_decision': 'WEIGHTED_MERGE'
                })
            else:
                merged.append({
                    'boxes': torch.empty((0, 4)),
                    'scores': torch.empty(0),
                    'labels': torch.empty(0),
                    'original_scores': torch.empty(0),
                    'ensemble_decision': 'NO_DETECTIONS'
                })

        return merged


# ==================== NEW: Hierarchical Ensemble ====================

class HierarchicalEnsemble(ModelEnsemble):
    """Optimized hierarchical ensemble for inference speed"""

    def __init__(self, models: Dict[str, BaseDetector]):
        super().__init__(models)

        # Define cascade order and early exit conditions
        self.cascade_order = [
            ('yolo_nano_bubbles', self._check_bubble_only),
            ('presentation_checker', self._check_presentation),
            ('micro_defect_specialist', self._check_micro_defects)
        ]

    def predict(self, images: torch.Tensor) -> List[Dict]:
        """Cascaded prediction with early exits"""
        batch_size = images.shape[0]
        final_predictions = [None] * batch_size
        remaining_indices = list(range(batch_size))

        for model_name, early_exit_check in self.cascade_order:
            if not remaining_indices or model_name not in self.models:
                break

            # Get subset of images still needing processing
            subset_images = images[remaining_indices]

            # Run model
            model_start = time.time()
            predictions = self.models[model_name].predict(subset_images)
            inference_ms = (time.time() - model_start) * 1000

            # Check which images can exit early
            new_remaining = []
            for i, pred_idx in enumerate(remaining_indices):
                pred = predictions[i]
                pred['inference_ms'] = inference_ms

                if early_exit_check(pred):
                    final_predictions[pred_idx] = pred
                else:
                    new_remaining.append(pred_idx)

            remaining_indices = new_remaining

        # Process any remaining images with full ensemble
        if remaining_indices:
            subset_images = images[remaining_indices]
            full_predictions = super().predict(subset_images)
            for i, pred_idx in enumerate(remaining_indices):
                final_predictions[pred_idx] = full_predictions[i]

        return final_predictions

    def _check_bubble_only(self, prediction: Dict) -> bool:
        """Check if image only has bubbles (can skip other models)"""
        if len(prediction['labels']) == 0:
            return True  # No defects, done

        # If only bubbles detected with high confidence
        bubble_label = 1  # Adjust based on your label mapping
        all_bubbles = all(label == bubble_label for label in prediction['labels'])
        high_confidence = all(score > 0.8 for score in prediction['scores'])

        return all_bubbles and high_confidence

    def _check_presentation(self, prediction: Dict) -> bool:
        """Check if presentation check is sufficient"""
        # Check for critical presentation issues
        if self._has_critical_presentation_issue(prediction):
            return True  # Critical issue found, no need for further checks

        # No critical issues and no other defects
        return len(prediction['labels']) == 0

    def _check_micro_defects(self, prediction: Dict) -> bool:
        """Always process through micro defect specialist - no early exit"""
        return False


# ==================== NEW: Edge Algorithm Router with Tag Processing ====================

class EdgeAlgorithmRouter:
    """Routes different data types to appropriate models per J&J Edge Algorithm requirements"""

    def __init__(self, registry: ModelRegistry):
        from armor_pipeline.utils.device_manager import DeviceManager
        from armor_pipeline.utils.performance_logger import PerformanceLogger
        
        # Initialize performance logger
        self.logger = PerformanceLogger.get_instance().get_logger(__name__)
        self.perf_logger = PerformanceLogger.get_instance()
        
        self.registry = registry
        self.device, self.batch_size = DeviceManager.get_optimal_device()

        # Fast presentation checker (must complete in 200ms total)
        self.presentation_model = registry.get_model("presentation_checker")
        self.presentation_model.to(self.device)

        # Main edge detection model
        self.edge_model = registry.get_model("yolov8m_edge")
        self.edge_model.to(self.device)

        # Tag data handler (for Edge Algorithm specific processing)
        self.tag_processor = TagDataProcessor()  # Now properly initialized

    def process_image(self, image: torch.Tensor, tag_data: Optional[Dict] = None) -> Dict:
        """Process image within 200ms constraint"""
        start_time = time.time()

        try:
            # 1. Quick presentation check (50ms budget)
            presentation_result = self.presentation_model.predict(image)

            # 2. Check for critical presentation issues
            if self._has_critical_issue(presentation_result[0]):
                return {
                    'pass_fail': 'FAIL',
                    'reason': 'presentation_deficient',
                    'time_ms': (time.time() - start_time) * 1000
                }

            # 3. Main defect detection (remaining time budget)
            defect_result = self.edge_model.predict(image)

            # 4. Process tag data if provided
            if tag_data:
                defect_result = self._process_tag_data(defect_result, tag_data)

            # 5. Make pass/fail decision
            decision = self._make_decision(defect_result[0])

            # Calculate inference time
            inference_time_ms = (time.time() - start_time) * 1000
            
            # Log performance metrics
            self.perf_logger.log_inference_time("EdgeAlgorithmRouter", inference_time_ms)
            
            # Log memory usage if CUDA is available
            if torch.cuda.is_available():
                allocated_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                reserved_mb = torch.cuda.memory_reserved() / (1024 * 1024)
                self.perf_logger.log_memory_usage("EdgeAlgorithmRouter", allocated_mb, reserved_mb)
            
            return {
                'pass_fail': decision,
                'detections': defect_result[0],
                'time_ms': inference_time_ms
            }
        except torch.cuda.OutOfMemoryError:
            # Handle out of memory error
            self.perf_logger.log_warning("CUDA out of memory error, clearing cache and retrying with fallback")
            torch.cuda.empty_cache()
            # Retry with smaller batch or CPU
            return self._process_fallback(image, tag_data)
        except Exception as e:
            # Handle general exceptions
            self.perf_logger.log_error(f"Processing failed", exception=e)
            
            # Calculate inference time
            inference_time_ms = (time.time() - start_time) * 1000
            
            return {
                'pass_fail': 'ERROR',
                'reason': str(e),
                'time_ms': inference_time_ms
            }

    def _has_critical_issue(self, result: Dict) -> bool:
        """Check for automatic fail conditions"""
        critical_labels = [10, 11, 12, 13]  # folded, inverted, missing, multiple
        return any(label in result.get('labels', []).tolist() for label in critical_labels)

    def _process_tag_data(self, detections: List[Dict], tag_data: Dict) -> List[Dict]:
        """Process tag-specific data per Edge Algorithm requirements"""
        for i, detection in enumerate(detections):
            detections[i] = self.tag_processor.process_tag_data(detection, tag_data)
        return detections

    def _make_decision(self, detections: Dict) -> str:
        """Make pass/fail decision based on J&J thresholds"""
        if not detections['boxes'].numel():
            return 'PASS'
        
        # Check sizes against 5-pixel threshold
        sizes = detections.get('sizes', torch.empty(0))
        if (sizes >= 5).any():
            return 'FAIL'
        
        return 'PASS'
        
    def _process_fallback(self, image: torch.Tensor, tag_data: Optional[Dict] = None) -> Dict:
        """Fallback processing strategy when out of memory occurs"""
        self.logger.warning("Using fallback processing strategy with CPU")
        
        # Move models to CPU
        original_device = self.device
        self.device = torch.device('cpu')
        self.presentation_model.to(self.device)
        self.edge_model.to(self.device)
        
        # Process with smaller batch or on CPU
        start_time = time.time()
        
        try:
            # 1. Quick presentation check
            presentation_result = self.presentation_model.predict(image.to(self.device))
            
            # 2. Check for critical presentation issues
            if self._has_critical_issue(presentation_result[0]):
                return {
                    'pass_fail': 'FAIL',
                    'reason': 'presentation_deficient',
                    'time_ms': (time.time() - start_time) * 1000,
                    'fallback_used': True
                }
            
            # 3. Main defect detection
            defect_result = self.edge_model.predict(image.to(self.device))
            
            # 4. Process tag data if provided
            if tag_data:
                defect_result = self._process_tag_data(defect_result, tag_data)
            
            # 5. Make pass/fail decision
            decision = self._make_decision(defect_result[0])
            
            # Calculate inference time
            inference_time_ms = (time.time() - start_time) * 1000
            
            # Log performance metrics
            self.perf_logger.log_inference_time("EdgeAlgorithmRouter_Fallback", inference_time_ms)
            
            # Log memory usage if CUDA is available
            if torch.cuda.is_available():
                allocated_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                reserved_mb = torch.cuda.memory_reserved() / (1024 * 1024)
                self.perf_logger.log_memory_usage("EdgeAlgorithmRouter_Fallback", allocated_mb, reserved_mb)
            
            result = {
                'pass_fail': decision,
                'detections': defect_result[0],
                'time_ms': inference_time_ms,
                'fallback_used': True
            }
            
            # Move models back to original device
            self.device = original_device
            self.presentation_model.to(self.device)
            self.edge_model.to(self.device)
            
            return result
            
        except Exception as e:
            # If even the fallback fails, return error
            self.logger.error(f"Fallback processing failed: {e}")
            
            # Move models back to original device
            self.device = original_device
            self.presentation_model.to(self.device)
            self.edge_model.to(self.device)
            
            return {
                'pass_fail': 'ERROR',
                'reason': f"Fallback processing failed: {str(e)}",
                'time_ms': (time.time() - start_time) * 1000,
                'fallback_used': True
            }


# ==================== NEW: Performance Monitoring ====================

class ConfidenceCalibrator:
    """Calibrate confidence scores for imbalanced classes"""

    def __init__(self, class_priors: Dict[int, float]):
        self.class_priors = class_priors

    def calibrate_predictions(self, predictions: List[Dict]) -> List[Dict]:
        """Adjust confidence based on class frequency"""
        calibrated = []
        for pred in predictions:
            scores = pred['scores']
            labels = pred['labels']

            # Apply isotonic regression or temperature scaling
            calibrated_scores = []
            for score, label in zip(scores, labels):
                prior = self.class_priors.get(label.item(), 0.1)
                # Adjust score based on prior
                calibrated_score = score * (1 - prior) + prior * 0.5
                calibrated_scores.append(calibrated_score)

            pred = pred.copy()
            pred['calibrated_scores'] = torch.tensor(calibrated_scores)
            calibrated.append(pred)

        return calibrated


class RareClassMonitor:
    """Special monitoring for rare defect types"""

    def __init__(self, rare_classes: List[int]):
        self.rare_classes = rare_classes
        self.metrics = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})

    def update(self, predictions: List[Dict], targets: List[Dict]):
        """Track performance on rare classes"""
        for pred, target in zip(predictions, targets):
            # Special attention to rare classes
            for rare_class in self.rare_classes:
                self._update_class_metrics(pred, target, rare_class)

    def _update_class_metrics(self, pred: Dict, target: Dict, class_id: int):
        """Update metrics for a specific class"""
        # Check if class is in predictions
        pred_has_class = class_id in pred['labels'].tolist() if 'labels' in pred else False

        # Check if class is in targets
        target_has_class = class_id in target['labels'].tolist() if 'labels' in target else False

        if pred_has_class and target_has_class:
            self.metrics[class_id]['tp'] += 1
        elif pred_has_class and not target_has_class:
            self.metrics[class_id]['fp'] += 1
        elif not pred_has_class and target_has_class:
            self.metrics[class_id]['fn'] += 1

    def get_rare_class_report(self) -> Dict[int, Dict[str, float]]:
        """Generate detailed report on rare class performance"""
        report = {}
        for class_id in self.rare_classes:
            metrics = self.metrics[class_id]
            precision = metrics['tp'] / (metrics['tp'] + metrics['fp'] + 1e-7)
            recall = metrics['tp'] / (metrics['tp'] + metrics['fn'] + 1e-7)
            report[class_id] = {
                'precision': precision,
                'recall': recall,
                'f1': 2 * precision * recall / (precision + recall + 1e-7),
                'support': metrics['tp'] + metrics['fn']
            }
        return report


class CheckpointManager:
    """Manages model checkpoint directory structure and paths"""
    
    def __init__(self, base_dir: Path = Path("checkpoints")):
        """
        Initialize the checkpoint manager.
        
        Args:
            base_dir: Base directory for checkpoints
        """
        self.base_dir = base_dir
        self.base_dir.mkdir(exist_ok=True)
        
    def get_checkpoint_path(self, model_name: str, epoch: int = None):
        """
        Get the path for a model checkpoint.
        
        Args:
            model_name: Name of the model
            epoch: Optional epoch number. If provided, returns path for that epoch.
                  If not provided, returns path for the best model.
                  
        Returns:
            Path object for the checkpoint
        """
        model_dir = self.base_dir / model_name
        model_dir.mkdir(exist_ok=True)
        
        if epoch:
            return model_dir / f"epoch_{epoch}.pt"
        return model_dir / "best.pt"


# ==================== Training Utilities ====================

def create_model_registry(num_classes: int, registry_path: Path = Path("model_zoo/registry.json")) -> Path:
    """Create default model registry with J&J optimizations"""
    registry = ModelRegistry(registry_path)

    # Update number of classes in all models
    for config in registry.registry.values():
        config.num_classes = num_classes

    registry.save_registry()
    return registry_path


def get_training_strategy(phase: str) -> Dict[str, Any]:
    """Get phase-specific training strategy"""
    strategies = {
        "phase1_binary": {
            "epochs": 20,
            "lr": 0.001,
            "objective": "Detect any defect (binary classification)",
            "modifications": {
                "num_classes": 2,
                "loss_weights": {"detection": 1.0},
                "use_balanced_sampler": False  # Binary is naturally balanced
            }
        },
        "phase2_location": {
            "epochs": 30,
            "lr": 0.0005,
            "objective": "Separate edge vs center defects",
            "modifications": {
                "num_classes": 3,
                "loss_weights": {"detection": 1.0, "localization": 0.5},
                "use_balanced_sampler": True,
                "min_samples_per_class": 20
            }
        },
        "phase3_full": {
            "epochs": 50,
            "lr": 0.0001,
            "objective": "Full defect taxonomy",
            "modifications": {
                "num_classes": 15,
                "loss_weights": {
                    "detection": 1.0,
                    "threshold": 0.5,
                    "critical": 2.0,
                    "precision": 0.3
                },
                "use_balanced_sampler": True,
                "min_samples_per_class": 50,
                "use_focal_loss": True,
                "focal_gamma": 2.0,
                "focal_alpha": 0.25
            }
        },
        "phase4_refinement": {
            "epochs": 30,
            "lr": 0.00005,
            "objective": "Measurement precision refinement",
            "modifications": {
                "freeze_backbone": True,
                "loss_weights": {
                    "threshold": 2.0,
                    "precision": 1.0,
                    "detection": 0.1
                },
                "use_balanced_sampler": True,
                "min_samples_per_class": 100,  # More samples for precision
                "augmentation_strategy": "conservative"  # Preserve measurements
            }
        }
    }

    return strategies.get(phase, strategies["phase3_full"])


# ==================== NEW: Multi-Stage Training ====================

def train_multi_stage(model_registry: ModelRegistry, dataset, base_path: Path):
    """Progressive training from easy to hard"""

    # Stage 1: Binary detection (defect/no defect)
    print("Stage 1: Training binary detector...")
    binary_config = ModelConfig(
        name="binary_detector",
        model_type="yolov8",
        backbone="yolov8m",
        num_classes=2,
        input_size=1024,
        confidence_threshold=0.5,
        nms_threshold=0.5,
        use_jj_loss=False
    )
    model_registry.register_model(binary_config)
    binary_model = model_registry.get_model("binary_detector")
    # Train binary model (implementation depends on your training loop)

    # Stage 2: Coarse categories (edge/center/presentation)
    print("Stage 2: Training coarse category detector...")
    coarse_config = ModelConfig(
        name="coarse_detector",
        model_type="yolov8",
        backbone="yolov8m",
        num_classes=4,  # background, edge, center, presentation
        input_size=1024,
        confidence_threshold=0.4,
        nms_threshold=0.5,
        use_jj_loss=True,
        checkpoint_path=str(base_path / "binary_detector_best.pt")
    )
    model_registry.register_model(coarse_config)
    coarse_model = model_registry.get_model("coarse_detector")
    # Train coarse model

    # Stage 3: Fine-grained with balanced sampling
    print("Stage 3: Training fine-grained detector with balanced sampling...")
    sampler = JJBalancedSampler(dataset, min_samples_per_class=50)
    # Use existing models from registry
    fine_model = model_registry.get_model("micro_defect_specialist")
    # Train with balanced sampler

    # Stage 4: Ensemble specialists
    print("Stage 4: Training specialist models...")
    specialists = ["edge_defect_specialist", "presentation_checker", "yolo_nano_bubbles"]
    for specialist_name in specialists:
        specialist = model_registry.get_model(specialist_name)
        # Train each specialist on relevant subset

    print("Multi-stage training complete!")