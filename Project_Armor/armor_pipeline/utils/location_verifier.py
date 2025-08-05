"""""
Location verification module for validating detection accuracy
Provides coordinate-based and visual confirmation of defect locations
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import pandas as pd
from dataclasses import dataclass
import json


@dataclass
class DetectionMatch:
    """Represents a match between prediction and ground truth"""
    pred_idx: int
    gt_idx: int
    iou: float
    center_distance: float  # Distance between centers in pixels
    pred_bbox: np.ndarray
    gt_bbox: np.ndarray
    pred_class: str
    gt_class: str
    is_correct_class: bool


class LocationVerifier:
    """Verify detection locations against ground truth with detailed metrics"""

    def __init__(
        self,
        iou_threshold: float = 0.5,
        center_distance_threshold: float = 50.0  # pixels
    ):
        self.iou_threshold = iou_threshold
        self.center_distance_threshold = center_distance_threshold
        self.matches = []
        self.unmatched_predictions = []
        self.unmatched_ground_truths = []

    def compute_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Compute Intersection over Union"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        # Intersection
        xi_min = max(x1_min, x2_min)
        yi_min = max(y1_min, y2_min)
        xi_max = min(x1_max, x2_max)
        yi_max = min(y1_max, y2_max)

        if xi_max < xi_min or yi_max < yi_min:
            return 0.0

        intersection = (xi_max - xi_min) * (yi_max - yi_min)

        # Union
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def compute_center_distance(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Compute Euclidean distance between box centers"""
        center1 = np.array([(box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2])
        center2 = np.array([(box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2])
        return np.linalg.norm(center1 - center2)

    def match_detections(
        self,
        predictions: List[Dict],
        ground_truths: List[Dict],
        class_names: List[str]
    ) -> List[DetectionMatch]:
        """Match predictions to ground truth based on IoU and center distance"""
        matches = []
        matched_gt_indices = set()

        # Sort predictions by confidence (descending)
        sorted_pred_indices = sorted(
            range(len(predictions)),
            key=lambda i: predictions[i]['confidence'],
            reverse=True
        )

        for pred_idx in sorted_pred_indices:
            pred = predictions[pred_idx]
            pred_bbox = pred['bbox']
            pred_class = class_names[pred['label']]

            best_match = None
            best_iou = self.iou_threshold
            best_distance = float('inf')

            # Find best matching ground truth
            for gt_idx, gt in enumerate(ground_truths):
                if gt_idx in matched_gt_indices:
                    continue

                gt_bbox = gt['bbox']
                gt_class = class_names[gt['label']]

                # Compute metrics
                iou = self.compute_iou(pred_bbox, gt_bbox)
                center_dist = self.compute_center_distance(pred_bbox, gt_bbox)

                # Check if this is a better match
                if iou >= self.iou_threshold:
                    if iou > best_iou or (iou == best_iou and center_dist < best_distance):
                        best_match = DetectionMatch(
                            pred_idx=pred_idx,
                            gt_idx=gt_idx,
                            iou=iou,
                            center_distance=center_dist,
                            pred_bbox=pred_bbox,
                            gt_bbox=gt_bbox,
                            pred_class=pred_class,
                            gt_class=gt_class,
                            is_correct_class=(pred_class == gt_class)
                        )
                        best_iou = iou
                        best_distance = center_dist

            if best_match:
                matches.append(best_match)
                matched_gt_indices.add(best_match.gt_idx)

        # Record unmatched detections
        self.matches = matches
        self.unmatched_predictions = [
            i for i in range(len(predictions))
            if i not in [m.pred_idx for m in matches]
        ]
        self.unmatched_ground_truths = [
            i for i in range(len(ground_truths))
            if i not in matched_gt_indices
        ]

        return matches

    def generate_location_report(self, output_path: Path) -> pd.DataFrame:
        """Generate detailed location accuracy report"""
        report_data = []

        for match in self.matches:
            # Calculate location metrics
            pred_center = [
                (match.pred_bbox[0] + match.pred_bbox[2]) / 2,
                (match.pred_bbox[1] + match.pred_bbox[3]) / 2
            ]
            gt_center = [
                (match.gt_bbox[0] + match.gt_bbox[2]) / 2,
                (match.gt_bbox[1] + match.gt_bbox[3]) / 2
            ]

            # Size comparison
            pred_area = (match.pred_bbox[2] - match.pred_bbox[0]) * (match.pred_bbox[3] - match.pred_bbox[1])
            gt_area = (match.gt_bbox[2] - match.gt_bbox[0]) * (match.gt_bbox[3] - match.gt_bbox[1])
            area_ratio = pred_area / gt_area if gt_area > 0 else 0

            report_data.append({
                'match_id': len(report_data),
                'pred_class': match.pred_class,
                'gt_class': match.gt_class,
                'class_correct': match.is_correct_class,
                'iou': match.iou,
                'center_distance_pixels': match.center_distance,
                'pred_center_x': pred_center[0],
                'pred_center_y': pred_center[1],
                'gt_center_x': gt_center[0],
                'gt_center_y': gt_center[1],
                'center_offset_x': pred_center[0] - gt_center[0],
                'center_offset_y': pred_center[1] - gt_center[1],
                'area_ratio': area_ratio,
                'pred_bbox': match.pred_bbox.tolist(),
                'gt_bbox': match.gt_bbox.tolist()
            })

        df = pd.DataFrame(report_data)
        df.to_csv(output_path, index=False)

        # Print summary statistics
        print("\n=== LOCATION ACCURACY SUMMARY ===")
        print(f"Total matches: {len(self.matches)}")
        print(f"Unmatched predictions: {len(self.unmatched_predictions)}")
        print(f"Unmatched ground truths: {len(self.unmatched_ground_truths)}")

        if len(df) > 0:
            print(f"\nAverage IoU: {df['iou'].mean():.3f}")
            print(f"Average center distance: {df['center_distance_pixels'].mean():.1f} pixels")
            print(f"Class accuracy: {df['class_correct'].mean():.1%}")
            print(f"\nCenter offset statistics:")
            print(f"  X offset: {df['center_offset_x'].mean():.1f} ± {df['center_offset_x'].std():.1f} pixels")
            print(f"  Y offset: {df['center_offset_y'].mean():.1f} ± {df['center_offset_y'].std():.1f} pixels")

        return df

    def visualize_location_comparison(
        self,
        image: np.ndarray,
        predictions: List[Dict],
        ground_truths: List[Dict],
        class_names: List[str],
        output_path: Path,
        show_metrics: bool = True
    ):
        """Create visual comparison of prediction vs ground truth locations"""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 7))

        # Ground Truth
        ax1.imshow(image)
        ax1.set_title("Ground Truth", fontsize=14, weight='bold')
        ax1.axis('off')

        for gt in ground_truths:
            bbox = gt['bbox']
            rect = FancyBboxPatch(
                (bbox[0], bbox[1]),
                bbox[2] - bbox[0],
                bbox[3] - bbox[1],
                boxstyle="round,pad=0.1",
                facecolor='none',
                edgecolor='green',
                linewidth=2
            )
            ax1.add_patch(rect)

            # Label
            label = class_names[gt['label']]
            ax1.text(
                bbox[0], bbox[1] - 5,
                label,
                color='white',
                backgroundcolor='green',
                fontsize=8,
                weight='bold'
            )

        # Predictions
        ax2.imshow(image)
        ax2.set_title("Predictions", fontsize=14, weight='bold')
        ax2.axis('off')

        for pred in predictions:
            bbox = pred['bbox']
            conf = pred['confidence']
            rect = FancyBboxPatch(
                (bbox[0], bbox[1]),
                bbox[2] - bbox[0],
                bbox[3] - bbox[1],
                boxstyle="round,pad=0.1",
                facecolor='none',
                edgecolor='red',
                linewidth=2
            )
            ax2.add_patch(rect)

            # Label with confidence
            label = f"{class_names[pred['label']]}: {conf:.2f}"
            ax2.text(
                bbox[0], bbox[1] - 5,
                label,
                color='white',
                backgroundcolor='red',
                fontsize=8,
                weight='bold'
            )

        # Overlay comparison
        ax3.imshow(image)
        ax3.set_title("Overlay Comparison", fontsize=14, weight='bold')
        ax3.axis('off')

        # Draw matches
        for match in self.matches:
            # Ground truth in green
            gt_rect = FancyBboxPatch(
                (match.gt_bbox[0], match.gt_bbox[1]),
                match.gt_bbox[2] - match.gt_bbox[0],
                match.gt_bbox[3] - match.gt_bbox[1],
                boxstyle="round,pad=0.1",
                facecolor='none',
                edgecolor='green',
                linewidth=2,
                linestyle='--',
                alpha=0.7
            )
            ax3.add_patch(gt_rect)

            # Prediction in red/yellow based on class match
            color = 'yellow' if match.is_correct_class else 'red'
            pred_rect = FancyBboxPatch(
                (match.pred_bbox[0], match.pred_bbox[1]),
                match.pred_bbox[2] - match.pred_bbox[0],
                match.pred_bbox[3] - match.pred_bbox[1],
                boxstyle="round,pad=0.1",
                facecolor='none',
                edgecolor=color,
                linewidth=2,
                alpha=0.7
            )
            ax3.add_patch(pred_rect)

            # Draw connection line between centers
            pred_center = [
                (match.pred_bbox[0] + match.pred_bbox[2]) / 2,
                (match.pred_bbox[1] + match.pred_bbox[3]) / 2
            ]
            gt_center = [
                (match.gt_bbox[0] + match.gt_bbox[2]) / 2,
                (match.gt_bbox[1] + match.gt_bbox[3]) / 2
            ]

            ax3.plot(
                [gt_center[0], pred_center[0]],
                [gt_center[1], pred_center[1]],
                'b-', linewidth=1, alpha=0.5
            )

            if show_metrics:
                # Show IoU and distance
                mid_x = (gt_center[0] + pred_center[0]) / 2
                mid_y = (gt_center[1] + pred_center[1]) / 2
                ax3.text(
                    mid_x, mid_y,
                    f"IoU:{match.iou:.2f}\nD:{match.center_distance:.0f}px",
                    fontsize=6,
                    ha='center',
                    va='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7)
                )

        # Legend
        legend_elements = [
            patches.Patch(color='green', label='Ground Truth'),
            patches.Patch(color='yellow', label='Correct Class Match'),
            patches.Patch(color='red', label='Wrong Class Match'),
            patches.Line2D([0], [0], color='blue', label='Center Connection')
        ]
        ax3.legend(
            handles=legend_elements,
            loc='upper right',
            fontsize=8,
            framealpha=0.8
        )

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Location comparison saved to: {output_path}")

    def export_coordinate_comparison(self, output_path: Path):
        """Export detailed coordinate comparison for analysis"""
        comparison_data = {
            'matches': [],
            'false_positives': [],
            'false_negatives': [],
            'summary': {
                'total_matches': len(self.matches),
                'avg_iou': np.mean([m.iou for m in self.matches]) if self.matches else 0,
                'avg_center_distance': np.mean([m.center_distance for m in self.matches]) if self.matches else 0,
                'class_accuracy': np.mean([m.is_correct_class for m in self.matches]) if self.matches else 0
            }
        }

        for match in self.matches:
            comparison_data['matches'].append({
                'pred_bbox': match.pred_bbox.tolist(),
                'gt_bbox': match.gt_bbox.tolist(),
                'pred_class': match.pred_class,
                'gt_class': match.gt_class,
                'iou': float(match.iou),
                'center_distance': float(match.center_distance),
                'class_match': match.is_correct_class
            })

        with open(output_path, 'w') as f:
            json.dump(comparison_data, f, indent=2)

        print(f"Coordinate comparison exported to: {output_path}")