"""
Evaluation Module for J&J Contact Lens Defect Detection
Computes AP metrics, confusion matrix, and Pass/Fail decisions
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import yaml
from collections import defaultdict
from sklearn.metrics import confusion_matrix
import cv2

from armor_pipeline.utils.bbox_utils import convert_bbox_format


class DefectEvaluator:
    """Comprehensive evaluator for defect detection"""

    def __init__(
            self,
            class_names: List[str],
            iou_threshold: float = 0.5,
            pass_fail_config_path: Optional[Path] = None
    ):
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.iou_threshold = iou_threshold

        # Load pass/fail thresholds
        if pass_fail_config_path:
            self.pass_fail_thresholds = self._load_pass_fail_config(pass_fail_config_path)
        else:
            # Default thresholds
            self.pass_fail_thresholds = {
                'default': {'confidence': 0.5, 'severity': 0.5}
            }

        # Initialize metrics storage
        self.reset()
        
        # Buffer to optionally store per-iteration metrics for live dashboards
        self.metrics_buffer: List[Dict[str, Any]] = []

    def _load_pass_fail_config(self, config_path: Path) -> Dict[str, Dict[str, float]]:
        """Load pass/fail configuration from YAML"""
        with open(config_path) as f:
            config = yaml.safe_load(f)
        return config['pass_fail_thresholds']

    def reset(self):
        """Reset evaluation metrics"""
        self.predictions = []
        self.ground_truths = []
        self.image_paths = []
        self.tp_fp_scores = defaultdict(list)  # For AP calculation
        self.num_gt = defaultdict(int)

    def update(
            self,
            predictions: List[Dict],
            targets: List[Dict],
            image_paths: List[str]
    ):
        """Update evaluator with batch of predictions"""
        for pred, target, img_path in zip(predictions, targets, image_paths):
            self.predictions.append(pred)
            self.ground_truths.append(target)
            self.image_paths.append(img_path)

            # Process for AP calculation
            self._process_image_for_ap(pred, target)

    def _process_image_for_ap(self, pred: Dict, target: Dict):
        """Process single image predictions for AP calculation"""
        pred_boxes = pred['boxes'].cpu().numpy()
        pred_scores = pred['scores'].cpu().numpy()
        pred_labels = pred['labels'].cpu().numpy()

        gt_boxes = target['boxes'].cpu().numpy()
        gt_labels = target['labels'].cpu().numpy()

        # Convert to xyxy format for consistent processing
        # This automatically detects the format and converts if needed
        if gt_boxes.shape[1] == 4 and len(gt_boxes) > 0:
            gt_boxes = convert_bbox_format(gt_boxes, source_format=None, target_format='xyxy')

        # Count ground truth per class
        for label in gt_labels:
            self.num_gt[int(label)] += 1

        # Match predictions to ground truth
        matched_gt = set()

        for i, (box, score, label) in enumerate(zip(pred_boxes, pred_scores, pred_labels)):
            best_iou = 0
            best_gt_idx = -1

            # Find best matching ground truth
            for j, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
                if gt_label != label or j in matched_gt:
                    continue

                iou = self._compute_iou(box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j

            # Record TP/FP
            if best_iou >= self.iou_threshold:
                self.tp_fp_scores[int(label)].append((score, 1))  # TP
                matched_gt.add(best_gt_idx)
            else:
                self.tp_fp_scores[int(label)].append((score, 0))  # FP

    def _compute_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Compute IoU between two boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        if x2 < x1 or y2 < y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def compute_ap_per_class(self) -> Dict[str, float]:
        """Compute Average Precision for each class"""
        ap_per_class = {}

        for class_idx in range(self.num_classes):
            if class_idx == 0:  # Skip background
                continue

            # Get all predictions for this class
            scores_labels = self.tp_fp_scores.get(class_idx, [])
            if not scores_labels:
                ap_per_class[self.class_names[class_idx]] = 0.0
                continue

            # Sort by score (descending)
            scores_labels = sorted(scores_labels, key=lambda x: x[0], reverse=True)

            # Compute precision-recall curve
            tp = 0
            fp = 0
            precisions = []
            recalls = []

            n_gt = self.num_gt.get(class_idx, 0)
            if n_gt == 0:
                ap_per_class[self.class_names[class_idx]] = 0.0
                continue

            for score, is_tp in scores_labels:
                if is_tp:
                    tp += 1
                else:
                    fp += 1

                precision = tp / (tp + fp)
                recall = tp / n_gt

                precisions.append(precision)
                recalls.append(recall)

            # Compute AP using all-point interpolation
            ap = self._compute_ap(precisions, recalls)
            ap_per_class[self.class_names[class_idx]] = ap

        return ap_per_class

    def _compute_ap(self, precisions: List[float], recalls: List[float]) -> float:
        """Compute Average Precision using all-point interpolation"""
        # Add sentinel values
        precisions = [0] + precisions + [0]
        recalls = [0] + recalls + [1]

        # Make precision monotonically decreasing
        for i in range(len(precisions) - 2, -1, -1):
            precisions[i] = max(precisions[i], precisions[i + 1])

        # Compute area under curve
        ap = 0
        for i in range(1, len(recalls)):
            ap += (recalls[i] - recalls[i - 1]) * precisions[i]

        return ap

    def compute_map(self) -> float:
        """Compute mean Average Precision across all classes"""
        ap_per_class = self.compute_ap_per_class()
        if not ap_per_class:
            return 0.0
        return np.mean(list(ap_per_class.values()))
        
    def compute_jj_metrics(self) -> Dict[str, float]:
        """Compute J&J-specific metrics beyond standard mAP"""
        metrics = {}
        
        # 1. Threshold accuracy: % of defects correctly classified as pass/fail at 5px
        threshold_correct = 0
        threshold_total = 0
        
        for pred, gt in zip(self.predictions, self.ground_truths):
            pred_sizes = torch.max(pred['boxes'][:, 2:] - pred['boxes'][:, :2], dim=1)[0]
            gt_sizes = torch.max(gt['boxes'][:, 2:] - gt['boxes'][:, :2], dim=1)[0]
            
            pred_fail = (pred_sizes > 5.0)
            gt_fail = (gt_sizes > 5.0)
            threshold_correct += (pred_fail == gt_fail).sum().item()
            threshold_total += len(gt_fail)
        
        metrics['threshold_accuracy'] = threshold_correct / max(threshold_total, 1)
        
        # 2. Critical defect recall (must be 100% for always-fail defects)
        critical_labels = [10, 11, 12, 13]  # folded, inverted, missing, multiple
        
        # Count critical defects in ground truth
        critical_gt_total = 0
        critical_detected = 0
        
        for pred, gt in zip(self.predictions, self.ground_truths):
            gt_labels = gt['labels'].cpu()
            gt_critical_mask = torch.tensor([label.item() in critical_labels for label in gt_labels], dtype=torch.bool)
            critical_gt_total += gt_critical_mask.sum().item()
            
            if gt_critical_mask.sum().item() > 0:
                # For each critical GT defect, check if it was detected
                gt_boxes_critical = gt['boxes'][gt_critical_mask]
                
                for gt_box in gt_boxes_critical:
                    # Find best matching prediction
                    best_iou = 0
                    best_pred_idx = -1
                    
                    for i, pred_box in enumerate(pred['boxes']):
                        iou = self._compute_iou(gt_box.cpu().numpy(), pred_box.cpu().numpy())
                        if iou > best_iou:
                            best_iou = iou
                            best_pred_idx = i
                    
                    # If IoU > threshold and label is correct, count as detected
                    if best_pred_idx >= 0 and best_iou >= self.iou_threshold:
                        pred_label = pred['labels'][best_pred_idx].item()
                        if pred_label in critical_labels:
                            critical_detected += 1
        
        metrics['critical_recall'] = critical_detected / max(critical_gt_total, 1)
        
        return metrics

    def compute_confusion_matrix(self) -> Tuple[np.ndarray, pd.DataFrame]:
        """Compute confusion matrix for detection"""
        all_true_labels = []
        all_pred_labels = []

        for pred, target in zip(self.predictions, self.ground_truths):
            # For each ground truth, find best matching prediction
            gt_labels = target['labels'].cpu().numpy()
            gt_boxes = target['boxes'].cpu().numpy()

            pred_labels = pred['labels'].cpu().numpy()
            pred_boxes = pred['boxes'].cpu().numpy()
            pred_scores = pred['scores'].cpu().numpy()

            # Convert COCO to xyxy if needed
            if gt_boxes.shape[1] == 4 and len(gt_boxes) > 0:
                if gt_boxes[0, 2] < gt_boxes[0, 0] * 2:
                    gt_boxes[:, 2] = gt_boxes[:, 0] + gt_boxes[:, 2]
                    gt_boxes[:, 3] = gt_boxes[:, 1] + gt_boxes[:, 3]

            # Match each GT to best prediction
            for gt_idx, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
                best_pred_label = 0  # Background/missed
                best_iou = self.iou_threshold

                for pred_idx, (pred_box, pred_label, pred_score) in enumerate(
                        zip(pred_boxes, pred_labels, pred_scores)
                ):
                    iou = self._compute_iou(gt_box, pred_box)
                    if iou >= best_iou:
                        best_iou = iou
                        best_pred_label = pred_label

                all_true_labels.append(int(gt_label))
                all_pred_labels.append(int(best_pred_label))

        # Create confusion matrix
        cm = confusion_matrix(
            all_true_labels,
            all_pred_labels,
            labels=list(range(self.num_classes))
        )

        # Create DataFrame for better visualization
        cm_df = pd.DataFrame(
            cm,
            index=self.class_names,
            columns=self.class_names
        )

        return cm, cm_df

    def generate_pass_fail_report(self, output_dir: Path) -> pd.DataFrame:
        """Generate Pass/Fail report for each lens"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        pass_fail_data = []

        for img_path, pred in zip(self.image_paths, self.predictions):
            image_id = Path(img_path).stem

            # Check if any defect exceeds threshold
            lens_pass = True
            worst_defect = None
            worst_severity = 0

            for box, score, label in zip(
                    pred['boxes'].cpu().numpy(),
                    pred['scores'].cpu().numpy(),
                    pred['labels'].cpu().numpy()
            ):
                class_name = self.class_names[int(label)]

                # Get thresholds for this class
                thresholds = self.pass_fail_thresholds.get(
                    class_name,
                    self.pass_fail_thresholds['default']
                )

                # Calculate severity (using confidence * area as proxy)
                area = (box[2] - box[0]) * (box[3] - box[1])
                severity = score * (area / (2048 * 2048))  # Normalize by image size

                # Check if fails
                if score >= thresholds['confidence'] and severity >= thresholds['severity']:
                    lens_pass = False
                    if severity > worst_severity:
                        worst_severity = severity
                        worst_defect = class_name

            pass_fail_data.append({
                'image_id': image_id,
                'predicted_label': worst_defect or 'clean',
                'confidence': float(pred['scores'].max()) if len(pred['scores']) > 0 else 0.0,
                'severity': worst_severity,
                'PASS_FAIL': 'PASS' if lens_pass else 'FAIL'
            })

        # Create DataFrame
        df = pd.DataFrame(pass_fail_data)

        # Save to CSV
        output_path = output_dir / f"pass_fail_{timestamp}.csv"
        df.to_csv(output_path, index=False)

        return df

    def plot_confusion_matrix(self, output_dir: Path):
        """Plot and save confusion matrix"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        cm, cm_df = self.compute_confusion_matrix()

        # Create figure
        plt.figure(figsize=(12, 10))

        # Plot heatmap
        sns.heatmap(
            cm_df,
            annot=True,
            fmt='d',
            cmap='Blues',
            cbar_kws={'label': 'Count'}
        )

        plt.title('Confusion Matrix - Contact Lens Defect Detection')
        plt.xlabel('Predicted Class')
        plt.ylabel('True Class')
        plt.tight_layout()

        # Save figure
        fig_path = output_dir / f"confusion_matrix_{timestamp}.png"
        plt.savefig(fig_path, dpi=150)
        plt.close()

        # Also save numeric CSV
        csv_path = output_dir / f"confusion_matrix_{timestamp}.csv"
        cm_df.to_csv(csv_path)

        return cm_df

    def export_defect_instances(self, output_dir: Path) -> Optional[Path]:
        """Export predicted defect instances with size stats for distribution analysis.
        Returns path to CSV or None if no predictions.
        """
        output_dir = Path(output_dir)
        rows = []
        for img_path, pred in zip(self.image_paths, self.predictions):
            image_id = Path(img_path).stem
            boxes = pred.get('boxes')
            scores = pred.get('scores')
            labels = pred.get('labels')
            if boxes is None or scores is None or labels is None:
                continue
            # Ensure tensors on CPU
            boxes_np = boxes.detach().cpu().numpy() if hasattr(boxes, 'detach') else np.array(boxes)
            scores_np = scores.detach().cpu().numpy() if hasattr(scores, 'detach') else np.array(scores)
            labels_np = labels.detach().cpu().numpy() if hasattr(labels, 'detach') else np.array(labels)
            for box, score, label in zip(boxes_np, scores_np, labels_np):
                x1, y1, x2, y2 = box.tolist()
                w = max(0.0, float(x2 - x1))
                h = max(0.0, float(y2 - y1))
                area = w * h
                class_idx = int(label)
                class_name = self.class_names[class_idx] if 0 <= class_idx < len(self.class_names) else str(class_idx)
                rows.append({
                    'image_id': image_id,
                    'class_idx': class_idx,
                    'class_name': class_name,
                    'score': float(score),
                    'x1': float(x1), 'y1': float(y1), 'x2': float(x2), 'y2': float(y2),
                    'width': w, 'height': h, 'area': area,
                })
        if not rows:
            return None
        df = pd.DataFrame(rows)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_path = output_dir / f'defect_instances_{timestamp}.csv'
        df.to_csv(out_path, index=False)
        return out_path

    def generate_full_report(self, output_dir: Path) -> Dict[str, Any]:
        """Generate comprehensive evaluation report"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        # Compute all metrics
        ap_per_class = self.compute_ap_per_class()
        mAP = self.compute_map()
        jj_metrics = self.compute_jj_metrics()

        # Generate visualizations and reports
        cm_df = self.plot_confusion_matrix(output_dir)
        pass_fail_df = self.generate_pass_fail_report(output_dir)
        defects_csv_path = self.export_defect_instances(output_dir)

        # Create summary report
        report = {
            'mAP': mAP,
            'AP_per_class': ap_per_class,
            'jj_metrics': jj_metrics,
            'total_images': len(self.image_paths),
            'pass_rate': (pass_fail_df['PASS_FAIL'] == 'PASS').mean(),
            'fail_rate': (pass_fail_df['PASS_FAIL'] == 'FAIL').mean(),
            'timestamp': datetime.now().isoformat(),
            'artifacts': {
                'confusion_matrix_csv': str(output_dir / f"confusion_matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"),
                'pass_fail_csv': str(output_dir / f"pass_fail_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"),
                'defect_instances_csv': str(defects_csv_path) if defects_csv_path else None,
            }
        }

        # Save summary
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(output_dir / f"evaluation_summary_{timestamp}.yaml", 'w') as f:
            yaml.dump(report, f)

        # Print summary
        print("\n" + "=" * 50)
        print("EVALUATION SUMMARY")
        print("=" * 50)
        print(f"Mean Average Precision (mAP): {mAP:.4f}")
        print(f"Target mAP: ≥ 0.90")
        print(f"Status: {'✓ PASS' if mAP >= 0.90 else '✗ FAIL'}")
        
        # Print J&J-specific metrics
        print("\nJ&J-SPECIFIC METRICS")
        print(f"Threshold Accuracy (5px): {jj_metrics['threshold_accuracy']:.4f}")
        print(f"Target Threshold Accuracy: ≥ 0.95")
        print(f"Status: {'✓ PASS' if jj_metrics['threshold_accuracy'] >= 0.95 else '✗ FAIL'}")
        
        print(f"Critical Defect Recall: {jj_metrics['critical_recall']:.4f}")
        print(f"Target Critical Recall: 1.00")
        print(f"Status: {'✓ PASS' if jj_metrics['critical_recall'] >= 0.99 else '✗ FAIL'}")
        
        print(f"\nPer-class AP:")
        for class_name, ap in ap_per_class.items():
            print(f"  {class_name}: {ap:.4f}")
        print(f"\nLens Pass Rate: {report['pass_rate']:.2%}")
        print(f"Total Images Evaluated: {report['total_images']}")
        print("=" * 50)

        return report