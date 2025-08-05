"""
Unit tests for evaluation metrics
"""

import pytest
import torch
import numpy as np
from armor_pipeline.eval.evaluator import DefectEvaluator


class TestEvaluator:
    """Test evaluation metrics computation"""

    @pytest.fixture
    def sample_predictions(self):
        """Create sample predictions and ground truth"""
        # Predictions for 2 images
        predictions = [
            {
                'boxes': torch.tensor([[100, 100, 200, 200], [300, 300, 400, 400]]),
                'scores': torch.tensor([0.9, 0.8]),
                'labels': torch.tensor([1, 2])
            },
            {
                'boxes': torch.tensor([[150, 150, 250, 250]]),
                'scores': torch.tensor([0.7]),
                'labels': torch.tensor([1])
            }
        ]

        # Ground truth
        targets = [
            {
                'boxes': torch.tensor([[95, 95, 205, 205], [500, 500, 600, 600]]),
                'labels': torch.tensor([1, 2])
            },
            {
                'boxes': torch.tensor([[140, 140, 260, 260], [400, 400, 500, 500]]),
                'labels': torch.tensor([1, 3])
            }
        ]

        image_paths = ["image1.bmp", "image2.bmp"]

        return predictions, targets, image_paths

    def test_iou_computation(self):
        """Test IoU calculation"""
        evaluator = DefectEvaluator(["background", "defect1", "defect2"])

        box1 = np.array([0, 0, 100, 100])
        box2 = np.array([50, 50, 150, 150])

        iou = evaluator._compute_iou(box1, box2)
        expected_iou = (50 * 50) / (100 * 100 + 100 * 100 - 50 * 50)
        assert abs(iou - expected_iou) < 0.001

    def test_ap_calculation(self, sample_predictions):
        """Test Average Precision calculation"""
        predictions, targets, image_paths = sample_predictions

        evaluator = DefectEvaluator(
            ["background", "defect1", "defect2", "defect3"],
            iou_threshold=0.5
        )

        evaluator.update(predictions, targets, image_paths)

        ap_per_class = evaluator.compute_ap_per_class()
        mAP = evaluator.compute_map()

        # Check that we have AP for each class
        assert "defect1" in ap_per_class
        assert "defect2" in ap_per_class

        # mAP should be between 0 and 1
        assert 0 <= mAP <= 1

    def test_confusion_matrix(self, sample_predictions):
        """Test confusion matrix generation"""
        predictions, targets, image_paths = sample_predictions

        evaluator = DefectEvaluator(
            ["background", "defect1", "defect2", "defect3"],
            iou_threshold=0.5
        )

        evaluator.update(predictions, targets, image_paths)

        cm, cm_df = evaluator.compute_confusion_matrix()

        # Check shape
        assert cm.shape == (4, 4)  # 4 classes including background
        assert cm_df.shape == (4, 4)

        # Check that diagonal has some true positives
        assert np.trace(cm) > 0

    def test_pass_fail_logic(self, sample_predictions):
        """Test Pass/Fail decision logic"""
        predictions, targets, image_paths = sample_predictions

        evaluator = DefectEvaluator(
            ["background", "defect1", "defect2", "defect3"],
            iou_threshold=0.5
        )

        # Set thresholds
        evaluator.pass_fail_thresholds = {
            'default': {'confidence': 0.5, 'severity': 0.01},
            'defect1': {'confidence': 0.8, 'severity': 0.005}
        }

        evaluator.update(predictions, targets, image_paths)

        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            df = evaluator.generate_pass_fail_report(Path(tmpdir))

            # Check DataFrame structure
            assert 'image_id' in df.columns
            assert 'PASS_FAIL' in df.columns
            assert 'severity' in df.columns

            # Check that we have results for both images
            assert len(df) == 2

            # Check Pass/Fail values
            assert all(df['PASS_FAIL'].isin(['PASS', 'FAIL']))