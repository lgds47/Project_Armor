"""
Test J&J compliance for evaluation logic
"""

import pytest
import numpy as np
from armor_pipeline.eval.jj_compliant_evaluator import JJCompliantEvaluator


class TestJJCompliance:
    """Test J&J evaluation logic compliance"""

    def test_pixel_threshold_scaling(self):
        """Test that pixel thresholds scale with resolution"""
        # At 2048x2048
        evaluator_2048 = JJCompliantEvaluator(
            original_resolution=(2048, 2048),
            processing_resolution=(2048, 2048)
        )
        assert evaluator_2048.scale_pixel_threshold(5.0) == 5.0

        # At 1024x1024
        evaluator_1024 = JJCompliantEvaluator(
            original_resolution=(2048, 2048),
            processing_resolution=(1024, 1024)
        )
        assert evaluator_1024.scale_pixel_threshold(5.0) == 2.5

    def test_edge_defect_evaluation(self):
        """Test edge defect pass/fail logic"""
        evaluator = JJCompliantEvaluator()

        # Create mock edge tear measurement
        from armor_pipeline.eval.jj_compliant_evaluator import JJDefectMeasurement

        # Should fail (5 pixels)
        measurement = JJDefectMeasurement(
            defect_type="edge-tear",
            pixel_measurement=5.0
        )
        passes, reason = evaluator.evaluate_defect(measurement)
        assert not passes

        # Should pass (4 pixels)
        measurement.pixel_measurement = 4.0
        passes, reason = evaluator.evaluate_defect(measurement)
        assert passes

    def test_percentage_parsing(self):
        """Test J&J percentage format parsing"""
        evaluator = JJCompliantEvaluator()

        # Test "<5%" → 3%
        attrs = evaluator.parse_cvat_attributes(
            '{"% Arc Length": "<5%"}'
        )
        assert attrs.get('arc_length_percent') == 3.0

        # Test "+5%" → 6%
        attrs = evaluator.parse_cvat_attributes(
            '{"% Arc Length": "+5%"}'
        )
        assert attrs.get('arc_length_percent') == 6.0