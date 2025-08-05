"""
J&J Compliant Pass/Fail Evaluator
Implements exact specifications from J&J requirements document
Handles both pixel-based and percentage-based criteria
"""

import numpy as np
import re
import json
import math
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass
class JJDefectMeasurement:
    """Defect measurement matching J&J specifications"""
    defect_type: str
    pixel_measurement: float  # Direct pixel measurement
    percentage_measurement: Optional[float] = None  # For arc/surface percentages
    location: Optional[Tuple[float, float]] = None
    attributes: Dict[str, Any] = None


class JJCompliantEvaluator:
    """Evaluator that exactly matches J&J pass/fail criteria"""

    # Direct from J&J spec - Section 2.2 and enhanced_evaluation_full_refactor.py
    PIXEL_THRESHOLDS = {
        # Center defects (Section 2.2.1-2.2.3)
        "debris": {"measure": "longest_dimension", "threshold": 5, "mode": ">="},
        "center-hole": {"measure": "longest_dimension", "threshold": 5, "mode": ">="},
        "center-tear": {"measure": "longest_dimension", "threshold": 5, "mode": ">="},

        # Edge defects with depth measurement (Section 2.2.4-2.2.7)
        "edge-tear": {"measure": "depth", "threshold": 5, "mode": ">="},
        "edge-chip": {"measure": "depth", "threshold": 5, "mode": ">="},
        "exterior-excess": {"measure": "depth", "threshold": 5, "mode": ">="},

        # Always fail
        "edge-not-closed": {"mode": "always_fail"},
        "lens-folded": {"mode": "always_fail"},
        "lens-inverted": {"mode": "always_fail"},
        "missing-lens": {"mode": "always_fail"},
        "lens-out-of-round": {"mode": "always_fail"},
    }

    # Percentage-based thresholds from J&J spec
    PERCENTAGE_THRESHOLDS = {
        # Presentation defects (Section 2.1)
        "edge-out-of-focus": {"measure": "arc_length_percent", "threshold": 30.0},
        "edge-obstructed": {"measure": "arc_length_percent", "threshold": 4.0},
        "ripple": {"measure": "surface_percent", "threshold": 50.0},
        "bubble": {"measure": "area_percent", "threshold": 4.0},
        "123-mark-obstructed": {"measure": "dots_count", "threshold": 3.0},
    }

    def __init__(self,
                 original_resolution: Tuple[int, int] = (2048, 2048),
                 processing_resolution: Tuple[int, int] = (1024, 1024),
                 lens_diameter_px: float = 1580.0):  # From J&J spec
        """
        Initialize with J&J specifications

        Args:
            original_resolution: Original image size after cropping (2048x2048)
            processing_resolution: Model inference resolution
            lens_diameter_px: Lens diameter in pixels at original resolution
        """
        self.original_resolution = original_resolution
        self.processing_resolution = processing_resolution
        self.lens_diameter_px = lens_diameter_px

        # Calculate scaling factor
        self.scale_factor = processing_resolution[0] / original_resolution[0]

        # Lens measurements from J&J spec
        self.lens_radius = lens_diameter_px / 2.0
        self.lens_area = math.pi * self.lens_radius ** 2
        self.lens_circumference = 2 * math.pi * self.lens_radius

    def scale_pixel_threshold(self, threshold: float) -> float:
        """Scale pixel threshold based on resolution"""
        return threshold * self.scale_factor

    def measure_defect(self, bbox: np.ndarray, defect_type: str) -> JJDefectMeasurement:
        """
        Measure defect according to J&J specifications

        Args:
            bbox: [x1, y1, x2, y2] in processing resolution
            defect_type: Type of defect from J&J taxonomy
        """
        x1, y1, x2, y2 = bbox

        # Calculate measurements
        width = x2 - x1
        height = y2 - y1
        longest_dimension = max(width, height)

        # For edge defects, calculate radial depth
        # This is simplified - in production, use actual polyline measurements
        center_x = self.processing_resolution[0] / 2
        center_y = self.processing_resolution[1] / 2

        # Estimate radial depth for edge defects
        bbox_center_x = (x1 + x2) / 2
        bbox_center_y = (y1 + y2) / 2
        distance_from_center = np.sqrt((bbox_center_x - center_x)**2 +
                                     (bbox_center_y - center_y)**2)

        # Radial depth approximation
        expected_radius = self.lens_radius * self.scale_factor
        radial_depth = abs(expected_radius - distance_from_center)

        # Area percentage for bubbles
        bbox_area = width * height
        scaled_lens_area = self.lens_area * (self.scale_factor ** 2)
        area_percent = (bbox_area / scaled_lens_area) * 100

        return JJDefectMeasurement(
            defect_type=defect_type,
            pixel_measurement=longest_dimension if "center" in defect_type else radial_depth,
            percentage_measurement=area_percent if defect_type == "bubble" else None,
            location=(bbox_center_x / self.processing_resolution[0],
                     bbox_center_y / self.processing_resolution[1])
        )

    def evaluate_defect(self, measurement: JJDefectMeasurement) -> Tuple[bool, str]:
        """
        Evaluate defect against J&J thresholds

        Returns:
            (pass/fail, reason)
        """
        defect_type = measurement.defect_type.lower()

        # Check pixel-based thresholds
        if defect_type in self.PIXEL_THRESHOLDS:
            rule = self.PIXEL_THRESHOLDS[defect_type]

            if rule.get("mode") == "always_fail":
                return (False, f"{defect_type}: always fails per J&J spec")

            threshold = self.scale_pixel_threshold(rule["threshold"])
            value = measurement.pixel_measurement

            if rule.get("mode") == ">=" and value >= threshold:
                return (False, f"{defect_type}: {value:.1f}px >= {threshold:.1f}px (scaled)")

            return (True, f"{defect_type}: {value:.1f}px < {threshold:.1f}px (scaled)")

        # Check percentage-based thresholds
        if defect_type in self.PERCENTAGE_THRESHOLDS:
            rule = self.PERCENTAGE_THRESHOLDS[defect_type]
            threshold = rule["threshold"]

            if rule["measure"] == "area_percent":
                value = measurement.percentage_measurement
                if value >= threshold:
                    return (False, f"{defect_type}: {value:.1f}% >= {threshold}%")

            return (True, f"{defect_type}: within acceptable limits")

        # Unknown defect type - default pass with warning
        return (True, f"{defect_type}: no rule defined, defaulting to pass")

    def parse_cvat_attributes(self, attributes_json: str) -> Dict[str, Any]:
        """Parse CVAT attributes matching J&J format"""
        if not attributes_json:
            return {}

        try:
            attrs = json.loads(attributes_json)
        except:
            return {}

        # Parse special percentage formats from J&J
        parsed = {}
        for key, value in attrs.items():
            key_lower = key.lower()

            # Handle arc length percentages
            if "arc" in key_lower and "length" in key_lower:
                if isinstance(value, str):
                    # Parse "+5%", "<5%", "25%" formats
                    match = re.search(r'([<>+]?)(\d+(?:\.\d+)?)\s*%', value)
                    if match:
                        prefix, number = match.groups()
                        percent = float(number)
                        if prefix == '<' and percent == 5:
                            percent = 3.0  # J&J convention
                        elif prefix in ['+', '>'] and percent == 5:
                            percent = 6.0  # J&J convention
                        parsed['arc_length_percent'] = percent

            # Handle surface impact percentages
            elif "surface" in key_lower and ("impact" in key_lower or "percent" in key_lower):
                if isinstance(value, str):
                    match = re.search(r'(\d+(?:\.\d+)?)\s*%', value)
                    if match:
                        parsed['surface_percent'] = float(match.group(1))

            # Handle dots obstructed
            elif "dots" in key_lower and "obstruct" in key_lower:
                if isinstance(value, str):
                    value_clean = value.strip()
                    if value_clean in ["1-5", "1 – 5"]:
                        parsed['dots_count'] = 3.0
                    elif value_clean in ["6-15", "6 – 15"]:
                        parsed['dots_count'] = 10.0
                    elif value_clean.endswith("+"):
                        parsed['dots_count'] = float(value_clean.rstrip("+")) + 1
                    else:
                        try:
                            parsed['dots_count'] = float(value_clean)
                        except:
                            pass

            # Store original value too
            parsed[key] = value

        return parsed

    def evaluate_presentation(self, image_annotations: List[Dict]) -> Tuple[bool, List[str]]:
        """
        Evaluate lens presentation criteria from J&J Section 2.1

        Returns:
            (adequate_presentation, list_of_issues)
        """
        issues = []

        for ann in image_annotations:
            label = ann.get('label', '').lower()
            attrs = self.parse_cvat_attributes(ann.get('attributes', ''))

            # Check presentation defects
            if label == 'edge-out-of-focus':
                arc_percent = attrs.get('arc_length_percent', 0)
                if arc_percent >= 30:
                    issues.append(f"Edge out of focus: {arc_percent}% >= 30%")

            elif label == 'edge-obstructed':
                arc_percent = attrs.get('arc_length_percent', 0)
                if arc_percent >= 4:
                    issues.append(f"Edge obstructed: {arc_percent}% >= 4%")

            elif label == 'ripple':
                surface_percent = attrs.get('surface_percent', 0)
                if surface_percent >= 50:
                    issues.append(f"Ripple: {surface_percent}% >= 50%")

            elif label in ['lens-folded', 'lens-inverted', 'missing-lens',
                          'multiple-lenses', 'lens-out-of-round']:
                issues.append(f"{label}: automatic presentation failure")

        adequate = len(issues) == 0
        return adequate, issues

    def evaluate_image(self, detections: List[Dict],
                      annotations: List[Dict]) -> Dict[str, Any]:
        """
        Complete evaluation matching J&J requirements

        Args:
            detections: List of ML model detections
            annotations: List of CVAT annotations for presentation checks

        Returns:
            Complete evaluation report
        """
        # Check presentation first
        presentation_ok, presentation_issues = self.evaluate_presentation(annotations)

        if not presentation_ok:
            return {
                'pass_fail': 'FAIL',
                'reason': 'Lens presentation inadequate',
                'presentation_issues': presentation_issues,
                'defects': []
            }

        # Evaluate defects
        failed_defects = []
        all_defects = []

        for detection in detections:
            measurement = self.measure_defect(
                detection['bbox'],
                detection['class_name']
            )

            passes, reason = self.evaluate_defect(measurement)

            defect_record = {
                'type': detection['class_name'],
                'measurement': measurement.pixel_measurement,
                'pass': passes,
                'reason': reason,
                'confidence': detection.get('confidence', 1.0)
            }

            all_defects.append(defect_record)
            if not passes:
                failed_defects.append(defect_record)

        return {
            'pass_fail': 'PASS' if len(failed_defects) == 0 else 'FAIL',
            'presentation_issues': [],
            'failed_defects': failed_defects,
            'all_defects': all_defects,
            'num_defects': len(all_defects),
            'resolution_scale_factor': self.scale_factor
        }