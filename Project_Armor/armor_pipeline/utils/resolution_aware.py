"""
Resolution-aware Pass/Fail evaluation with proper scaling
Handles different input resolutions and maintains consistent severity thresholds
"""

import numpy as np
from typing import Dict, Tuple, List
from dataclasses import dataclass
import yaml
from pathlib import Path


@dataclass
class DefectMeasurement:
    """Physical defect measurements with units"""
    area_mm2: float  # Area in square millimeters
    length_mm: float  # Max dimension in millimeters
    pixel_area: int  # Original pixel area
    location: Tuple[float, float]  # Center coordinates (normalized 0-1)


class ResolutionAwareEvaluator:
    """Evaluator that properly handles resolution scaling for Pass/Fail decisions"""

    def __init__(
        self,
        pass_fail_config_path: Path,
        original_resolution: Tuple[int, int] = (2048, 2048),
        physical_size_mm: Tuple[float, float] = (14.2, 14.2),  # Typical contact lens diameter
        processing_resolution: Tuple[int, int] = (1024, 1024)
    ):
        """
        Args:
            pass_fail_config_path: Path to pass/fail thresholds
            original_resolution: Original image resolution after cropping
            physical_size_mm: Physical size of the imaging area in mm
            processing_resolution: Resolution used during inference
        """
        self.original_resolution = original_resolution
        self.physical_size_mm = physical_size_mm
        self.processing_resolution = processing_resolution

        # Calculate pixel-to-mm conversion factors
        self.mm_per_pixel_original = (
            physical_size_mm[0] / original_resolution[0],
            physical_size_mm[1] / original_resolution[1]
        )

        self.mm_per_pixel_processing = (
            physical_size_mm[0] / processing_resolution[0],
            physical_size_mm[1] / processing_resolution[1]
        )

        # Load pass/fail config with physical units
        self.load_pass_fail_config(pass_fail_config_path)

    def load_pass_fail_config(self, config_path: Path):
        """Load pass/fail configuration with physical measurements"""
        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Expected format:
        # defect_type:
        #   confidence: 0.6
        #   max_area_mm2: 0.05  # Maximum area in square millimeters
        #   max_length_mm: 0.5   # Maximum dimension in millimeters
        self.thresholds = config['pass_fail_thresholds']

    def convert_detection_to_measurement(
        self,
        bbox: np.ndarray,  # [x1, y1, x2, y2] in processing resolution
        confidence: float,
        defect_type: str
    ) -> DefectMeasurement:
        """Convert detection bbox to physical measurements"""
        x1, y1, x2, y2 = bbox

        # Calculate pixel area and dimensions at processing resolution
        pixel_width = x2 - x1
        pixel_height = y2 - y1
        pixel_area = pixel_width * pixel_height

        # Convert to physical measurements
        width_mm = pixel_width * self.mm_per_pixel_processing[0]
        height_mm = pixel_height * self.mm_per_pixel_processing[1]
        area_mm2 = width_mm * height_mm

        # Maximum dimension (length)
        length_mm = max(width_mm, height_mm)

        # Normalized center location (0-1)
        center_x = (x1 + x2) / 2 / self.processing_resolution[0]
        center_y = (y1 + y2) / 2 / self.processing_resolution[1]

        return DefectMeasurement(
            area_mm2=area_mm2,
            length_mm=length_mm,
            pixel_area=int(pixel_area),
            location=(center_x, center_y)
        )

    def evaluate_defect(
        self,
        measurement: DefectMeasurement,
        confidence: float,
        defect_type: str
    ) -> Tuple[bool, str, Dict]:
        """
        Evaluate if a defect causes the lens to fail

        Returns:
            fail: Boolean indicating if this defect fails the lens
            reason: String explanation of failure
            details: Dict with measurement details
        """
        # Get thresholds for this defect type
        if defect_type in self.thresholds:
            thresholds = self.thresholds[defect_type]
        else:
            thresholds = self.thresholds['default']

        # Check confidence first
        if confidence < thresholds['confidence']:
            return False, "Below confidence threshold", {
                'confidence': confidence,
                'threshold': thresholds['confidence']
            }

        # Check physical measurements
        fail_reasons = []

        if 'max_area_mm2' in thresholds:
            if measurement.area_mm2 > thresholds['max_area_mm2']:
                fail_reasons.append(
                    f"Area {measurement.area_mm2:.3f}mm² exceeds {thresholds['max_area_mm2']}mm²"
                )

        if 'max_length_mm' in thresholds:
            if measurement.length_mm > thresholds['max_length_mm']:
                fail_reasons.append(
                    f"Length {measurement.length_mm:.2f}mm exceeds {thresholds['max_length_mm']}mm"
                )

        # Location-based criteria (e.g., edge defects more critical)
        if 'edge_distance_factor' in thresholds:
            edge_distance = min(
                measurement.location[0],
                measurement.location[1],
                1 - measurement.location[0],
                1 - measurement.location[1]
            )
            if edge_distance < 0.1:  # Within 10% of edge
                # Apply stricter threshold for edge defects
                adjusted_area_threshold = thresholds['max_area_mm2'] * thresholds['edge_distance_factor']
                if measurement.area_mm2 > adjusted_area_threshold:
                    fail_reasons.append(
                        f"Edge defect area {measurement.area_mm2:.3f}mm² exceeds "
                        f"adjusted threshold {adjusted_area_threshold:.3f}mm²"
                    )

        details = {
            'area_mm2': measurement.area_mm2,
            'length_mm': measurement.length_mm,
            'location': measurement.location,
            'confidence': confidence,
            'thresholds': thresholds
        }

        if fail_reasons:
            return True, "; ".join(fail_reasons), details
        else:
            return False, "Pass", details

    def evaluate_lens(
        self,
        detections: List[Dict],
        image_id: str
    ) -> Dict:
        """
        Evaluate all detections for a lens

        Args:
            detections: List of detections with 'bbox', 'confidence', 'class_name'
            image_id: Identifier for the lens/image

        Returns:
            Evaluation report with pass/fail decision and details
        """
        lens_pass = True
        failed_defects = []
        all_measurements = []

        for detection in detections:
            measurement = self.convert_detection_to_measurement(
                detection['bbox'],
                detection['confidence'],
                detection['class_name']
            )

            fail, reason, details = self.evaluate_defect(
                measurement,
                detection['confidence'],
                detection['class_name']
            )

            measurement_record = {
                'defect_type': detection['class_name'],
                'confidence': detection['confidence'],
                'area_mm2': measurement.area_mm2,
                'length_mm': measurement.length_mm,
                'location': measurement.location,
                'pixel_bbox': detection['bbox'].tolist(),
                'pass': not fail,
                'reason': reason
            }

            all_measurements.append(measurement_record)

            if fail:
                lens_pass = False
                failed_defects.append(measurement_record)

        return {
            'image_id': image_id,
            'PASS_FAIL': 'PASS' if lens_pass else 'FAIL',
            'num_defects': len(detections),
            'num_failed_defects': len(failed_defects),
            'failed_defects': failed_defects,
            'all_measurements': all_measurements,
            'processing_resolution': self.processing_resolution,
            'physical_size_mm': self.physical_size_mm
        }


# Updated pass_fail.yaml format with physical units
PHYSICAL_PASS_FAIL_CONFIG = """
# Pass/Fail thresholds with physical measurements
# All measurements in real-world units (mm)

pass_fail_thresholds:
  default:
    confidence: 0.5
    max_area_mm2: 0.1      # 0.1 square millimeters
    max_length_mm: 0.5     # 0.5 millimeters
  
  edge_tear:
    confidence: 0.6
    max_area_mm2: 0.05     # Stricter for edge tears
    max_length_mm: 0.3
    edge_distance_factor: 0.5  # Even stricter near edges
  
  scratch:
    confidence: 0.6
    max_area_mm2: 0.02     # Very thin scratches
    max_length_mm: 1.0     # Can be longer but must be thin
  
  surface_blob:
    confidence: 0.5
    max_area_mm2: 0.15     # Larger area allowed
    max_length_mm: 0.6
  
  bubble:
    confidence: 0.7
    max_area_mm2: 0.08
    max_length_mm: 0.4
  
  ring_defect:
    confidence: 0.7
    max_area_mm2: 0.1
    max_length_mm: 0.5
    edge_distance_factor: 0.3  # Very strict near edges
  
  macro_defect:
    confidence: 0.4        # Lower confidence OK for large defects
    max_area_mm2: 0.2      # Larger defects
    max_length_mm: 0.8
"""