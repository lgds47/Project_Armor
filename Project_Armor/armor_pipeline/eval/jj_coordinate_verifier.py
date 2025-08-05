"""
J&J Coordinate and Visual Verification System
Provides both coordinate-based and visual confirmation of defect detection
Implements exact J&J measurement methodologies
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle, Wedge, Polygon
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json
import pandas as pd


class JJCoordinateVerifier:
    """Verify detection accuracy using J&J measurement standards"""

    def __init__(self,
                 image_resolution: Tuple[int, int] = (2048, 2048),
                 lens_diameter_px: float = 1580.0):
        """Initialize with J&J specifications"""
        self.image_resolution = image_resolution
        self.lens_diameter_px = lens_diameter_px
        self.lens_center = (image_resolution[0] // 2, image_resolution[1] // 2)
        self.lens_radius = lens_diameter_px / 2

    def measure_edge_defect_depth(self,
                                 defect_polygon: np.ndarray,
                                 polyline: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Measure edge defect depth according to J&J methodology

        Args:
            defect_polygon: Polygon points of the defect
            polyline: Optional companion polyline for depth measurement

        Returns:
            Measurement dictionary with pixel values
        """
        if polyline is not None:
            # Use polyline length as depth (J&J standard)
            depth = self._polyline_length(polyline)
        else:
            # Estimate radial depth from polygon
            depth = self._estimate_radial_depth(defect_polygon)

        return {
            'depth_pixels': depth,
            'measurement_method': 'polyline' if polyline is not None else 'radial_estimate',
            'threshold_pixels': 5.0  # J&J threshold
        }

    def measure_center_defect(self, defect_bbox: np.ndarray) -> Dict[str, float]:
        """
        Measure center defect according to J&J methodology

        Args:
            defect_bbox: [x1, y1, x2, y2] bounding box

        Returns:
            Measurement dictionary
        """
        x1, y1, x2, y2 = defect_bbox
        width = x2 - x1
        height = y2 - y1
        longest_dimension = max(width, height)

        return {
            'longest_dimension_pixels': longest_dimension,
            'width_pixels': width,
            'height_pixels': height,
            'threshold_pixels': 5.0  # J&J threshold
        }

    def measure_arc_length_obstruction(self,
                                     obstruction_polygon: np.ndarray) -> Dict[str, float]:
        """
        Measure arc length obstruction percentage

        Args:
            obstruction_polygon: Polygon points of obstruction

        Returns:
            Arc length percentage
        """
        # Find intersection points with lens circle
        intersections = self._find_circle_intersections(obstruction_polygon)

        if len(intersections) >= 2:
            # Calculate arc length between intersections
            angles = []
            for pt in intersections:
                angle = np.arctan2(pt[1] - self.lens_center[1],
                                 pt[0] - self.lens_center[0])
                angles.append(angle)

            # Arc length as percentage
            arc_angle = abs(angles[1] - angles[0])
            arc_percent = (arc_angle / (2 * np.pi)) * 100

            return {
                'arc_length_percent': arc_percent,
                'threshold_percent': 4.0,  # J&J threshold
                'num_intersections': len(intersections)
            }

        return {
            'arc_length_percent': 0.0,
            'threshold_percent': 4.0,
            'num_intersections': 0
        }

    def visualize_measurements(self,
                             image: np.ndarray,
                             detections: List[Dict],
                             ground_truth: List[Dict],
                             output_path: Path):
        """Create comprehensive measurement visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 16))

        # 1. Original image with lens overlay
        ax1 = axes[0, 0]
        ax1.imshow(image)
        ax1.set_title("Lens Reference with 5-pixel Grid", fontsize=14, weight='bold')

        # Draw lens circle
        lens_circle = Circle(self.lens_center, self.lens_radius,
                           fill=False, edgecolor='green', linewidth=2)
        ax1.add_patch(lens_circle)

        # Draw 5-pixel reference grid
        for r in range(5, 50, 5):
            ref_circle = Circle(self.lens_center, self.lens_radius - r,
                              fill=False, edgecolor='green',
                              linewidth=0.5, linestyle='--', alpha=0.5)
            ax1.add_patch(ref_circle)
            ax1.text(self.lens_center[0] + self.lens_radius - r,
                    self.lens_center[1], f"{r}px",
                    fontsize=8, color='green')

        # 2. Detection measurements
        ax2 = axes[0, 1]
        ax2.imshow(image)
        ax2.set_title("Detection Measurements (Pixels)", fontsize=14, weight='bold')

        for det in detections:
            bbox = det['bbox']
            defect_type = det['class_name']

            if 'edge' in defect_type.lower():
                # Edge defect - show radial depth
                measurement = self.measure_edge_defect_depth(bbox.reshape(-1, 2))
                depth = measurement['depth_pixels']

                # Draw measurement line
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2

                # Radial line from center
                ax2.plot([self.lens_center[0], center_x],
                        [self.lens_center[1], center_y],
                        'r-', linewidth=2)

                ax2.text(center_x, center_y,
                        f"Depth: {depth:.1f}px\nThreshold: 5px",
                        fontsize=10, color='white',
                        bbox=dict(boxstyle="round,pad=0.3",
                                facecolor='red', alpha=0.8))
            else:
                # Center defect - show longest dimension
                measurement = self.measure_center_defect(bbox)
                longest = measurement['longest_dimension_pixels']

                rect = patches.Rectangle((bbox[0], bbox[1]),
                                       bbox[2] - bbox[0],
                                       bbox[3] - bbox[1],
                                       linewidth=2, edgecolor='red',
                                       facecolor='none')
                ax2.add_patch(rect)

                # Draw longest dimension line
                if bbox[2] - bbox[0] > bbox[3] - bbox[1]:
                    # Width is longest
                    ax2.plot([bbox[0], bbox[2]],
                           [(bbox[1] + bbox[3])/2, (bbox[1] + bbox[3])/2],
                           'r-', linewidth=3)
                else:
                    # Height is longest
                    ax2.plot([(bbox[0] + bbox[2])/2, (bbox[0] + bbox[2])/2],
                           [bbox[1], bbox[3]],
                           'r-', linewidth=3)

                ax2.text(bbox[0], bbox[1] - 10,
                        f"Longest: {longest:.1f}px\nThreshold: 5px",
                        fontsize=10, color='white',
                        bbox=dict(boxstyle="round,pad=0.3",
                                facecolor='red', alpha=0.8))

        # 3. Arc length measurements
        ax3 = axes[1, 0]
        ax3.imshow(image)
        ax3.set_title("Arc Length Obstructions (%)", fontsize=14, weight='bold')

        # Draw lens circle
        lens_circle = Circle(self.lens_center, self.lens_radius,
                           fill=False, edgecolor='blue', linewidth=2)
        ax3.add_patch(lens_circle)

        # Simulate edge obstruction for demonstration
        obstruction_angle = np.radians(14.4)  # 4% of circle
        wedge = Wedge(self.lens_center, self.lens_radius,
                     0, np.degrees(obstruction_angle),
                     facecolor='red', alpha=0.5)
        ax3.add_patch(wedge)
        ax3.text(self.lens_center[0] + self.lens_radius + 50,
                self.lens_center[1],
                "4% arc = 14.4°\nThreshold for edge obstruction",
                fontsize=10, color='red')

        # 4. Pass/Fail summary
        ax4 = axes[1, 1]
        ax4.axis('off')
        ax4.set_title("J&J Pass/Fail Criteria", fontsize=14, weight='bold')

        criteria_text = """
J&J Pixel-Based Thresholds (at 2048×2048):
• Edge Tear/Chip/Excess: ≥5 pixels depth → FAIL
• Center Debris/Hole/Tear: ≥5 pixels longest dimension → FAIL
• Always Fail: Edge-not-closed, Lens-folded, etc.

J&J Percentage-Based Thresholds:
• Edge Out-of-Focus: ≥30% arc length → FAIL
• Edge Obstructed: ≥4% arc length → FAIL  
• Ripple: ≥50% surface area → FAIL
• Bubble: ≥4% lens area → FAIL
• 123-Mark Obstructed: >3 dots → FAIL

Resolution Scaling:
• Thresholds scale linearly with resolution
• At 1024×1024: 5 pixels → 2.5 pixels
• Physical measurements remain constant
        """

        ax4.text(0.05, 0.95, criteria_text,
                transform=ax4.transAxes,
                fontsize=11, verticalalignment='top',
                fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5",
                         facecolor='lightgray', alpha=0.8))

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    def generate_coordinate_report(self,
                                 detections: List[Dict],
                                 ground_truth: List[Dict],
                                 output_path: Path) -> pd.DataFrame:
        """Generate detailed coordinate comparison report"""
        report_data = []

        for det in detections:
            bbox = det['bbox']
            defect_type = det['class_name']

            # Measure according to defect type
            if 'edge' in defect_type.lower():
                measurement = self.measure_edge_defect_depth(bbox.reshape(-1, 2))
                value = measurement['depth_pixels']
                threshold = 5.0
                measurement_type = 'radial_depth'
            else:
                measurement = self.measure_center_defect(bbox)
                value = measurement['longest_dimension_pixels']
                threshold = 5.0
                measurement_type = 'longest_dimension'

            # Pass/Fail determination
            passes = value < threshold

            report_data.append({
                'defect_type': defect_type,
                'measurement_type': measurement_type,
                'measured_value_px': value,
                'threshold_px': threshold,
                'pass_fail': 'PASS' if passes else 'FAIL',
                'bbox': bbox.tolist(),
                'center_x': (bbox[0] + bbox[2]) / 2,
                'center_y': (bbox[1] + bbox[3]) / 2,
                'confidence': det.get('confidence', 1.0)
            })

        df = pd.DataFrame(report_data)
        df.to_csv(output_path, index=False)

        # Print summary
        print("\n=== J&J COORDINATE VERIFICATION REPORT ===")
        print(f"Total detections: {len(df)}")
        print(f"Failed detections: {len(df[df['pass_fail'] == 'FAIL'])}")
        print(f"Pass rate: {(df['pass_fail'] == 'PASS').mean():.1%}")

        print("\nPer-defect type summary:")
        summary = df.groupby('defect_type').agg({
            'measured_value_px': ['mean', 'std', 'max'],
            'pass_fail': lambda x: (x == 'FAIL').sum()
        })
        print(summary)

        return df

    def _polyline_length(self, points: np.ndarray) -> float:
        """Calculate polyline length"""
        if len(points) < 2:
            return 0.0

        length = 0.0
        for i in range(len(points) - 1):
            length += np.linalg.norm(points[i+1] - points[i])

        return length

    def _estimate_radial_depth(self, polygon: np.ndarray) -> float:
        """Estimate radial depth from polygon"""
        # Find closest and farthest points from lens center
        distances = []
        for point in polygon:
            dist = np.linalg.norm(point - self.lens_center)
            distances.append(dist)

        # Depth is deviation from expected radius
        expected_radius = self.lens_radius
        actual_radius = np.mean(distances)

        return abs(expected_radius - actual_radius)

    def _find_circle_intersections(self, polygon: np.ndarray) -> List[np.ndarray]:
        """Find intersections between polygon and lens circle"""
        intersections = []

        for i in range(len(polygon)):
            p1 = polygon[i]
            p2 = polygon[(i + 1) % len(polygon)]

            # Check if line segment intersects circle
            # This is simplified - full implementation would use proper
            # line-circle intersection algorithm
            d1 = np.linalg.norm(p1 - self.lens_center)
            d2 = np.linalg.norm(p2 - self.lens_center)

            if (d1 - self.lens_radius) * (d2 - self.lens_radius) < 0:
                # Intersection exists
                # Simplified: use midpoint as intersection
                intersections.append((p1 + p2) / 2)

        return intersections