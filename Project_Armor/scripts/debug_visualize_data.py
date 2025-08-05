#!/usr/bin/env python3
"""
Visualization script for J&J Contact Lens annotations
Helps verify data loading and annotation parsing
"""

import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from typing import List
import random

from armor_pipeline.data.parser import parse_all_annotations, DefectType


def visualize_annotations(
        annotations_dir: Path,
        images_dir: Path,
        output_dir: Path,
        num_samples: int = 5
):
    """Visualize sample annotations to verify parsing"""
    output_dir.mkdir(exist_ok=True, parents=True)

    print(f"Parsing annotations from {annotations_dir}...")
    annotations = parse_all_annotations(annotations_dir, images_dir, crop_to_upper_half=True)

    print(f"Found {len(annotations)} annotated images")

    # Get defect statistics
    defect_counts = {}
    for ann in annotations:
        for defect in ann.defects:
            defect_counts[defect.name] = defect_counts.get(defect.name, 0) + 1

    print("\nDefect type distribution:")
    for name, count in sorted(defect_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {name}: {count}")

    # Sample random annotations
    sample_annotations = random.sample(annotations, min(num_samples, len(annotations)))

    # Visualize each sample
    for idx, ann in enumerate(sample_annotations):
        if not ann.image_path.exists():
            print(f"Warning: Image not found: {ann.image_path}")
            continue

        # Load image
        image = cv2.imread(str(ann.image_path))
        if image is None:
            print(f"Error loading image: {ann.image_path}")
            continue

        # Convert BGR to RGB and crop to upper half
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image[:2048, :]

        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # Show original image
        ax1.imshow(image)
        ax1.set_title(f"Original: {ann.image_path.name}")
        ax1.axis('off')

        # Show image with annotations
        ax2.imshow(image)
        ax2.set_title(f"Annotated: {len(ann.defects)} defects")

        # Define colors for different defect types
        type_colors = {
            DefectType.POLYLINE: 'red',
            DefectType.POLYGON: 'blue',
            DefectType.ELLIPSE: 'green',
            DefectType.BBOX: 'yellow'
        }

        # Draw each defect
        for defect in ann.defects:
            color = type_colors.get(defect.defect_type, 'white')

            if defect.defect_type == DefectType.POLYLINE:
                # Draw polyline
                points = np.array([p.to_tuple() for p in defect.points])
                for i in range(len(points) - 1):
                    ax2.plot(
                        [points[i][0], points[i + 1][0]],
                        [points[i][1], points[i + 1][1]],
                        color=color, linewidth=2
                    )

            elif defect.defect_type == DefectType.POLYGON:
                # Draw polygon
                points = np.array([p.to_tuple() for p in defect.points])
                polygon = patches.Polygon(points, fill=False, edgecolor=color, linewidth=2)
                ax2.add_patch(polygon)

            elif defect.defect_type == DefectType.ELLIPSE:
                # Draw ellipse
                ellipse = patches.Ellipse(
                    (defect.center.x, defect.center.y),
                    2 * defect.rx, 2 * defect.ry,
                    angle=defect.angle,
                    fill=False, edgecolor=color, linewidth=2
                )
                ax2.add_patch(ellipse)

            elif defect.defect_type == DefectType.BBOX:
                # Draw bounding box
                rect = patches.Rectangle(
                    (defect.x_min, defect.y_min),
                    defect.x_max - defect.x_min,
                    defect.y_max - defect.y_min,
                    fill=False, edgecolor=color, linewidth=2
                )
                ax2.add_patch(rect)

            # Add label
            bbox = defect.to_bbox()
            ax2.text(
                bbox[0], bbox[1] - 5,
                f"{defect.name} ({defect.defect_type.value})",
                color='white', backgroundcolor=color,
                fontsize=8, weight='bold'
            )

        ax2.axis('off')

        # Save figure
        output_path = output_dir / f"visualization_{idx:03d}.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved: {output_path}")

    # Create defect type legend
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis('off')

    legend_elements = [
        patches.Patch(color='red', label='Polyline (Edge tears/scratches)'),
        patches.Patch(color='blue', label='Polygon (Surface blobs)'),
        patches.Patch(color='green', label='Ellipse (Bubbles/rings)'),
        patches.Patch(color='yellow', label='Bounding Box (Macro defects)')
    ]

    ax.legend(handles=legend_elements, loc='center', fontsize=12)
    ax.set_title('Defect Type Legend', fontsize=14, weight='bold')

    legend_path = output_dir / "defect_type_legend.png"
    plt.savefig(legend_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nVisualization complete! Check {output_dir}")


def visualize_augmentations(
        annotations_dir: Path,
        images_dir: Path,
        output_dir: Path
):
    """Visualize data augmentations"""
    from armor_pipeline.data.dataset import ContactLensDataset, get_transform
    from armor_pipeline.data.parser import parse_all_annotations

    output_dir.mkdir(exist_ok=True, parents=True)

    # Parse annotations
    annotations = parse_all_annotations(annotations_dir, images_dir, crop_to_upper_half=True)

    if not annotations:
        print("No annotations found!")
        return

    # Create dataset with augmentations
    train_transform = get_transform("train", img_size=1024)
    dataset = ContactLensDataset(annotations[:5], transform=train_transform)

    # Visualize augmentations for first image
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for i in range(8):
        image, target = dataset[0]

        # Denormalize image
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_np = image.numpy().transpose(1, 2, 0)
        image_np = image_np * std + mean
        image_np = np.clip(image_np, 0, 1)

        axes[i].imshow(image_np)
        axes[i].set_title(f"Augmentation {i + 1}")
        axes[i].axis('off')

        # Draw bboxes
        for box in target['boxes']:
            x, y, w, h = box
            rect = patches.Rectangle(
                (x, y), w, h,
                fill=False, edgecolor='red', linewidth=1
            )
            axes[i].add_patch(rect)

    plt.tight_layout()
    aug_path = output_dir / "augmentation_examples.png"
    plt.savefig(aug_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Augmentation examples saved to: {aug_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize J&J contact lens annotations")
    parser.add_argument(
        '--data-root',
        type=Path,
        default=None,
        help='Root directory containing data (overrides environment variables)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default='visualization_output',
        help='Output directory for visualizations'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=5,
        help='Number of samples to visualize'
    )
    parser.add_argument(
        '--show-augmentations',
        action='store_true',
        help='Show data augmentation examples'
    )

    args = parser.parse_args()

    # Use ProjectConfig to manage paths
    from armor_pipeline.utils.project_config import get_project_config
    project_config = get_project_config(base_path=args.data_root)
    
    # Get paths from ProjectConfig
    annotations_dir = project_config.annotation_path
    images_dir = project_config.image_path

    if not annotations_dir.exists():
        print(f"Error: Annotations directory not found: {annotations_dir}")
        return

    if not images_dir.exists():
        print(f"Error: Images directory not found: {images_dir}")
        return

    # Visualize annotations
    visualize_annotations(
        annotations_dir,
        images_dir,
        args.output_dir,
        args.num_samples
    )

    # Visualize augmentations if requested
    if args.show_augmentations:
        visualize_augmentations(
            annotations_dir,
            images_dir,
            args.output_dir
        )


if __name__ == '__main__':
    main()