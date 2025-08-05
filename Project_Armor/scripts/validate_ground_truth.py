#!/usr/bin/env python3
"""
Validate predictions against J&J ground truth CSV
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from typing import Dict, List


def validate_predictions(predictions_path: Path, ground_truth_path: Path):
    """Compare predictions with J&J ground truth"""

    # Load data
    pred_df = pd.read_csv(predictions_path)
    gt_df = pd.read_csv(ground_truth_path)

    # Match by annotation_id
    merged = pd.merge(
        pred_df,
        gt_df,
        on=['annotation_id', 'image'],
        suffixes=('_pred', '_gt')
    )

    # Compare pass/fail decisions
    matches = merged['pass_fail_pred'] == merged['pass_fail_gt']
    accuracy = matches.mean()

    print(f"\n=== VALIDATION RESULTS ===")
    print(f"Total annotations: {len(merged)}")
    print(f"Matching decisions: {matches.sum()}")
    print(f"Accuracy: {accuracy:.2%}")

    # Analyze mismatches
    mismatches = merged[~matches]
    if len(mismatches) > 0:
        print(f"\nMismatches: {len(mismatches)}")
        print("\nMismatch details:")
        for _, row in mismatches.head(10).iterrows():
            print(f"\nImage: {row['image']}")
            print(f"Label: {row['label']}")
            print(f"Your decision: {row['pass_fail_pred']}")
            print(f"J&J decision: {row['pass_fail_gt']}")
            print(f"J&J criteria: {row['criteria']}")

    # Per-defect type analysis
    print("\n=== PER-DEFECT TYPE ACCURACY ===")
    for defect_type in merged['label'].unique():
        defect_matches = merged[merged['label'] == defect_type]
        defect_accuracy = (defect_matches['pass_fail_pred'] ==
                           defect_matches['pass_fail_gt']).mean()
        print(f"{defect_type}: {defect_accuracy:.2%} ({len(defect_matches)} samples)")

    return accuracy, merged


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--ground-truth', type=Path, required=True)
    args = parser.parse_args()

    validate_predictions(args.predictions, args.ground_truth)