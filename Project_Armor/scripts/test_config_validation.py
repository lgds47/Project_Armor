#!/usr/bin/env python
"""
Test script for demonstrating the pipeline configuration validation.

This script demonstrates the use of the ArmorPipeline.validate_pipeline_config() method
by testing it with different configuration scenarios.

Usage:
    python test_config_validation.py [--config CONFIG_PATH]

Examples:
    python test_config_validation.py
    python test_config_validation.py --config path/to/custom/config.yaml
"""

import argparse
import json
import os
import sys
import yaml
from pathlib import Path
import tempfile

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import from project
from Project_Armor.cli import ArmorPipeline
from Project_Armor.armor_pipeline.utils.performance_logger import PerformanceLogger


def create_test_config(scenario="valid"):
    """
    Create a test configuration file for the specified scenario.
    
    Args:
        scenario: The test scenario to create a configuration for.
                 Options: "valid", "missing_params", "invalid_paths", "invalid_registry"
                 
    Returns:
        Path to the created configuration file
    """
    # Create a temporary directory for test files
    temp_dir = Path(tempfile.mkdtemp())
    
    # Create subdirectories
    data_dir = temp_dir / "data"
    data_dir.mkdir(exist_ok=True)
    
    output_dir = temp_dir / "output"
    output_dir.mkdir(exist_ok=True)
    
    model_zoo_dir = temp_dir / "model_zoo"
    model_zoo_dir.mkdir(exist_ok=True)
    
    config_dir = temp_dir / "config"
    config_dir.mkdir(exist_ok=True)
    
    # Create a valid registry file
    registry_path = model_zoo_dir / "registry.json"
    valid_registry = {
        "faster_rcnn": {
            "name": "faster_rcnn",
            "model_type": "faster_rcnn",
            "backbone": "resnet50",
            "num_classes": 2,
            "input_size": 1024,
            "confidence_threshold": 0.5,
            "nms_threshold": 0.5,
            "pretrained": True
        }
    }
    
    with open(registry_path, 'w') as f:
        json.dump(valid_registry, f, indent=2)
    
    # Create a valid pass/fail config
    pass_fail_path = config_dir / "pass_fail.yaml"
    valid_pass_fail = {
        "pass_fail_thresholds": {
            "default": 0.5,
            "edge_tear": 0.7,
            "bubble": 0.6
        }
    }
    
    with open(pass_fail_path, 'w') as f:
        yaml.dump(valid_pass_fail, f)
    
    # Create config based on scenario
    config_path = temp_dir / "pipeline_config.yaml"
    
    if scenario == "valid":
        config = {
            "data": {
                "root_dir": str(data_dir),
                "num_workers": 2,
                "img_size": 1024,
                "train_split": 0.8
            },
            "training": {
                "batch_size": 4,
                "learning_rate": 0.001,
                "epochs": 10,
                "patience": 5,
                "save_frequency": 2
            },
            "models": {
                "registry_path": str(registry_path)
            },
            "output": {
                "dir": str(output_dir)
            },
            "eval": {
                "iou_threshold": 0.5,
                "pass_fail_config": str(pass_fail_path)
            }
        }
    elif scenario == "missing_params":
        # Missing some required parameters
        config = {
            "data": {
                "root_dir": str(data_dir),
                # Missing num_workers
                "img_size": 1024,
                # Missing train_split
            },
            "training": {
                "batch_size": 4,
                # Missing learning_rate
                "epochs": 10
            },
            "models": {
                "registry_path": str(registry_path)
            },
            "output": {
                "dir": str(output_dir)
            },
            # Missing eval section
        }
    elif scenario == "invalid_paths":
        # Invalid file paths
        config = {
            "data": {
                "root_dir": str(temp_dir / "nonexistent"),
                "num_workers": 2,
                "img_size": 1024,
                "train_split": 0.8
            },
            "training": {
                "batch_size": 4,
                "learning_rate": 0.001,
                "epochs": 10
            },
            "models": {
                "registry_path": str(temp_dir / "nonexistent_registry.json")
            },
            "output": {
                "dir": str(output_dir)
            },
            "eval": {
                "iou_threshold": 0.5,
                "pass_fail_config": str(temp_dir / "nonexistent_pass_fail.yaml")
            }
        }
    elif scenario == "invalid_registry":
        # Create an invalid registry file
        invalid_registry_path = model_zoo_dir / "invalid_registry.json"
        with open(invalid_registry_path, 'w') as f:
            f.write("{ invalid json }")
        
        config = {
            "data": {
                "root_dir": str(data_dir),
                "num_workers": 2,
                "img_size": 1024,
                "train_split": 0.8
            },
            "training": {
                "batch_size": 4,
                "learning_rate": 0.001,
                "epochs": 10
            },
            "models": {
                "registry_path": str(invalid_registry_path)
            },
            "output": {
                "dir": str(output_dir)
            },
            "eval": {
                "iou_threshold": 0.5,
                "pass_fail_config": str(pass_fail_path)
            }
        }
    
    # Write config to file
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    return config_path, temp_dir


def test_validation(config_path):
    """
    Test the validate_pipeline_config method with the given configuration.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Validation report
    """
    # Initialize logger
    logger = PerformanceLogger.get_instance()
    
    print(f"\nTesting configuration validation with: {config_path}")
    
    # Initialize pipeline
    pipeline = ArmorPipeline(config_path)
    
    # Validate configuration
    print("Validating pipeline configuration...")
    report = pipeline.validate_pipeline_config()
    
    # Print summary
    print(f"\nValidation Summary: {report['summary']}")
    
    # Print issues by category
    for category, data in report["categories"].items():
        if data["issues"]:
            print(f"\n{category.upper()} ISSUES:")
            for issue in data["issues"]:
                print(f"  [{issue['severity'].upper()}] {issue['message']}")
                if issue["recommendation"]:
                    print(f"    Recommendation: {issue['recommendation']}")
    
    return report


def main():
    """Main function to run the test script."""
    parser = argparse.ArgumentParser(description="Test pipeline configuration validation")
    parser.add_argument(
        "--config",
        help="Path to a custom configuration file to validate"
    )
    parser.add_argument(
        "--scenario",
        choices=["valid", "missing_params", "invalid_paths", "invalid_registry", "all"],
        default="all",
        help="Test scenario to run"
    )
    
    args = parser.parse_args()
    
    print("=== ArmorPipeline Configuration Validation Test ===")
    
    if args.config:
        # Test with custom config
        test_validation(Path(args.config))
    else:
        # Test with generated configs
        scenarios = ["valid", "missing_params", "invalid_paths", "invalid_registry"]
        
        if args.scenario != "all":
            scenarios = [args.scenario]
        
        for scenario in scenarios:
            print(f"\n\n=== Testing {scenario.upper()} Configuration ===")
            config_path, temp_dir = create_test_config(scenario)
            
            try:
                test_validation(config_path)
            except Exception as e:
                print(f"Error during validation: {e}")
            
            # Clean up
            try:
                import shutil
                shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"Error cleaning up temporary directory: {e}")
    
    print("\n=== Test Complete ===")


if __name__ == "__main__":
    main()