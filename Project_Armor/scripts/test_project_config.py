#!/usr/bin/env python3
"""
Test script for ProjectConfig class.

This script tests the functionality of the ProjectConfig class to ensure that
paths are correctly resolved, directories are created when they don't exist,
and environment variables are properly used.
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import ProjectConfig
from armor_pipeline.utils.project_config import get_project_config, ProjectConfig


def test_default_config():
    """Test default configuration."""
    config = get_project_config()
    print("\nDefault configuration:")
    print(config)
    
    # Check that directories exist
    print("\nChecking if directories exist:")
    print(f"  data_path: {config.data_path.exists()}")
    print(f"  image_path: {config.image_path.exists()}")
    print(f"  annotation_path: {config.annotation_path.exists()}")


def test_custom_base_path():
    """Test configuration with custom base path."""
    # Create a temporary directory
    temp_dir = Path("./temp_test_dir")
    
    # Get configuration with custom base path
    config = get_project_config(base_path=temp_dir)
    print("\nConfiguration with custom base path:")
    print(config)
    
    # Check that directories exist
    print("\nChecking if directories exist:")
    print(f"  data_path: {config.data_path.exists()}")
    print(f"  image_path: {config.image_path.exists()}")
    print(f"  annotation_path: {config.annotation_path.exists()}")
    
    # Clean up
    try:
        import shutil
        shutil.rmtree(temp_dir)
        print(f"\nRemoved temporary directory: {temp_dir}")
    except Exception as e:
        print(f"\nError removing temporary directory: {e}")


def test_environment_variables():
    """Test configuration with environment variables."""
    # Set environment variables
    os.environ["PROJECT_ARMOR_ROOT"] = "./env_test_dir"
    os.environ["DATA_ROOT"] = "./env_test_dir/custom_data"
    os.environ["IMAGE_PATH"] = "./env_test_dir/custom_images"
    os.environ["ANNOTATION_PATH"] = "./env_test_dir/custom_annotations"
    
    # Get configuration from environment variables
    config = ProjectConfig.from_env()
    print("\nConfiguration from environment variables:")
    print(config)
    
    # Check that directories exist
    print("\nChecking if directories exist:")
    print(f"  data_path: {config.data_path.exists()}")
    print(f"  image_path: {config.image_path.exists()}")
    print(f"  annotation_path: {config.annotation_path.exists()}")
    
    # Clean up
    try:
        import shutil
        shutil.rmtree(Path("./env_test_dir"))
        print(f"\nRemoved temporary directory: ./env_test_dir")
    except Exception as e:
        print(f"\nError removing temporary directory: {e}")
    
    # Unset environment variables
    del os.environ["PROJECT_ARMOR_ROOT"]
    del os.environ["DATA_ROOT"]
    del os.environ["IMAGE_PATH"]
    del os.environ["ANNOTATION_PATH"]


def main():
    """Run tests."""
    print("Testing ProjectConfig class...")
    
    # Test default configuration
    test_default_config()
    
    # Test configuration with custom base path
    test_custom_base_path()
    
    # Test configuration with environment variables
    test_environment_variables()
    
    print("\nAll tests completed.")


if __name__ == "__main__":
    main()