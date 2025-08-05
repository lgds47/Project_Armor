#!/usr/bin/env python3
"""
Test script for PerformanceLogger class.

This script tests the functionality of the PerformanceLogger class to ensure that
it correctly logs messages, tracks performance metrics, and provides a singleton instance.
"""

import os
import sys
import time
import json
from pathlib import Path
import logging

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import PerformanceLogger
from armor_pipeline.utils.performance_logger import PerformanceLogger


def test_basic_logging():
    """Test basic logging functionality."""
    print("\nTesting basic logging functionality...")
    
    # Create a temporary directory for testing
    temp_dir = Path("./temp_logs")
    
    # Create a PerformanceLogger instance
    logger = PerformanceLogger(log_dir=temp_dir)
    
    # Check that the log directory was created
    print(f"Log directory exists: {temp_dir.exists()}")
    
    # Log messages at different levels
    logger.log_info("This is an info message")
    logger.log_warning("This is a warning message")
    logger.log_error("This is an error message")
    logger.log_debug("This is a debug message")
    
    # Check that the log file was created
    log_files = list(temp_dir.glob("*.log"))
    print(f"Log files: {log_files}")
    
    # Clean up
    try:
        import shutil
        shutil.rmtree(temp_dir)
        print(f"Removed temporary directory: {temp_dir}")
    except Exception as e:
        print(f"Error removing temporary directory: {e}")


def test_performance_metrics():
    """Test performance metrics tracking."""
    print("\nTesting performance metrics tracking...")
    
    # Create a temporary directory for testing
    temp_dir = Path("./temp_logs")
    
    # Create a PerformanceLogger instance
    logger = PerformanceLogger(log_dir=temp_dir)
    
    # Log inference time
    logger.log_inference_time("test_model", 123.45)
    
    # Log memory usage
    logger.log_memory_usage("test_model", 256.0, 512.0)
    
    # Log model accuracy
    logger.log_model_accuracy("test_model", {
        "precision": 0.95,
        "recall": 0.92,
        "f1_score": 0.93
    })
    
    # Get performance summary
    summary = logger.get_performance_summary()
    print("Performance summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Save metrics to a file
    metrics_file = logger.save_metrics()
    print(f"Metrics saved to: {metrics_file}")
    
    # Check that the metrics file was created and contains the expected data
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            metrics_data = json.load(f)
        
        print("Metrics data:")
        print(f"  Session duration: {metrics_data['session_duration_seconds']:.2f} seconds")
        print(f"  Inference times: {len(metrics_data['metrics']['inference_times'])}")
        print(f"  Memory usage: {len(metrics_data['metrics']['memory_usage'])}")
        print(f"  Model accuracy: {list(metrics_data['metrics']['model_accuracy'].keys())}")
    
    # Clean up
    try:
        import shutil
        shutil.rmtree(temp_dir)
        print(f"Removed temporary directory: {temp_dir}")
    except Exception as e:
        print(f"Error removing temporary directory: {e}")


def test_singleton_pattern():
    """Test singleton pattern."""
    print("\nTesting singleton pattern...")
    
    # Create a temporary directory for testing
    temp_dir = Path("./temp_logs")
    
    # Create a PerformanceLogger instance
    logger1 = PerformanceLogger.get_instance(log_dir=temp_dir)
    
    # Create another PerformanceLogger instance
    logger2 = PerformanceLogger.get_instance()
    
    # Check that both instances are the same
    print(f"logger1 id: {id(logger1)}")
    print(f"logger2 id: {id(logger2)}")
    print(f"Same instance: {logger1 is logger2}")
    
    # Log a message using the second instance
    logger2.log_info("This message should appear in the log file created by the first instance")
    
    # Clean up
    try:
        import shutil
        shutil.rmtree(temp_dir)
        print(f"Removed temporary directory: {temp_dir}")
    except Exception as e:
        print(f"Error removing temporary directory: {e}")
    
    # Reset the singleton instance for other tests
    PerformanceLogger._instance = None


def test_exception_logging():
    """Test exception logging."""
    print("\nTesting exception logging...")
    
    # Create a temporary directory for testing
    temp_dir = Path("./temp_logs")
    
    # Create a PerformanceLogger instance
    logger = PerformanceLogger(log_dir=temp_dir)
    
    # Log an exception
    try:
        # Raise an exception
        raise ValueError("This is a test exception")
    except Exception as e:
        logger.log_error("An error occurred", exception=e)
    
    # Clean up
    try:
        import shutil
        shutil.rmtree(temp_dir)
        print(f"Removed temporary directory: {temp_dir}")
    except Exception as e:
        print(f"Error removing temporary directory: {e}")


def main():
    """Run tests."""
    print("Testing PerformanceLogger class...")
    
    # Test basic logging
    test_basic_logging()
    
    # Test performance metrics
    test_performance_metrics()
    
    # Test singleton pattern
    test_singleton_pattern()
    
    # Test exception logging
    test_exception_logging()
    
    print("\nAll tests completed.")


if __name__ == "__main__":
    main()