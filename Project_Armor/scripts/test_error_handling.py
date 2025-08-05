#!/usr/bin/env python
"""
Test script for demonstrating error handling in ArmorPipeline.

This script demonstrates the error handling capabilities of the ArmorPipeline.train() method
by simulating different error scenarios and showing how to handle and recover from them.

Usage:
    python test_error_handling.py [--scenario SCENARIO]

Scenarios:
    oom: Simulate out-of-memory error
    interrupt: Simulate keyboard interruption
    transient: Simulate transient error
    file_not_found: Simulate file not found error
    all: Run all scenarios (default)
"""

import argparse
import os
import signal
import sys
import time
from pathlib import Path

import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import from project
from Project_Armor.armor_pipeline.utils.performance_logger import PerformanceLogger
from Project_Armor.cli import ArmorPipeline


class ErrorSimulator:
    """Utility class for simulating different error scenarios."""

    @staticmethod
    def simulate_oom_error():
        """Simulate out-of-memory error by allocating a large tensor."""
        print("\n=== Simulating Out-of-Memory Error ===")
        
        # Only simulate if CUDA is available
        if not torch.cuda.is_available():
            print("CUDA not available, skipping OOM simulation")
            return
        
        # Try to allocate a tensor that's too large
        try:
            # Get current free memory
            free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
            
            # Try to allocate more than available
            print(f"Attempting to allocate tensor larger than available memory ({free_memory / 1e9:.2f} GB)")
            large_tensor = torch.zeros((int(free_memory), 1), device='cuda')
            
            # If we get here, we didn't OOM, so force an error
            raise torch.cuda.OutOfMemoryError("Simulated OOM error")
        except torch.cuda.OutOfMemoryError as e:
            print(f"Successfully simulated OOM error: {e}")
            return e

    @staticmethod
    def simulate_keyboard_interrupt(delay=5):
        """Simulate keyboard interruption by sending SIGINT after delay."""
        print(f"\n=== Simulating Keyboard Interrupt after {delay} seconds ===")
        print(f"Training will start and be interrupted after {delay} seconds...")
        
        # Schedule SIGINT after delay
        def send_interrupt():
            time.sleep(delay)
            print("\nSending interrupt signal...")
            os.kill(os.getpid(), signal.SIGINT)
        
        # Start timer in separate thread
        import threading
        interrupt_thread = threading.Thread(target=send_interrupt)
        interrupt_thread.daemon = True
        interrupt_thread.start()

    @staticmethod
    def simulate_transient_error():
        """Create a file that will cause a transient error when loaded."""
        print("\n=== Simulating Transient Error ===")
        
        # Create a mock model registry with invalid content
        registry_path = Path("model_zoo/registry.json")
        registry_path.parent.mkdir(exist_ok=True)
        
        # First write invalid JSON to cause an error
        with open(registry_path, 'w') as f:
            f.write("{invalid json")
        
        print(f"Created invalid registry file at {registry_path}")
        
        # Schedule fix after a delay to simulate transient error
        def fix_registry():
            time.sleep(2)
            print("\nFixing registry file to simulate recovery from transient error...")
            with open(registry_path, 'w') as f:
                f.write('{"faster_rcnn": {"name": "faster_rcnn", "model_type": "faster_rcnn", "backbone": "resnet50", "num_classes": 2, "input_size": 1024, "confidence_threshold": 0.5, "nms_threshold": 0.5, "pretrained": true}}')
        
        # Start timer in separate thread
        import threading
        fix_thread = threading.Thread(target=fix_registry)
        fix_thread.daemon = True
        fix_thread.start()

    @staticmethod
    def simulate_file_not_found():
        """Simulate file not found error by trying to load a non-existent file."""
        print("\n=== Simulating File Not Found Error ===")
        
        # Ensure the file doesn't exist
        registry_path = Path("model_zoo/nonexistent_registry.json")
        if registry_path.exists():
            registry_path.unlink()
        
        print(f"Attempting to load non-existent file: {registry_path}")
        return registry_path


def run_oom_scenario():
    """Run scenario to demonstrate handling of out-of-memory errors."""
    # Initialize pipeline with small batch size
    config_path = Path("config/pipeline_config.yaml")
    
    # Ensure config directory exists
    config_path.parent.mkdir(exist_ok=True)
    
    # Create minimal config if it doesn't exist
    if not config_path.exists():
        with open(config_path, 'w') as f:
            f.write("""
data:
  root_dir: data
  num_workers: 2
  img_size: 1024
  train_split: 0.8
training:
  batch_size: 16  # Start with large batch size to trigger OOM
  learning_rate: 0.001
  epochs: 10
  patience: 5
  save_frequency: 2
models:
  registry_path: model_zoo/registry.json
output:
  dir: output
eval:
  iou_threshold: 0.5
  pass_fail_config: config/pass_fail.yaml
            """)
    
    # Create registry if it doesn't exist
    registry_path = Path("model_zoo/registry.json")
    registry_path.parent.mkdir(exist_ok=True)
    if not registry_path.exists():
        with open(registry_path, 'w') as f:
            f.write('{"faster_rcnn": {"name": "faster_rcnn", "model_type": "faster_rcnn", "backbone": "resnet50", "num_classes": 2, "input_size": 1024, "confidence_threshold": 0.5, "nms_threshold": 0.5, "pretrained": true}}')
    
    # Initialize pipeline
    pipeline = ArmorPipeline(config_path)
    
    # Simulate OOM error during training
    try:
        # First trigger OOM error
        ErrorSimulator.simulate_oom_error()
        
        # Then try to train (should handle OOM gracefully)
        pipeline.train("faster_rcnn", epochs=1)
    except torch.cuda.OutOfMemoryError as e:
        print(f"\nCaught OOM error: {e}")
        print("Reducing batch size and retrying...")
        
        # Modify config to use smaller batch size
        with open(config_path, 'r') as f:
            config = f.read()
        
        config = config.replace("batch_size: 16", "batch_size: 2")
        
        with open(config_path, 'w') as f:
            f.write(config)
        
        # Reinitialize pipeline with new config
        pipeline = ArmorPipeline(config_path)
        
        # Try training again
        print("\nRetrying with smaller batch size...")
        pipeline.train("faster_rcnn", epochs=1)


def run_interrupt_scenario():
    """Run scenario to demonstrate handling of keyboard interruptions."""
    # Initialize pipeline
    config_path = Path("config/pipeline_config.yaml")
    
    # Ensure config exists
    if not config_path.exists():
        run_oom_scenario()  # This will create the config
    
    # Initialize pipeline
    pipeline = ArmorPipeline(config_path)
    
    # Set up interrupt simulation
    ErrorSimulator.simulate_keyboard_interrupt(delay=3)
    
    try:
        # Start training (will be interrupted)
        pipeline.train("faster_rcnn", epochs=10)
    except KeyboardInterrupt:
        print("\nTraining was interrupted as expected")
        print("Checking for checkpoint files...")
        
        # Check for checkpoint files
        checkpoint_dir = Path("output/checkpoints/faster_rcnn")
        if checkpoint_dir.exists():
            checkpoints = list(checkpoint_dir.glob("interrupted_*.pt"))
            if checkpoints:
                print(f"Found checkpoint files: {[cp.name for cp in checkpoints]}")
                print("\nYou can resume training from these checkpoints")
            else:
                print("No checkpoint files found")
        else:
            print("Checkpoint directory not found")


def run_transient_error_scenario():
    """Run scenario to demonstrate handling of transient errors."""
    # Initialize pipeline with non-existent registry
    config_path = Path("config/pipeline_config.yaml")
    
    # Ensure config exists
    if not config_path.exists():
        run_oom_scenario()  # This will create the config
    
    # Simulate transient error
    ErrorSimulator.simulate_transient_error()
    
    # Initialize pipeline
    pipeline = ArmorPipeline(config_path)
    
    try:
        # Start training (should retry after transient error)
        pipeline.train("faster_rcnn", epochs=1)
        print("\nTraining completed successfully after recovering from transient error")
    except Exception as e:
        print(f"\nFailed to recover from transient error: {e}")


def run_file_not_found_scenario():
    """Run scenario to demonstrate handling of file not found errors."""
    # Initialize pipeline with non-existent registry
    config_path = Path("config/pipeline_config.yaml")
    
    # Ensure config exists
    if not config_path.exists():
        run_oom_scenario()  # This will create the config
    
    # Modify config to use non-existent registry
    nonexistent_registry = ErrorSimulator.simulate_file_not_found()
    
    with open(config_path, 'r') as f:
        config = f.read()
    
    config = config.replace(
        "registry_path: model_zoo/registry.json",
        f"registry_path: {nonexistent_registry}"
    )
    
    with open(config_path, 'w') as f:
        f.write(config)
    
    # Initialize pipeline
    pipeline = ArmorPipeline(config_path)
    
    try:
        # Start training (should fail with file not found)
        pipeline.train("faster_rcnn", epochs=1)
    except FileNotFoundError as e:
        print(f"\nCaught FileNotFoundError as expected: {e}")
        print("This error is not transient and requires manual intervention")
        
        # Restore original config
        with open(config_path, 'r') as f:
            config = f.read()
        
        config = config.replace(
            f"registry_path: {nonexistent_registry}",
            "registry_path: model_zoo/registry.json"
        )
        
        with open(config_path, 'w') as f:
            f.write(config)
        
        print("\nRestored original config file")


def main():
    """Main function to run the test script."""
    parser = argparse.ArgumentParser(description="Test error handling in ArmorPipeline")
    parser.add_argument(
        "--scenario",
        choices=["oom", "interrupt", "transient", "file_not_found", "all"],
        default="all",
        help="Error scenario to test"
    )
    
    args = parser.parse_args()
    
    # Initialize logger
    logger = PerformanceLogger.get_instance()
    
    print("=== ArmorPipeline Error Handling Test ===")
    print("This script demonstrates the error handling capabilities of ArmorPipeline")
    print("See docs/error_handling.md for more information")
    
    if args.scenario == "oom" or args.scenario == "all":
        run_oom_scenario()
    
    if args.scenario == "interrupt" or args.scenario == "all":
        run_interrupt_scenario()
    
    if args.scenario == "transient" or args.scenario == "all":
        run_transient_error_scenario()
    
    if args.scenario == "file_not_found" or args.scenario == "all":
        run_file_not_found_scenario()
    
    print("\n=== Test Complete ===")
    print("Check the logs directory for detailed logs of the error handling")


if __name__ == "__main__":
    main()