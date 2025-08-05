"""
Main CLI for J&J Contact Lens Defect Detection Pipeline
Provides train/eval/infer commands with plug-and-play model support
"""

import argparse
import torch
from pathlib import Path
import yaml
import json
import time
from datetime import datetime
from tqdm import tqdm
import warnings
from typing import Optional, List, Dict, Any, Union, Tuple

warnings.filterwarnings('ignore')

from armor_pipeline.data.dataset import DataModule
from armor_pipeline.models.registry import ModelRegistry, create_model_registry, CheckpointManager
from armor_pipeline.eval.evaluator import DefectEvaluator
from armor_pipeline.eval.jj_compliant_evaluator import JJCompliantEvaluator


class ArmorPipeline:
    """
    Main pipeline orchestrator for training, evaluation, and inference.
    
    This class provides methods for training models, evaluating their performance,
    and running inference on single images. It includes comprehensive error handling
    to ensure robustness and reliability during training.
    
    For detailed information about the error handling features, see:
    docs/error_handling.md
    """

    def __init__(self, config_path: Path):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        # Import ProjectConfig
        from armor_pipeline.utils.project_config import get_project_config

        # Setup paths
        data_root = self.config['data'].get('root_dir')
        self.project_config = get_project_config(base_path=Path(data_root) if data_root else None)
        self.output_dir = Path(self.config['output']['dir'])
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Initialize components
        self.data_module = None
        self.model_registry = None
        self.evaluator = None
        
        # Initialize checkpoint manager
        self.checkpoint_manager = CheckpointManager(base_dir=self.output_dir / "checkpoints")
        
    def validate_pipeline_config(self) -> Dict[str, Any]:
        """
        Checks all aspects of the pipeline configuration for validity.
        
        This method:
        1. Checks all required config parameters exist
        2. Validates file paths and directories
        3. Verifies model registry entries
        4. Tests device availability and memory
        5. Returns detailed validation report with any issues found
        
        Returns:
            Dict containing validation results with the following structure:
            {
                "status": "success" or "failure",
                "categories": {
                    "config_params": {"status": "success" or "failure", "issues": [...]},
                    "file_paths": {"status": "success" or "failure", "issues": [...]},
                    "model_registry": {"status": "success" or "failure", "issues": [...]},
                    "device": {"status": "success" or "failure", "issues": [...]}
                },
                "issues": [
                    {"severity": "error" or "warning" or "info", 
                     "category": "category_name", 
                     "message": "issue description",
                     "recommendation": "how to fix"}
                ],
                "summary": "Overall validation summary"
            }
        """
        from armor_pipeline.utils.performance_logger import PerformanceLogger
        from armor_pipeline.utils.device_manager import DeviceManager
        from pathlib import Path
        import os
        
        # Get logger
        logger = PerformanceLogger.get_instance().get_logger()
        logger.info("Validating pipeline configuration...")
        
        # Initialize validation report
        report = {
            "status": "success",
            "categories": {
                "config_params": {"status": "success", "issues": []},
                "file_paths": {"status": "success", "issues": []},
                "model_registry": {"status": "success", "issues": []},
                "device": {"status": "success", "issues": []}
            },
            "issues": [],
            "summary": ""
        }
        
        # Helper function to add issues to the report
        def add_issue(severity, category, message, recommendation=""):
            issue = {
                "severity": severity,
                "category": category,
                "message": message,
                "recommendation": recommendation
            }
            report["issues"].append(issue)
            report["categories"][category]["issues"].append(issue)
            
            # Update category status if error
            if severity == "error":
                report["categories"][category]["status"] = "failure"
                report["status"] = "failure"
            
            # Log the issue
            if severity == "error":
                logger.error(f"{category}: {message}")
            elif severity == "warning":
                logger.warning(f"{category}: {message}")
            else:
                logger.info(f"{category}: {message}")
        
        # 1. Check all required config parameters exist
        logger.info("Checking configuration parameters...")
        
        # Required configuration sections
        required_sections = ["data", "training", "models", "output", "eval"]
        for section in required_sections:
            if section not in self.config:
                add_issue(
                    "error", 
                    "config_params", 
                    f"Missing required configuration section: {section}",
                    f"Add '{section}:' section to your configuration file"
                )
        
        # Check data configuration parameters
        if "data" in self.config:
            data_params = ["num_workers", "img_size", "train_split"]
            for param in data_params:
                if param not in self.config["data"]:
                    add_issue(
                        "error", 
                        "config_params", 
                        f"Missing required data parameter: {param}",
                        f"Add '{param}' to the 'data' section of your configuration file"
                    )
        
        # Check training configuration parameters
        if "training" in self.config:
            training_params = ["batch_size", "learning_rate", "epochs"]
            for param in training_params:
                if param not in self.config["training"]:
                    add_issue(
                        "error", 
                        "config_params", 
                        f"Missing required training parameter: {param}",
                        f"Add '{param}' to the 'training' section of your configuration file"
                    )
            
            # Check optional training parameters with defaults
            optional_params = {
                "patience": 10,
                "save_frequency": 5,
                "save_period": 5,
                "weight_decay": 0.0005,
                "lr_step_size": 10
            }
            for param, default in optional_params.items():
                if param not in self.config["training"]:
                    add_issue(
                        "info", 
                        "config_params", 
                        f"Optional training parameter not specified: {param}",
                        f"Default value of {default} will be used. Add '{param}' to customize."
                    )
        
        # Check models configuration parameters
        if "models" in self.config:
            if "registry_path" not in self.config["models"]:
                add_issue(
                    "error", 
                    "config_params", 
                    "Missing required models parameter: registry_path",
                    "Add 'registry_path' to the 'models' section of your configuration file"
                )
        
        # Check output configuration parameters
        if "output" in self.config:
            if "dir" not in self.config["output"]:
                add_issue(
                    "error", 
                    "config_params", 
                    "Missing required output parameter: dir",
                    "Add 'dir' to the 'output' section of your configuration file"
                )
        
        # Check eval configuration parameters
        if "eval" in self.config:
            eval_params = ["iou_threshold", "pass_fail_config"]
            for param in eval_params:
                if param not in self.config["eval"]:
                    add_issue(
                        "error", 
                        "config_params", 
                        f"Missing required eval parameter: {param}",
                        f"Add '{param}' to the 'eval' section of your configuration file"
                    )
        
        # 2. Validate file paths and directories
        logger.info("Validating file paths and directories...")
        
        # Check data root directory if specified
        if "data" in self.config and "root_dir" in self.config["data"]:
            data_root = Path(self.config["data"]["root_dir"])
            if not data_root.exists():
                add_issue(
                    "error", 
                    "file_paths", 
                    f"Data root directory not found: {data_root}",
                    "Create the directory or specify a valid path in the configuration"
                )
            elif not data_root.is_dir():
                add_issue(
                    "error", 
                    "file_paths", 
                    f"Data root path is not a directory: {data_root}",
                    "Specify a valid directory path in the configuration"
                )
        
        # Check output directory
        if "output" in self.config and "dir" in self.config["output"]:
            output_dir = Path(self.config["output"]["dir"])
            if not output_dir.exists():
                add_issue(
                    "warning", 
                    "file_paths", 
                    f"Output directory does not exist: {output_dir}",
                    "Directory will be created automatically, but ensure the parent directory exists"
                )
            elif not output_dir.is_dir():
                add_issue(
                    "error", 
                    "file_paths", 
                    f"Output path is not a directory: {output_dir}",
                    "Specify a valid directory path in the configuration"
                )
        
        # Check model registry path
        if "models" in self.config and "registry_path" in self.config["models"]:
            registry_path = Path(self.config["models"]["registry_path"])
            if not registry_path.exists():
                add_issue(
                    "warning", 
                    "file_paths", 
                    f"Model registry file not found: {registry_path}",
                    "A default registry will be created, but ensure the parent directory exists"
                )
                # Check if parent directory exists
                if not registry_path.parent.exists():
                    add_issue(
                        "error", 
                        "file_paths", 
                        f"Parent directory for model registry does not exist: {registry_path.parent}",
                        "Create the directory or specify a valid path in the configuration"
                    )
            elif registry_path.is_dir():
                add_issue(
                    "error", 
                    "file_paths", 
                    f"Model registry path is a directory, not a file: {registry_path}",
                    "Specify a valid file path in the configuration"
                )
        
        # Check pass_fail_config path
        if "eval" in self.config and "pass_fail_config" in self.config["eval"]:
            pass_fail_path = Path(self.config["eval"]["pass_fail_config"])
            if not pass_fail_path.exists():
                add_issue(
                    "error", 
                    "file_paths", 
                    f"Pass/fail configuration file not found: {pass_fail_path}",
                    "Create the file or specify a valid path in the configuration"
                )
            elif pass_fail_path.is_dir():
                add_issue(
                    "error", 
                    "file_paths", 
                    f"Pass/fail configuration path is a directory, not a file: {pass_fail_path}",
                    "Specify a valid file path in the configuration"
                )
        
        # 3. Verify model registry entries
        logger.info("Verifying model registry entries...")
        
        try:
            # Try to initialize model registry
            if "models" in self.config and "registry_path" in self.config["models"]:
                registry_path = Path(self.config["models"]["registry_path"])
                
                # If registry file exists, check its contents
                if registry_path.exists() and not registry_path.is_dir():
                    try:
                        # Check if file is valid JSON or YAML
                        if registry_path.suffix == '.json':
                            with open(registry_path) as f:
                                registry_data = json.load(f)
                        elif registry_path.suffix in ['.yaml', '.yml']:
                            with open(registry_path) as f:
                                registry_data = yaml.safe_load(f)
                        else:
                            add_issue(
                                "error", 
                                "model_registry", 
                                f"Unsupported registry format: {registry_path.suffix}",
                                "Use .json, .yaml, or .yml format for the registry file"
                            )
                            registry_data = {}
                        
                        # Check if registry contains any models
                        if not registry_data:
                            add_issue(
                                "warning", 
                                "model_registry", 
                                "Model registry is empty",
                                "Add model configurations to the registry file"
                            )
                        else:
                            # Check each model configuration
                            for model_name, config in registry_data.items():
                                # Check required fields
                                required_fields = ["model_type", "backbone", "num_classes", "input_size"]
                                for field in required_fields:
                                    if field not in config:
                                        add_issue(
                                            "error", 
                                            "model_registry", 
                                            f"Missing required field '{field}' in model configuration for '{model_name}'",
                                            f"Add '{field}' to the configuration for '{model_name}'"
                                        )
                                
                                # Check if model_type is supported
                                if "model_type" in config and config["model_type"] not in ModelRegistry.SUPPORTED_MODELS:
                                    add_issue(
                                        "error", 
                                        "model_registry", 
                                        f"Unsupported model type '{config['model_type']}' for model '{model_name}'",
                                        f"Use one of the supported model types: {', '.join(ModelRegistry.SUPPORTED_MODELS.keys())}"
                                    )
                    except json.JSONDecodeError:
                        add_issue(
                            "error", 
                            "model_registry", 
                            f"Invalid JSON format in model registry file: {registry_path}",
                            "Fix the JSON syntax in the registry file"
                        )
                    except yaml.YAMLError:
                        add_issue(
                            "error", 
                            "model_registry", 
                            f"Invalid YAML format in model registry file: {registry_path}",
                            "Fix the YAML syntax in the registry file"
                        )
                    except Exception as e:
                        add_issue(
                            "error", 
                            "model_registry", 
                            f"Error reading model registry file: {str(e)}",
                            "Check the file format and contents"
                        )
        except Exception as e:
            add_issue(
                "error", 
                "model_registry", 
                f"Error verifying model registry: {str(e)}",
                "Check the registry path and file format"
            )
        
        # 4. Test device availability and memory
        logger.info("Testing device availability and memory...")
        
        try:
            # Get device information
            device_info = DeviceManager.get_device_info()
            
            # Check CUDA availability
            if device_info["cuda_available"]:
                logger.info(f"CUDA is available with {device_info['device_count']} device(s)")
                
                # Check each device
                for device in device_info["devices"]:
                    logger.info(f"Found GPU: {device['name']} with {device['total_memory_gb']:.2f} GB memory")
                    
                    # Check if memory is sufficient
                    if device["total_memory_gb"] < 4:
                        add_issue(
                            "warning", 
                            "device", 
                            f"Low GPU memory: {device['total_memory_gb']:.2f} GB",
                            "Consider using a GPU with at least 4 GB memory for optimal performance"
                        )
                    
                    # Check compute capability
                    compute_capability = float(device["compute_capability"])
                    if compute_capability < 3.5:
                        add_issue(
                            "warning", 
                            "device", 
                            f"Low compute capability: {device['compute_capability']}",
                            "Consider using a GPU with compute capability 3.5 or higher for optimal performance"
                        )
                
                # Get current memory stats
                memory_stats = DeviceManager.get_memory_stats()
                for device_id, stats in memory_stats.items():
                    logger.info(f"GPU {device_id} memory: {stats['allocated']:.2f} GB allocated, {stats['reserved']:.2f} GB reserved")
                    
                    # Check if memory is already heavily used
                    if stats["allocated"] > 1.0:
                        add_issue(
                            "warning", 
                            "device", 
                            f"GPU {device_id} already has {stats['allocated']:.2f} GB memory allocated",
                            "Free up GPU memory before running the pipeline for optimal performance"
                        )
            else:
                add_issue(
                    "warning", 
                    "device", 
                    "CUDA is not available, will use CPU (slow)",
                    "Consider using a CUDA-compatible GPU for better performance"
                )
                
                # Check if batch size is too large for CPU
                if "training" in self.config and "batch_size" in self.config["training"]:
                    batch_size = self.config["training"]["batch_size"]
                    if batch_size > 4:
                        add_issue(
                            "warning", 
                            "device", 
                            f"Batch size {batch_size} may be too large for CPU training",
                            "Consider reducing batch size to 2-4 for CPU training"
                        )
        except Exception as e:
            add_issue(
                "warning", 
                "device", 
                f"Error testing device availability: {str(e)}",
                "This may indicate issues with your CUDA installation or GPU drivers"
            )
        
        # Generate summary
        error_count = sum(1 for issue in report["issues"] if issue["severity"] == "error")
        warning_count = sum(1 for issue in report["issues"] if issue["severity"] == "warning")
        info_count = sum(1 for issue in report["issues"] if issue["severity"] == "info")
        
        if error_count > 0:
            report["summary"] = f"Validation failed with {error_count} error(s), {warning_count} warning(s), and {info_count} info message(s)"
            logger.error(report["summary"])
        elif warning_count > 0:
            report["summary"] = f"Validation passed with {warning_count} warning(s) and {info_count} info message(s)"
            logger.warning(report["summary"])
        else:
            report["summary"] = f"Validation passed successfully with {info_count} info message(s)"
            logger.info(report["summary"])
        
        return report

    def setup_data(self):
        """
        Initialize data module with validation checks.
        
        This method:
        1. Verifies annotation files exist and are parseable
        2. Checks image-annotation correspondence
        3. Validates class mapping consistency
        4. Adds progress indicators for large dataset processing
        
        Raises:
            FileNotFoundError: If annotations directory doesn't exist or is empty
            ValueError: If class mapping is inconsistent
        """
        from armor_pipeline.utils.performance_logger import PerformanceLogger
        import os
        
        # Get logger
        logger = PerformanceLogger.get_instance().get_logger()
        logger.info("Setting up data module...")
        
        # Initialize data module
        self.data_module = DataModule(
            # Pass None for data_root to use the project_config from environment variables
            # The DataModule will create its own ProjectConfig instance
            batch_size=self.config['training']['batch_size'],
            num_workers=self.config['data']['num_workers'],
            img_size=self.config['data']['img_size'],
            train_split=self.config['data']['train_split']
        )
        
        # Verify annotations directory exists and contains XML files
        annotations_dir = self.data_module.annotations_dir
        if not annotations_dir.exists():
            error_msg = f"Annotations directory not found: {annotations_dir}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
            
        # Check if directory contains XML files
        xml_files = list(annotations_dir.glob("**/*.xml"))
        if not xml_files:
            error_msg = f"No XML annotation files found in {annotations_dir}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
            
        logger.info(f"Found {len(xml_files)} XML annotation files in {annotations_dir}")
        
        # Verify images directory exists
        images_dir = self.data_module.images_dir
        if not images_dir.exists():
            error_msg = f"Images directory not found: {images_dir}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
            
        # Check if directory contains image files
        image_files = []
        for ext in ['.bmp', '.jpg', '.jpeg', '.png']:
            image_files.extend(list(images_dir.glob(f"**/*{ext}")))
            image_files.extend(list(images_dir.glob(f"**/*{ext.upper()}")))
            
        if not image_files:
            error_msg = f"No image files found in {images_dir}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
            
        logger.info(f"Found {len(image_files)} image files in {images_dir}")
        
        # Setup data module with validation
        try:
            logger.info("Parsing annotations and creating dataloaders...")
            self.train_loader, self.val_loader = self.data_module.setup()
            
            # Log validation results
            total_annotations = len(self.data_module.all_annotations)
            valid_annotations = sum(len(loader.dataset.valid_annotations) for loader in [self.train_loader, self.val_loader])
            filtered_annotations = total_annotations - valid_annotations
            
            logger.info(f"Successfully parsed {total_annotations} annotations")
            logger.info(f"Found {valid_annotations} valid annotations with corresponding images")
            
            if filtered_annotations > 0:
                logger.warning(f"{filtered_annotations} annotations were filtered out due to missing images")
                
            # Validate class mapping
            class_mapping = self.data_module.class_mapping
            logger.info(f"Class mapping contains {len(class_mapping)} classes")
            
            # Log class distribution
            class_counts = {}
            for loader in [self.train_loader, self.val_loader]:
                for _, targets in loader:
                    for target in targets:
                        for label in target['labels']:
                            label_id = label.item()
                            if label_id > 0:  # Skip background class
                                class_counts[label_id] = class_counts.get(label_id, 0) + 1
            
            # Get class names
            class_names = self.data_module.get_class_names()
            
            # Log class distribution
            logger.info("Class distribution:")
            for class_id, count in sorted(class_counts.items()):
                if class_id < len(class_names):
                    class_name = class_names[class_id]
                    logger.info(f"  {class_name}: {count} instances")
                    
            # Check for classes with few samples
            min_samples_warning = 10
            classes_with_few_samples = [
                class_names[class_id] for class_id, count in class_counts.items()
                if count < min_samples_warning and class_id < len(class_names)
            ]
            
            if classes_with_few_samples:
                logger.warning(f"The following classes have fewer than {min_samples_warning} samples: {', '.join(classes_with_few_samples)}")
                
            # Check for classes with no samples
            classes_with_no_samples = [
                name for i, name in enumerate(class_names)
                if i > 0 and i not in class_counts
            ]
            
            if classes_with_no_samples:
                logger.warning(f"The following classes have no samples: {', '.join(classes_with_no_samples)}")
                
        except Exception as e:
            error_msg = f"Error setting up data module: {str(e)}"
            logger.error(error_msg)
            raise
            
        # Create/update model registry with correct number of classes
        num_classes = len(self.data_module.class_mapping) + 1
        registry_path = Path(self.config['models']['registry_path'])

        if not registry_path.exists():
            create_model_registry(num_classes)

        self.model_registry = ModelRegistry(registry_path)
        
        logger.info("Data module setup completed successfully")

    def train(self, model_name: str, epochs: int = None):
        """
        Train a model with comprehensive error handling.
        
        This method implements robust error handling to ensure training is as reliable
        and resilient as possible, even in the face of various errors and interruptions.
        
        Error handling features:
        - Try-catch blocks around model loading and training steps
        - Graceful degradation when device memory is insufficient (reduces batch size)
        - Checkpoint saving on training interruption (Ctrl+C, out of memory, etc.)
        - Better error messages with actionable recommendations
        - Automatic retry logic for transient failures
        - Memory monitoring throughout training
        
        The method will:
        1. Load the specified model from the registry
        2. Determine the optimal device (GPU or CPU) and batch size
        3. Train the model using either YOLOv8 or PyTorch native training
        4. Save checkpoints periodically and on interruption
        5. Log detailed information about training progress and errors
        
        For detailed information about error handling, see:
        docs/error_handling.md
        
        Args:
            model_name: Name of the model to train (must exist in the model registry)
            epochs: Number of epochs to train for (uses config value if None)
            
        Returns:
            None
            
        Raises:
            FileNotFoundError: If model registry or checkpoint files are not found
            ValueError: If model name is invalid or configuration is incorrect
            RuntimeError: If training fails due to non-transient errors
            torch.cuda.OutOfMemoryError: If GPU memory is exhausted and cannot be recovered
            KeyboardInterrupt: If training is interrupted by the user
            
        Examples:
            >>> pipeline = ArmorPipeline(config_path)
            >>> pipeline.train("faster_rcnn", epochs=10)
            
            # With error handling for out of memory
            >>> try:
            ...     pipeline.train("yolov8_large")
            ... except torch.cuda.OutOfMemoryError:
            ...     # Try a smaller model instead
            ...     pipeline.train("yolov8_small")
        """
        from armor_pipeline.utils.device_manager import DeviceManager
        from armor_pipeline.utils.performance_logger import PerformanceLogger
        from armor_pipeline.utils.retry_utils import is_transient_error, get_retry_recommendation
        import torch
        import traceback
        import signal
        import sys
        
        # Get logger
        logger = PerformanceLogger.get_instance().get_logger()
        
        # Initialize variables for checkpoint saving on interruption
        self.current_model = None
        self.current_epoch = 0
        self.training_interrupted = False
        
        # Setup signal handlers for graceful interruption
        original_sigint_handler = signal.getsignal(signal.SIGINT)
        original_sigterm_handler = signal.getsignal(signal.SIGTERM)
        
        def signal_handler(sig, frame):
            logger.warning("Training interrupted by user. Saving checkpoint...")
            self.training_interrupted = True
            # Restore original signal handlers
            signal.signal(signal.SIGINT, original_sigint_handler)
            signal.signal(signal.SIGTERM, original_sigterm_handler)
            
            # Save checkpoint if model is initialized
            if self.current_model is not None:
                try:
                    interrupted_checkpoint_path = self.checkpoint_manager.get_checkpoint_path(
                        model_name, epoch=f"interrupted_{self.current_epoch}"
                    )
                    self.current_model.save_checkpoint(interrupted_checkpoint_path)
                    logger.info(f"Saved interrupted checkpoint to {interrupted_checkpoint_path}")
                    print(f"\nTraining interrupted. Checkpoint saved to {interrupted_checkpoint_path}")
                except Exception as e:
                    logger.error(f"Failed to save checkpoint on interruption: {str(e)}")
                    print(f"\nTraining interrupted. Failed to save checkpoint: {str(e)}")
            
            # Exit gracefully
            sys.exit(0)
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        try:
            # Setup data if not already done
            if self.data_module is None:
                try:
                    logger.info("Setting up data module...")
                    self.setup_data()
                except Exception as e:
                    error_msg = f"Failed to set up data module: {str(e)}"
                    recommendation = get_retry_recommendation(e)
                    logger.error(f"{error_msg}\nRecommendation: {recommendation}")
                    print(f"\nError: {error_msg}\nRecommendation: {recommendation}")
                    raise

            # Get model with retry for transient errors
            try:
                logger.info(f"Loading model: {model_name}")
                model = self.model_registry.get_model(model_name)
                self.current_model = model
            except Exception as e:
                error_msg = f"Failed to load model {model_name}: {str(e)}"
                recommendation = get_retry_recommendation(e)
                logger.error(f"{error_msg}\nRecommendation: {recommendation}")
                print(f"\nError: {error_msg}\nRecommendation: {recommendation}")
                raise

            # Get optimal device with memory monitoring
            try:
                device, recommended_batch_size = DeviceManager.get_optimal_device()
                model = model.to(device)
                
                # Log initial memory stats
                if torch.cuda.is_available():
                    memory_stats = DeviceManager.get_memory_stats()
                    for device_id, stats in memory_stats.items():
                        logger.info(f"Initial GPU {device_id} memory: {stats['allocated']:.2f} GB allocated, "
                                   f"{stats['reserved']:.2f} GB reserved")
            except Exception as e:
                error_msg = f"Failed to initialize device: {str(e)}"
                recommendation = "Try using CPU if GPU is not available or has issues."
                logger.error(f"{error_msg}\nRecommendation: {recommendation}")
                print(f"\nError: {error_msg}\nRecommendation: {recommendation}")
                
                # Fallback to CPU
                logger.info("Falling back to CPU")
                device = torch.device('cpu')
                recommended_batch_size = 2
                model = model.to(device)

            # Set epochs
            epochs = epochs or self.config['training']['epochs']
            
            # Adjust batch size based on DeviceManager recommendation
            original_batch_size = self.config['training']['batch_size']
            adjusted_batch_size = min(original_batch_size, recommended_batch_size)
            if adjusted_batch_size != original_batch_size:
                logger.info(f"Adjusting batch size from {original_batch_size} to {adjusted_batch_size} based on available resources")
                print(f"Adjusting batch size from {original_batch_size} to {adjusted_batch_size} based on available resources")
                self.config['training']['batch_size'] = adjusted_batch_size

            logger.info(f"Training {model_name} for {epochs} epochs on {device} with batch size {adjusted_batch_size}")
            print(f"\nTraining {model_name} for {epochs} epochs...")
            print(f"Device: {device}")
            print(f"Batch size: {adjusted_batch_size}")
            print(f"Train samples: {len(self.train_loader.dataset)}")
            print(f"Val samples: {len(self.val_loader.dataset)}")

            # Model-specific training with error handling
            try:
                if model.config.model_type == 'yolov8':
                    # YOLOv8 requires data YAML
                    self._train_yolov8(model, epochs)
                else:
                    # PyTorch native training loop
                    self._train_pytorch_model(model, epochs)
            except torch.cuda.OutOfMemoryError as e:
                # Handle out of memory errors with graceful degradation
                error_msg = f"CUDA out of memory error: {str(e)}"
                recommendation = (
                    "Reduce batch size, use a smaller model, or free up GPU memory. "
                    "Trying to continue with a smaller batch size..."
                )
                logger.error(f"{error_msg}\nRecommendation: {recommendation}")
                print(f"\nError: {error_msg}\nRecommendation: {recommendation}")
                
                # Save checkpoint before reducing batch size
                interrupted_checkpoint_path = self.checkpoint_manager.get_checkpoint_path(
                    model_name, epoch=f"oom_{self.current_epoch}"
                )
                model.save_checkpoint(interrupted_checkpoint_path)
                logger.info(f"Saved checkpoint before reducing batch size to {interrupted_checkpoint_path}")
                
                # Reduce batch size and try again if possible
                if adjusted_batch_size > 1:
                    new_batch_size = max(1, adjusted_batch_size // 2)
                    logger.info(f"Reducing batch size from {adjusted_batch_size} to {new_batch_size} and retrying")
                    print(f"\nReducing batch size to {new_batch_size} and retrying...")
                    self.config['training']['batch_size'] = new_batch_size
                    
                    # Clear CUDA cache
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # Retry training with reduced batch size
                    if model.config.model_type == 'yolov8':
                        self._train_yolov8(model, epochs)
                    else:
                        self._train_pytorch_model(model, epochs)
                else:
                    logger.error("Batch size already at minimum. Cannot reduce further.")
                    raise
            except KeyboardInterrupt:
                # Handle keyboard interruption (Ctrl+C)
                logger.warning("Training interrupted by user")
                print("\nTraining interrupted by user")
                
                # Save checkpoint
                interrupted_checkpoint_path = self.checkpoint_manager.get_checkpoint_path(
                    model_name, epoch=f"interrupted_{self.current_epoch}"
                )
                model.save_checkpoint(interrupted_checkpoint_path)
                logger.info(f"Saved interrupted checkpoint to {interrupted_checkpoint_path}")
                print(f"Checkpoint saved to {interrupted_checkpoint_path}")
                raise
            except Exception as e:
                # Handle other exceptions
                error_msg = f"Error during training: {str(e)}"
                recommendation = get_retry_recommendation(e)
                logger.error(f"{error_msg}\nRecommendation: {recommendation}\n{traceback.format_exc()}")
                print(f"\nError: {error_msg}\nRecommendation: {recommendation}")
                
                # Save checkpoint if possible
                try:
                    if self.current_epoch > 0:
                        error_checkpoint_path = self.checkpoint_manager.get_checkpoint_path(
                            model_name, epoch=f"error_{self.current_epoch}"
                        )
                        model.save_checkpoint(error_checkpoint_path)
                        logger.info(f"Saved checkpoint before error to {error_checkpoint_path}")
                        print(f"Checkpoint saved to {error_checkpoint_path}")
                except Exception as save_error:
                    logger.error(f"Failed to save checkpoint after error: {str(save_error)}")
                
                # Retry if it's a transient error
                if is_transient_error(e):
                    logger.info("Detected transient error. Retrying...")
                    print("\nDetected transient error. Retrying...")
                    
                    # Clear CUDA cache if available
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # Retry training
                    if model.config.model_type == 'yolov8':
                        self._train_yolov8(model, epochs)
                    else:
                        self._train_pytorch_model(model, epochs)
                else:
                    raise
        finally:
            # Restore original signal handlers
            signal.signal(signal.SIGINT, original_sigint_handler)
            signal.signal(signal.SIGTERM, original_sigterm_handler)
            
            # Log final memory stats
            if torch.cuda.is_available():
                try:
                    memory_stats = DeviceManager.get_memory_stats()
                    for device_id, stats in memory_stats.items():
                        logger.info(f"Final GPU {device_id} memory: {stats['allocated']:.2f} GB allocated, "
                                   f"{stats['reserved']:.2f} GB reserved, "
                                   f"{stats['max_allocated']:.2f} GB max allocated")
                except Exception as e:
                    logger.error(f"Failed to get final memory stats: {str(e)}")
            
            # Save performance metrics
            try:
                PerformanceLogger.get_instance().save_metrics(self.output_dir / "performance_metrics.json")
            except Exception as e:
                logger.error(f"Failed to save performance metrics: {str(e)}")

    def _train_yolov8(self, model, epochs):
        """
        Train YOLOv8 model with enhanced error handling.
        
        This method includes:
        - Try-catch blocks for error handling
        - Memory monitoring during training
        - Checkpoint saving on interruption
        - Tracking of current epoch
        
        Args:
            model: The YOLOv8 model to train
            epochs: Number of epochs to train for
            
        Returns:
            None
        """
        from armor_pipeline.utils.device_manager import DeviceManager
        from armor_pipeline.utils.performance_logger import PerformanceLogger
        from armor_pipeline.utils.retry_utils import get_retry_recommendation
        
        # Get logger
        logger = PerformanceLogger.get_instance().get_logger()
        
        # Track start time for performance monitoring
        start_time = time.time()
        
        try:
            # Monitor memory before training
            if torch.cuda.is_available():
                memory_stats = DeviceManager.get_memory_stats()
                for device_id, stats in memory_stats.items():
                    logger.info(f"GPU {device_id} memory before YOLOv8 training: "
                               f"{stats['allocated']:.2f} GB allocated, "
                               f"{stats['reserved']:.2f} GB reserved")
            
            # Create data YAML for YOLOv8
            try:
                logger.info("Creating YOLO data YAML")
                data_yaml = self._create_yolo_data_yaml()
            except Exception as e:
                error_msg = f"Failed to create YOLO data YAML: {str(e)}"
                recommendation = get_retry_recommendation(e)
                logger.error(f"{error_msg}\nRecommendation: {recommendation}")
                print(f"\nError: {error_msg}\nRecommendation: {recommendation}")
                raise
            
            # Set up run name with timestamp
            run_name = f"{model.config.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            logger.info(f"Starting YOLOv8 training with run name: {run_name}")
            
            # Set up save period for checkpoints (default to every 5 epochs)
            save_period = self.config['training'].get('save_period', 5)
            
            # Set up callbacks to track progress and handle interruptions
            def on_train_epoch_end(trainer):
                """Callback for end of each training epoch"""
                # Update current epoch for checkpoint saving on interruption
                self.current_epoch = trainer.epoch + 1
                
                # Log progress
                logger.info(f"YOLOv8 training epoch {trainer.epoch + 1}/{epochs} completed")
                
                # Check if training was interrupted
                if self.training_interrupted:
                    logger.warning("YOLOv8 training interrupted by user")
                    trainer.stop = True  # Signal YOLOv8 to stop training
                
                # Monitor memory
                if torch.cuda.is_available():
                    memory_stats = DeviceManager.get_memory_stats()
                    for device_id, stats in memory_stats.items():
                        logger.info(f"GPU {device_id} memory after epoch {trainer.epoch + 1}: "
                                   f"{stats['allocated']:.2f} GB allocated, "
                                   f"{stats['reserved']:.2f} GB reserved")
            
            # Train with error handling
            try:
                # Start training
                results = model.train(
                    data_yaml,
                    epochs=epochs,
                    batch=self.config['training']['batch_size'],
                    imgsz=self.config['data']['img_size'],
                    patience=self.config['training'].get('patience', 10),
                    save_period=save_period,
                    project=str(self.output_dir / 'yolo_runs'),
                    name=run_name,
                    callbacks={"on_train_epoch_end": on_train_epoch_end}
                )
                
                # Log results
                logger.info(f"YOLOv8 training completed. Best model saved to: {results}")
                print(f"\nTraining completed. Best model saved to: {results}")
                
                # Calculate total training time
                total_time = time.time() - start_time
                hours, remainder = divmod(total_time, 3600)
                minutes, seconds = divmod(remainder, 60)
                
                logger.info(f"YOLOv8 training completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
                print(f"Training completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
                
            except torch.cuda.OutOfMemoryError as e:
                # Handle out of memory errors
                error_msg = f"CUDA out of memory during YOLOv8 training: {str(e)}"
                recommendation = (
                    "Reduce batch size, use a smaller model, or free up GPU memory. "
                    "YOLOv8 training requires significant GPU memory."
                )
                logger.error(f"{error_msg}\nRecommendation: {recommendation}")
                print(f"\nError: {error_msg}\nRecommendation: {recommendation}")
                
                # Clear cache and raise to be handled by the train() method
                torch.cuda.empty_cache()
                raise
                
            except KeyboardInterrupt:
                # Handle keyboard interruption (Ctrl+C)
                logger.warning("YOLOv8 training interrupted by user")
                print("\nYOLOv8 training interrupted by user")
                
                # YOLOv8 should save a checkpoint automatically, but log it
                logger.info("YOLOv8 should have saved a checkpoint automatically")
                print("YOLOv8 should have saved a checkpoint automatically")
                raise
                
            except Exception as e:
                # Handle other exceptions
                error_msg = f"Error during YOLOv8 training: {str(e)}"
                recommendation = get_retry_recommendation(e)
                logger.error(f"{error_msg}\nRecommendation: {recommendation}")
                print(f"\nError: {error_msg}\nRecommendation: {recommendation}")
                raise
                
        finally:
            # Log final memory stats
            if torch.cuda.is_available():
                try:
                    memory_stats = DeviceManager.get_memory_stats()
                    for device_id, stats in memory_stats.items():
                        logger.info(f"Final GPU {device_id} memory after YOLOv8 training: "
                                   f"{stats['allocated']:.2f} GB allocated, "
                                   f"{stats['reserved']:.2f} GB reserved, "
                                   f"{stats['max_allocated']:.2f} GB max allocated")
                except Exception as e:
                    logger.error(f"Failed to get final memory stats: {str(e)}")
            
            # Log performance metrics
            try:
                PerformanceLogger.get_instance().log_model_accuracy(model.config.name, {
                    'training_completed': not self.training_interrupted,
                    'epochs_completed': self.current_epoch,
                    'total_epochs': epochs
                })
            except Exception as e:
                logger.error(f"Failed to log performance metrics: {str(e)}")

    def _train_pytorch_model(self, model, epochs):
        """
        Train PyTorch-based models (Faster R-CNN, etc.) with enhanced error handling.
        
        This method includes:
        - Tracking of current epoch for checkpoint saving on interruption
        - Memory monitoring during training
        - Periodic checkpoint saving
        - Enhanced error handling for training steps
        
        Args:
            model: The model to train
            epochs: Number of epochs to train for
            
        Returns:
            None
        """
        from armor_pipeline.utils.device_manager import DeviceManager
        from armor_pipeline.utils.performance_logger import PerformanceLogger
        
        # Get logger
        logger = PerformanceLogger.get_instance().get_logger()
        
        # Setup optimizer
        optimizer = torch.optim.SGD(
            model.model.parameters(),
            lr=self.config['training']['learning_rate'],
            momentum=0.9,
            weight_decay=self.config['training'].get('weight_decay', 0.0005)
        )

        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.config['training'].get('lr_step_size', 10),
            gamma=0.1
        )

        best_map = 0
        patience_counter = 0
        patience = self.config['training'].get('patience', 10)
        
        # Get checkpoint saving frequency (default to every 5 epochs)
        save_frequency = self.config['training'].get('save_frequency', 5)
        
        # Track start time for performance monitoring
        start_time = time.time()

        for epoch in range(epochs):
            # Update current epoch for checkpoint saving on interruption
            self.current_epoch = epoch + 1
            
            logger.info(f"Starting epoch {epoch + 1}/{epochs}")
            print(f"\nEpoch {epoch + 1}/{epochs}")

            # Training phase
            model.model.train()
            train_losses = []
            
            # Monitor memory before training
            if torch.cuda.is_available():
                memory_stats = DeviceManager.get_memory_stats()
                for device_id, stats in memory_stats.items():
                    logger.info(f"GPU {device_id} memory before epoch {epoch + 1}: "
                               f"{stats['allocated']:.2f} GB allocated, "
                               f"{stats['reserved']:.2f} GB reserved")

            # Training loop with enhanced error handling
            pbar = tqdm(self.train_loader, desc="Training")
            batch_idx = 0
            for images, targets in pbar:
                # Check if training was interrupted
                if self.training_interrupted:
                    logger.warning("Training loop interrupted")
                    break
                
                # Monitor memory periodically (every 10 batches)
                if batch_idx % 10 == 0 and torch.cuda.is_available():
                    memory_stats = DeviceManager.get_memory_stats()
                    for device_id, stats in memory_stats.items():
                        logger.debug(f"GPU {device_id} memory at batch {batch_idx}: "
                                    f"{stats['allocated']:.2f} GB allocated")
                
                try:
                    # Move data to device
                    images = images.to(model.device)
                    targets = [{k: v.to(model.device) for k, v in t.items() if k != 'image_path'}
                              for t in targets]

                    # Zero gradients
                    optimizer.zero_grad()

                    # Forward and backward pass
                    losses = model.train_step(images, targets)
                    train_losses.append(losses['loss_total'])

                    # Update weights
                    optimizer.step()

                    # Update progress bar
                    pbar.set_postfix({'loss': f"{losses['loss_total']:.4f}"})
                    
                except torch.cuda.OutOfMemoryError as e:
                    # Handle out of memory errors
                    logger.error(f"CUDA out of memory at epoch {epoch + 1}, batch {batch_idx}: {str(e)}")
                    
                    # Clear cache and raise to be handled by the train() method
                    torch.cuda.empty_cache()
                    raise
                    
                except Exception as e:
                    # Handle other exceptions in training step
                    logger.error(f"Error in training step at epoch {epoch + 1}, batch {batch_idx}: {str(e)}")
                    print(f"Error in training step: {e}")
                    
                    # Skip this batch and continue
                    continue
                
                batch_idx += 1

            # Skip validation if training was interrupted
            if self.training_interrupted:
                logger.warning("Skipping validation due to interruption")
                break
                
            # Validation phase
            try:
                val_map = self._validate(model)
                
                # Learning rate scheduling
                lr_scheduler.step()

                # Logging
                avg_train_loss = sum(train_losses) / max(len(train_losses), 1)
                logger.info(f"Epoch {epoch + 1} - Train Loss: {avg_train_loss:.4f}, Val mAP: {val_map:.4f}")
                print(f"Train Loss: {avg_train_loss:.4f}")
                print(f"Val mAP: {val_map:.4f}")

                # Save best model
                if val_map > best_map:
                    best_map = val_map
                    patience_counter = 0

                    checkpoint_path = self.checkpoint_manager.get_checkpoint_path(model.config.name)
                    model.save_checkpoint(checkpoint_path)
                    logger.info(f"Saved best model with mAP: {best_map:.4f} to {checkpoint_path}")
                    print(f"Saved best model with mAP: {best_map:.4f} to {checkpoint_path}")
                else:
                    patience_counter += 1
                
                # Periodic checkpoint saving
                if (epoch + 1) % save_frequency == 0:
                    periodic_checkpoint_path = self.checkpoint_manager.get_checkpoint_path(
                        model.config.name, epoch=epoch + 1
                    )
                    model.save_checkpoint(periodic_checkpoint_path)
                    logger.info(f"Saved periodic checkpoint at epoch {epoch + 1} to {periodic_checkpoint_path}")
                
                # Log performance metrics
                PerformanceLogger.get_instance().log_model_accuracy(model.config.name, {
                    'epoch': epoch + 1,
                    'train_loss': avg_train_loss,
                    'val_map': val_map,
                    'best_map': best_map,
                    'patience_counter': patience_counter
                })

                # Early stopping
                if patience_counter >= patience:
                    logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                    print(f"Early stopping triggered after {epoch + 1} epochs")
                    break
                    
            except Exception as e:
                # Handle validation errors
                logger.error(f"Error during validation at epoch {epoch + 1}: {str(e)}")
                print(f"Error during validation: {e}")
                
                # Save checkpoint before potentially exiting
                error_checkpoint_path = self.checkpoint_manager.get_checkpoint_path(
                    model.config.name, epoch=f"error_{epoch + 1}"
                )
                model.save_checkpoint(error_checkpoint_path)
                logger.info(f"Saved checkpoint after validation error to {error_checkpoint_path}")
                
                # Re-raise the exception to be handled by the train() method
                raise

        # Calculate total training time
        total_time = time.time() - start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        logger.info(f"Training completed in {int(hours)}h {int(minutes)}m {int(seconds)}s. Best mAP: {best_map:.4f}")
        print(f"\nTraining completed in {int(hours)}h {int(minutes)}m {int(seconds)}s. Best mAP: {best_map:.4f}")

    def _validate(self, model) -> float:
        """Validate model and return mAP"""
        model.model.eval()

        evaluator = DefectEvaluator(
            class_names=self.data_module.get_class_names(),
            iou_threshold=0.5
        )

        with torch.no_grad():
            for images, targets in tqdm(self.val_loader, desc="Validation"):
                images = images.to(model.device)

                predictions = model.predict(images)
                image_paths = [t['image_path'] for t in targets]

                evaluator.update(predictions, targets, image_paths)

        return evaluator.compute_map()

    def evaluate(self, model_name: str, checkpoint_path: Optional[Path] = None):
        """Run full evaluation on a model"""
        from armor_pipeline.utils.device_manager import DeviceManager
        
        if self.data_module is None:
            self.setup_data()

        # Get model
        model = self.model_registry.get_model(model_name)

        # Load checkpoint if provided, otherwise use best checkpoint from CheckpointManager
        if checkpoint_path:
            model.load_checkpoint(checkpoint_path)
        else:
            # Try to load the best checkpoint for this model
            best_checkpoint_path = self.checkpoint_manager.get_checkpoint_path(model_name)
            if best_checkpoint_path.exists():
                print(f"Loading best checkpoint from {best_checkpoint_path}")
                model.load_checkpoint(best_checkpoint_path)
            else:
                print(f"No checkpoint found for {model_name}, using initial weights")

        device, recommended_batch_size = DeviceManager.get_optimal_device()
        model = model.to(device)
        
        # Adjust batch size based on DeviceManager recommendation
        original_batch_size = self.config['training']['batch_size']
        adjusted_batch_size = min(original_batch_size, recommended_batch_size)
        if adjusted_batch_size != original_batch_size:
            print(f"Adjusting batch size from {original_batch_size} to {adjusted_batch_size} based on available resources")
            self.config['training']['batch_size'] = adjusted_batch_size

        print(f"\nEvaluating {model_name}...")
        print(f"Device: {device}")
        print(f"Batch size: {adjusted_batch_size}")

        # Setup evaluator
        self.evaluator = DefectEvaluator(
            class_names=self.data_module.get_class_names(),
            iou_threshold=self.config['eval']['iou_threshold'],
            pass_fail_config_path=Path(self.config['eval']['pass_fail_config'])
        )

        # Run evaluation
        model.model.eval()

        with torch.no_grad():
            for images, targets in tqdm(self.val_loader, desc="Evaluating"):
                images = images.to(device)

                predictions = model.predict(images)
                image_paths = [t['image_path'] for t in targets]

                self.evaluator.update(predictions, targets, image_paths)

        # Generate comprehensive report
        eval_output_dir = self.output_dir / 'evaluation_results'
        report = self.evaluator.generate_full_report(eval_output_dir)

        return report

    def infer(self, model_name: str, image_path: Path, checkpoint_path: Optional[Path] = None):
        """Run inference on a single image"""
        from armor_pipeline.utils.device_manager import DeviceManager
        
        # Get model
        model = self.model_registry.get_model(model_name)

        # Load checkpoint if provided, otherwise use best checkpoint from CheckpointManager
        if checkpoint_path:
            model.load_checkpoint(checkpoint_path)
        else:
            # Try to load the best checkpoint for this model
            best_checkpoint_path = self.checkpoint_manager.get_checkpoint_path(model_name)
            if best_checkpoint_path.exists():
                print(f"Loading best checkpoint from {best_checkpoint_path}")
                model.load_checkpoint(best_checkpoint_path)
            else:
                print(f"No checkpoint found for {model_name}, using initial weights")

        device, _ = DeviceManager.get_optimal_device()
        model = model.to(device)
        model.model.eval()

        # Load and preprocess image
        import cv2
        from armor_pipeline.data.dataset import get_transform

        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image[:2048, :]  # Crop to upper half

        # Apply transform
        transform = get_transform("val", self.config['data']['img_size'])
        transformed = transform(image=image)
        image_tensor = transformed['image'].unsqueeze(0).to(device)

        # Run inference
        with torch.no_grad():
            predictions = model.predict(image_tensor)

        # Visualize results
        self._visualize_predictions(image, predictions[0], image_path)

        return predictions[0]

    def _visualize_predictions(self, image, predictions, image_path):
        """Visualize detection results"""
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        ax.imshow(image)

        # Draw predictions
        boxes = predictions['boxes'].cpu().numpy()
        scores = predictions['scores'].cpu().numpy()
        labels = predictions['labels'].cpu().numpy()

        class_names = self.data_module.get_class_names() if self.data_module else None

        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1

            # Create rectangle
            rect = patches.Rectangle(
                (x1, y1), w, h,
                linewidth=2,
                edgecolor='red',
                facecolor='none'
            )
            ax.add_patch(rect)

            # Add label
            label_text = f"{class_names[int(label)] if class_names else label}: {score:.2f}"
            ax.text(
                x1, y1 - 5, label_text,
                color='white',
                backgroundcolor='red',
                fontsize=10
            )

        ax.set_title(f"Predictions for {image_path.name}")
        ax.axis('off')

        # Save
        output_path = self.output_dir / f"inference_{image_path.stem}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Visualization saved to: {output_path}")

    def _create_yolo_data_yaml(self) -> str:
        """Create YAML file for YOLOv8 training"""
        # This is a simplified version - in production, you'd need to
        # convert annotations to YOLO format and organize files appropriately

        data_config = {
            'path': str(self.data_root),
            'train': 'train/images',
            'val': 'val/images',
            'nc': len(self.data_module.class_mapping),
            'names': list(self.data_module.class_mapping.keys())
        }

        yaml_path = self.output_dir / 'yolo_data.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(data_config, f)

        return str(yaml_path)


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="J&J Contact Lens Defect Detection Pipeline"
    )

    parser.add_argument(
        '--config',
        type=Path,
        default='config/pipeline_config.yaml',
        help='Path to pipeline configuration file'
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train a model')
    train_parser.add_argument('--model', required=True, help='Model name from registry')
    train_parser.add_argument('--epochs', type=int, help='Number of epochs')

    # Evaluate command
    eval_parser = subparsers.add_parser('eval', help='Evaluate a model')
    eval_parser.add_argument('--model', required=True, help='Model name from registry')
    eval_parser.add_argument('--checkpoint', type=Path, help='Path to model checkpoint')

    # Infer command
    infer_parser = subparsers.add_parser('infer', help='Run inference on single image')
    infer_parser.add_argument('--model', required=True, help='Model name from registry')
    infer_parser.add_argument('--image', type=Path, required=True, help='Path to image')
    infer_parser.add_argument('--checkpoint', type=Path, help='Path to model checkpoint')

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = ArmorPipeline(args.config)

    # Execute command
    if args.command == 'train':
        pipeline.train(args.model, args.epochs)

    elif args.command == 'eval':
        report = pipeline.evaluate(args.model, args.checkpoint)

    elif args.command == 'infer':
        predictions = pipeline.infer(args.model, args.image, args.checkpoint)
        print(f"\nDetected {len(predictions['boxes'])} defects")

    else:
        parser.print_help()


if __name__ == '__main__':
    main()