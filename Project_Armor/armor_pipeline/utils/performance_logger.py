"""
Logging and Monitoring module for Project Armor.

This module provides functionality for logging and monitoring performance metrics.
"""

import logging
from datetime import datetime
from pathlib import Path
import threading
import os
import time
import json
from typing import Dict, Any, Optional, Union, List

class PerformanceLogger:
    """Logger for tracking performance metrics and general logging."""
    _lock = threading.Lock()
    
    def __init__(self, log_dir: Path = Path("logs")):
        """
        Initialize the performance logger.
        
        Args:
            log_dir: Path to the directory where logs will be stored
        """
        self.log_dir = log_dir
        self.log_dir.mkdir(exist_ok=True)
        
        # Setup file and console logging
        log_file = self.log_dir / f"armor_{datetime.now():%Y%m%d_%H%M%S}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger("armor_pipeline")
        self.logger.info(f"Initialized PerformanceLogger with log file: {log_file}")
        
        # Performance metrics tracking
        self.metrics = {
            "inference_times": [],
            "memory_usage": [],
            "model_accuracy": {}
        }
        
        # Start time for the session
        self.session_start_time = time.time()
    
    def get_logger(self, name: str = None) -> logging.Logger:
        """
        Get a logger instance.
        
        Args:
            name: Name for the logger (defaults to armor_pipeline)
            
        Returns:
            Logger instance
        """
        if name:
            return logging.getLogger(name)
        return self.logger
    
    def log_inference_time(self, model_name: str, inference_time_ms: float) -> None:
        """
        Log inference time for a model.
        
        Args:
            model_name: Name of the model
            inference_time_ms: Inference time in milliseconds
        """
        self.metrics["inference_times"].append({
            "model": model_name,
            "time_ms": inference_time_ms,
            "timestamp": time.time()
        })
        
        self.logger.info(f"Model {model_name} inference time: {inference_time_ms:.2f} ms")
    
    def log_memory_usage(self, model_name: str, allocated_mb: float, reserved_mb: float) -> None:
        """
        Log memory usage for a model.
        
        Args:
            model_name: Name of the model
            allocated_mb: Allocated memory in MB
            reserved_mb: Reserved memory in MB
        """
        self.metrics["memory_usage"].append({
            "model": model_name,
            "allocated_mb": allocated_mb,
            "reserved_mb": reserved_mb,
            "timestamp": time.time()
        })
        
        self.logger.info(f"Model {model_name} memory usage: {allocated_mb:.2f} MB allocated, {reserved_mb:.2f} MB reserved")
    
    def log_model_accuracy(self, model_name: str, metrics: Dict[str, float]) -> None:
        """
        Log model accuracy metrics.
        
        Args:
            model_name: Name of the model
            metrics: Dictionary of accuracy metrics (e.g., precision, recall, F1)
        """
        if model_name not in self.metrics["model_accuracy"]:
            self.metrics["model_accuracy"][model_name] = []
        
        self.metrics["model_accuracy"][model_name].append({
            "metrics": metrics,
            "timestamp": time.time()
        })
        
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"Model {model_name} accuracy metrics: {metrics_str}")
    
    def log_error(self, error_message: str, exception: Optional[Exception] = None) -> None:
        """
        Log an error.
        
        Args:
            error_message: Error message
            exception: Exception object (optional)
        """
        if exception:
            self.logger.error(f"{error_message}: {str(exception)}", exc_info=True)
        else:
            self.logger.error(error_message)
    
    def log_warning(self, warning_message: str) -> None:
        """
        Log a warning.
        
        Args:
            warning_message: Warning message
        """
        self.logger.warning(warning_message)
    
    def log_info(self, info_message: str) -> None:
        """
        Log an info message.
        
        Args:
            info_message: Info message
        """
        self.logger.info(info_message)
    
    def log_debug(self, debug_message: str) -> None:
        """
        Log a debug message.
        
        Args:
            debug_message: Debug message
        """
        self.logger.debug(debug_message)
    
    def save_metrics(self, output_path: Optional[Path] = None) -> Path:
        """
        Save performance metrics to a JSON file.
        
        Args:
            output_path: Path to save the metrics (optional)
            
        Returns:
            Path to the saved metrics file
        """
        if output_path is None:
            output_path = self.log_dir / f"metrics_{datetime.now():%Y%m%d_%H%M%S}.json"
        
        # Add session duration
        session_duration = time.time() - self.session_start_time
        metrics_with_summary = {
            "session_duration_seconds": session_duration,
            "session_start": datetime.fromtimestamp(self.session_start_time).isoformat(),
            "session_end": datetime.now().isoformat(),
            "metrics": self.metrics
        }
        
        with open(output_path, 'w') as f:
            json.dump(metrics_with_summary, f, indent=2)
        
        self.logger.info(f"Performance metrics saved to {output_path}")
        return output_path
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get a summary of performance metrics.
        
        Returns:
            Dictionary with performance summary
        """
        # Calculate average inference time
        inference_times = [entry["time_ms"] for entry in self.metrics["inference_times"]]
        avg_inference_time = sum(inference_times) / len(inference_times) if inference_times else 0
        
        # Calculate average memory usage
        allocated_memory = [entry["allocated_mb"] for entry in self.metrics["memory_usage"]]
        avg_allocated_memory = sum(allocated_memory) / len(allocated_memory) if allocated_memory else 0
        
        # Calculate session duration
        session_duration = time.time() - self.session_start_time
        
        summary = {
            "session_duration_seconds": session_duration,
            "average_inference_time_ms": avg_inference_time,
            "average_allocated_memory_mb": avg_allocated_memory,
            "total_inferences": len(inference_times),
            "models_evaluated": list(self.metrics["model_accuracy"].keys())
        }
        
        return summary
    
    @staticmethod
    def get_instance(log_dir: Optional[Path] = None) -> 'PerformanceLogger':
        """
        Get a singleton instance of PerformanceLogger.
        
        Args:
            log_dir: Path to the log directory (optional)
            
        Returns:
            PerformanceLogger instance
        """
        with PerformanceLogger._lock:
            if not hasattr(PerformanceLogger, "_instance") or PerformanceLogger._instance is None:
                PerformanceLogger._instance = PerformanceLogger(log_dir=log_dir or Path("logs"))
        return PerformanceLogger._instance