"""
Retry Utilities for Project Armor.

This module provides utilities for implementing retry logic for transient failures
in the ArmorPipeline training process. It includes functions for:

1. Retrying operations with exponential backoff when transient errors occur
2. Detecting whether an exception represents a transient error
3. Generating actionable recommendations for handling specific exceptions

These utilities are used by the ArmorPipeline class to implement robust error
handling during model training. For more information, see:
docs/error_handling.md

Example usage:

```python
from armor_pipeline.utils.retry_utils import retry_with_backoff, is_transient_error

# Decorate a function to retry on transient errors
@retry_with_backoff(max_retries=3)
def load_data():
    # Function that might fail transiently
    pass

# Check if an error is transient
try:
    some_operation()
except Exception as e:
    if is_transient_error(e):
        # Retry the operation
        pass
    else:
        # Handle permanent error
        raise
```
"""

import time
import logging
import functools
import random
from typing import Type, Callable, Any, List, Optional, Union, Tuple

logger = logging.getLogger(__name__)

def retry_with_backoff(
    max_retries: int = 3,
    initial_backoff: float = 1.0,
    backoff_factor: float = 2.0,
    max_backoff: float = 60.0,
    jitter: bool = True,
    retryable_exceptions: List[Type[Exception]] = None
) -> Callable:
    """
    Decorator that retries a function with exponential backoff when specified exceptions occur.
    
    Args:
        max_retries: Maximum number of retries before giving up
        initial_backoff: Initial backoff time in seconds
        backoff_factor: Factor by which the backoff time increases after each failure
        max_backoff: Maximum backoff time in seconds
        jitter: Whether to add random jitter to backoff time
        retryable_exceptions: List of exception types that should trigger a retry
                             (defaults to [Exception] if None)
    
    Returns:
        Decorated function with retry logic
    """
    if retryable_exceptions is None:
        retryable_exceptions = [Exception]
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            retry_count = 0
            backoff = initial_backoff
            
            while True:
                try:
                    return func(*args, **kwargs)
                except tuple(retryable_exceptions) as e:
                    retry_count += 1
                    
                    if retry_count > max_retries:
                        logger.error(f"Maximum retries ({max_retries}) exceeded. Last error: {str(e)}")
                        raise
                    
                    # Calculate backoff time with optional jitter
                    if jitter:
                        # Add random jitter between 0% and 25%
                        jitter_amount = random.uniform(0, 0.25 * backoff)
                        current_backoff = backoff + jitter_amount
                    else:
                        current_backoff = backoff
                    
                    # Cap at max_backoff
                    current_backoff = min(current_backoff, max_backoff)
                    
                    logger.warning(
                        f"Retry {retry_count}/{max_retries} after error: {str(e)}. "
                        f"Backing off for {current_backoff:.2f} seconds."
                    )
                    
                    time.sleep(current_backoff)
                    
                    # Increase backoff for next retry
                    backoff *= backoff_factor
                    
                except Exception as e:
                    # Non-retryable exception
                    logger.error(f"Non-retryable error occurred: {str(e)}")
                    raise
        
        return wrapper
    
    return decorator

def is_transient_error(exception: Exception) -> bool:
    """
    Determine if an exception represents a transient error that can be retried.
    
    Args:
        exception: The exception to check
        
    Returns:
        True if the exception is likely transient, False otherwise
    """
    # List of error messages that indicate transient failures
    transient_patterns = [
        "CUDA out of memory",
        "CUDA error: out of memory",
        "RuntimeError: CUDA error",
        "Connection reset by peer",
        "Connection refused",
        "Connection timed out",
        "Temporary failure in name resolution",
        "Resource temporarily unavailable",
        "Too many open files",
        "Cannot allocate memory",
        "Disk quota exceeded",
        "Network is unreachable",
        "Operation timed out",
        "Resource temporarily unavailable",
        "Temporary failure in name resolution",
        "The service is unavailable",
        "Internal Server Error",
        "Service Unavailable",
        "Gateway Time-out",
        "Bad Gateway",
        "Gateway Timeout",
        "Request Timeout",
        "Timeout",
        "timeout",
        "timed out",
        "ConnectionError",
        "ConnectionResetError",
        "ConnectionRefusedError",
        "ConnectionAbortedError",
        "BrokenPipeError",
        "OSError: [Errno 12]",  # Cannot allocate memory
        "OSError: [Errno 24]",  # Too many open files
        "OSError: [Errno 28]",  # No space left on device
    ]
    
    # Check if the exception message contains any of the transient patterns
    error_message = str(exception)
    for pattern in transient_patterns:
        if pattern.lower() in error_message.lower():
            return True
    
    # Check exception types that are typically transient
    transient_types = [
        "TimeoutError",
        "ConnectionError",
        "ConnectionResetError",
        "ConnectionRefusedError",
        "ConnectionAbortedError",
        "BrokenPipeError",
        "InterruptedError",
    ]
    
    exception_type = type(exception).__name__
    if exception_type in transient_types:
        return True
    
    return False

def get_retry_recommendation(exception: Exception) -> str:
    """
    Get a recommendation for handling a specific exception.
    
    Args:
        exception: The exception to get a recommendation for
        
    Returns:
        A string with a recommendation for handling the exception
    """
    error_message = str(exception)
    
    # Memory-related errors
    if "CUDA out of memory" in error_message or "out of memory" in error_message:
        return (
            "Memory error detected. Try reducing batch size, using a smaller model, "
            "or freeing up GPU memory by closing other applications."
        )
    
    # File-related errors
    if "No such file or directory" in error_message or "FileNotFoundError" in error_message:
        return (
            "File not found. Check that the specified file exists and the path is correct."
        )
    
    # Permission errors
    if "Permission denied" in error_message or "PermissionError" in error_message:
        return (
            "Permission error. Check that you have the necessary permissions to access the file or directory."
        )
    
    # Network-related errors
    if any(pattern in error_message for pattern in ["Connection", "Timeout", "Network"]):
        return (
            "Network error detected. Check your internet connection and try again later."
        )
    
    # Disk space errors
    if "No space left on device" in error_message or "Disk quota exceeded" in error_message:
        return (
            "Disk space error. Free up disk space or use a different location with more available space."
        )
    
    # Default recommendation
    return (
        "An error occurred. Check the logs for more details and try again."
    )