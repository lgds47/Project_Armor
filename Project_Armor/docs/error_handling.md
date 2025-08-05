# Error Handling in ArmorPipeline

This document describes the comprehensive error handling implemented in the ArmorPipeline training process.

## Overview

The ArmorPipeline training process includes robust error handling mechanisms to ensure that training is as reliable and resilient as possible. The error handling features include:

1. **Try-catch blocks around model loading and training steps**
2. **Graceful degradation when device memory is insufficient**
3. **Checkpoint saving on training interruption**
4. **Better error messages with actionable recommendations**
5. **Automatic retry logic for transient failures**

## Features in Detail

### Try-catch Blocks

The training process is wrapped in try-catch blocks at multiple levels:

- Data setup
- Model loading
- Device initialization
- Training loop
- Validation

Each level has specific error handling tailored to the types of errors that might occur at that stage.

### Graceful Degradation for Memory Issues

When CUDA out-of-memory errors occur, the system:

1. Logs detailed error information
2. Saves a checkpoint of the current model state
3. Reduces the batch size automatically
4. Clears CUDA cache
5. Retries the training with the reduced batch size

If the GPU is not available or has issues, the system falls back to CPU training with an appropriate batch size.

### Checkpoint Saving on Interruption

The training process saves checkpoints in several scenarios:

- Periodically during training (configurable frequency)
- When the best validation performance is achieved
- When training is interrupted by the user (Ctrl+C)
- When out-of-memory errors occur
- When other errors occur during training

Checkpoints are saved with descriptive names that include the epoch number and the reason for saving (e.g., "interrupted_5", "oom_3", "error_7").

### Better Error Messages

Error messages include:

- Detailed description of the error
- Context information (epoch, batch, etc.)
- Actionable recommendations for resolving the issue
- Memory usage statistics for memory-related errors

### Automatic Retry for Transient Failures

The system can automatically retry training when transient errors occur:

- Network-related errors
- Temporary resource unavailability
- Some types of CUDA errors
- Connection issues

The retry mechanism uses exponential backoff to avoid overwhelming the system.

## Memory Monitoring

Memory usage is monitored throughout the training process:

- Before training starts
- At the beginning of each epoch
- Periodically during training (every 10 batches)
- After each epoch
- At the end of training

This information is logged and can be used to diagnose memory-related issues.

## Configuration Options

The following configuration options affect error handling:

- `training.batch_size`: Initial batch size (will be reduced if memory issues occur)
- `training.save_frequency`: How often to save periodic checkpoints (default: 5 epochs)
- `training.patience`: Number of epochs with no improvement before early stopping

## Example Error Scenarios

### Out of Memory Error

```
Error: CUDA out of memory error: CUDA out of memory. Tried to allocate 2.00 GiB
Recommendation: Reduce batch size, use a smaller model, or free up GPU memory. Trying to continue with a smaller batch size...

Reducing batch size to 4 and retrying...
```

### File Not Found Error

```
Error: Failed to load model faster_rcnn: FileNotFoundError: [Errno 2] No such file or directory: 'model_zoo/registry.json'
Recommendation: File not found. Check that the specified file exists and the path is correct.
```

### Training Interruption

```
Training interrupted by user
Checkpoint saved to checkpoints/faster_rcnn/interrupted_7.pt
```

## Logging

All errors, warnings, and important events are logged using the PerformanceLogger:

- To the console for immediate feedback
- To log files for later analysis
- To performance metrics files for tracking

## Best Practices

1. **Monitor Memory Usage**: Check the logs for memory usage information to identify potential memory issues before they cause training failures.

2. **Use Appropriate Batch Size**: Start with a conservative batch size to avoid out-of-memory errors.

3. **Save Checkpoints Frequently**: Configure `training.save_frequency` to save checkpoints frequently enough to minimize data loss in case of interruptions.

4. **Check Error Recommendations**: When errors occur, read the recommendations provided in the error messages for guidance on how to resolve the issue.

5. **Resume from Checkpoints**: After an interruption or error, you can resume training from the saved checkpoint.

## Implementation Details

The error handling is implemented in the following files:

- `cli.py`: ArmorPipeline.train(), _train_pytorch_model(), and _train_yolov8() methods
- `armor_pipeline/utils/retry_utils.py`: Utilities for retry logic and error recommendations
- `armor_pipeline/utils/device_manager.py`: Memory monitoring and device selection
- `armor_pipeline/utils/performance_logger.py`: Logging and performance tracking