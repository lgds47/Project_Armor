# DeviceManager

## Overview

The `DeviceManager` class provides functionality to detect and manage hardware devices for optimal performance based on available resources. It helps the Project Armor pipeline adapt to different hardware configurations by automatically selecting the appropriate device (GPU or CPU) and recommending batch sizes based on available memory.

## Location

The `DeviceManager` class is located in:
```
armor_pipeline/utils/device_manager.py
```

## Key Features

1. **Automatic Device Selection**: Detects available hardware and selects the optimal device (CUDA GPU if available, otherwise CPU).
2. **Batch Size Recommendations**: Recommends appropriate batch sizes based on available GPU memory.
3. **Device Information**: Provides detailed information about available devices.
4. **Memory Statistics**: Tracks memory usage for CUDA devices.

## API Reference

### `get_optimal_device()`

Determines the optimal device and batch size based on available hardware.

**Returns:**
- `Tuple[torch.device, int]`: A tuple containing:
  - `torch.device`: The optimal device (CUDA or CPU)
  - `int`: Recommended batch size for the device

**Example:**
```python
from armor_pipeline.utils.device_manager import DeviceManager

device, batch_size = DeviceManager.get_optimal_device()
print(f"Using device: {device}")
print(f"Recommended batch size: {batch_size}")
```

### `get_ensemble_optimal_device(num_models=3)`

Determines the optimal device and per-model batch size for ensemble models, adjusting memory calculations for multiple simultaneous models.

**Parameters:**
- `num_models`: Number of models in the ensemble (default: 3)

**Returns:**
- `Tuple[torch.device, int]`: A tuple containing:
  - `torch.device`: The optimal device (CUDA or CPU)
  - `int`: Recommended batch size per model

**Example:**
```python
from armor_pipeline.utils.device_manager import DeviceManager

# For an ensemble with 5 models
device, batch_size = DeviceManager.get_ensemble_optimal_device(num_models=5)
print(f"Using device: {device}")
print(f"Recommended batch size per model: {batch_size}")
print(f"Total effective batch size: {batch_size * 5}")
```

### `get_device_info()`

Gets detailed information about available devices.

**Returns:**
- `dict`: Dictionary containing device information:
  - `cuda_available`: Boolean indicating if CUDA is available
  - `device_count`: Number of CUDA devices
  - `devices`: List of dictionaries with device details:
    - `name`: Device name
    - `total_memory_gb`: Total memory in GB
    - `compute_capability`: Compute capability version
    - `multi_processor_count`: Number of multiprocessors

**Example:**
```python
info = DeviceManager.get_device_info()
print(f"CUDA available: {info['cuda_available']}")
print(f"Device count: {info['device_count']}")
```

### `get_memory_stats()`

Gets current memory statistics for CUDA devices.

**Returns:**
- `dict`: Dictionary mapping device IDs to memory statistics:
  - `allocated`: Memory allocated in GB
  - `reserved`: Memory reserved in GB
  - `max_allocated`: Maximum allocated memory in GB

**Example:**
```python
stats = DeviceManager.get_memory_stats()
for device_id, device_stats in stats.items():
    print(f"Device {device_id} allocated memory: {device_stats['allocated']:.2f} GB")
```

## Integration with Project Armor

The `DeviceManager` has been integrated into the following components:

### 1. Model Classes

All model classes in `armor_pipeline/models/registry.py` now use `DeviceManager` for device selection:

- `BaseDetector.__init__()`: Base class for all detectors (uses `get_optimal_device()`)
- `ModelEnsemble.__init__()`: Ensemble of specialized models (uses `get_ensemble_optimal_device()` for optimized ensemble memory management)
- `EdgeAlgorithmRouter.__init__()`: Router for different data types (uses `get_optimal_device()`)

### 2. CLI Commands

All commands in `cli.py` now use `DeviceManager` for device selection and batch size adjustment:

- `train()`: Training command
- `evaluate()`: Evaluation command
- `infer()`: Inference command

## Batch Size Adjustment

### Single Model Batch Sizes

The `DeviceManager.get_optimal_device()` method recommends batch sizes based on available GPU memory:

- For GPUs with less than 16GB memory: Batch size of 4
- For GPUs with 16GB or more memory: Batch size of 8
- For CPU: Batch size of 2

### Ensemble Model Batch Sizes

The `DeviceManager.get_ensemble_optimal_device()` method recommends per-model batch sizes based on available GPU memory and the number of models in the ensemble:

- For high-end GPUs (>=32GB):
  - With 2-3 models: Batch size of 6 per model
  - With 4+ models: Dynamically reduced based on ensemble size
- For mid-range GPUs (16-32GB):
  - With 2-3 models: Batch size of 4 per model
  - With 4+ models: Dynamically reduced based on ensemble size
- For low-end GPUs (<16GB):
  - With 2-3 models: Batch size of 2 per model
  - With 4+ models: Batch size of 1 per model
- For CPU: Batch size of 1 per model

The method includes fallback logic for insufficient memory scenarios, automatically reducing batch sizes when necessary to prevent out-of-memory errors.

The CLI commands and model classes automatically adjust the batch size based on these recommendations, ensuring optimal performance on different hardware configurations.

## Testing

A test script is provided to verify the functionality of the `DeviceManager`:

```
scripts/test_device_manager.py
```

The test script includes tests for:
- Basic device detection and batch size recommendations
- Detailed device information retrieval
- Memory statistics monitoring
- Ensemble device selection with different ensemble sizes

Run this script to check if the `DeviceManager` correctly detects your hardware and recommends appropriate settings:

```bash
python scripts/test_device_manager.py
```

The test output will show recommendations for both single models and ensembles of different sizes, helping you verify that the memory management is working correctly for your hardware configuration.

## Benefits

1. **Hardware Adaptability**: The pipeline automatically adapts to different hardware configurations.
2. **Optimal Performance**: Batch sizes are adjusted based on available resources for optimal performance.
3. **Memory Management**: Helps prevent out-of-memory errors by recommending appropriate batch sizes.
4. **Simplified Code**: Centralizes device management logic in one place, making the codebase more maintainable.

## Future Enhancements

Potential future enhancements for the `DeviceManager`:

1. **Multi-GPU Support**: Add support for distributing workloads across multiple GPUs.
2. **Dynamic Batch Sizing**: Adjust batch sizes during training based on memory usage.
3. **Mixed Precision Training**: Add support for automatic mixed precision training.
4. **Memory Optimization**: Implement memory optimization techniques like gradient checkpointing.