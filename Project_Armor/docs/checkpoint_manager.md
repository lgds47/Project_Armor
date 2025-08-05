# CheckpointManager

## Overview

The `CheckpointManager` class provides a standardized way to manage model checkpoint paths and directories in the Project Armor pipeline. It ensures a consistent directory structure for model checkpoints, making it easier to save, load, and organize model weights.

## Location

The `CheckpointManager` class is located in:
```
armor_pipeline/models/registry.py
```

## Key Features

1. **Standardized Directory Structure**: Creates a consistent directory structure for model checkpoints.
2. **Automatic Directory Creation**: Automatically creates necessary directories if they don't exist.
3. **Model-Specific Directories**: Organizes checkpoints by model name.
4. **Epoch-Specific Checkpoints**: Supports both best model checkpoints and epoch-specific checkpoints.

## API Reference

### `__init__(base_dir: Path = Path("checkpoints"))`

Initializes the checkpoint manager with a base directory.

**Parameters:**
- `base_dir`: Base directory for checkpoints. Defaults to "checkpoints" in the current directory.

**Example:**
```python
from armor_pipeline.models.registry import CheckpointManager
from pathlib import Path

# Create a checkpoint manager with the default base directory
checkpoint_manager = CheckpointManager()

# Create a checkpoint manager with a custom base directory
checkpoint_manager = CheckpointManager(base_dir=Path("./output/checkpoints"))
```

### `get_checkpoint_path(model_name: str, epoch: int = None)`

Gets the path for a model checkpoint.

**Parameters:**
- `model_name`: Name of the model.
- `epoch`: Optional epoch number. If provided, returns path for that epoch. If not provided, returns path for the best model.

**Returns:**
- `Path`: Path object for the checkpoint.

**Example:**
```python
# Get the path for the best checkpoint of a model
best_path = checkpoint_manager.get_checkpoint_path("yolov8m_edge")

# Get the path for a specific epoch checkpoint
epoch_path = checkpoint_manager.get_checkpoint_path("yolov8m_edge", epoch=50)
```

## Integration with Project Armor

The `CheckpointManager` has been integrated into the following components:

### 1. ArmorPipeline Class

The `ArmorPipeline` class in `cli.py` now uses `CheckpointManager` for managing checkpoint paths:

```python
def __init__(self, config_path: Path):
    # ...
    self.checkpoint_manager = CheckpointManager(base_dir=self.output_dir / "checkpoints")
```

### 2. Training

The `_train_pytorch_model` method in `cli.py` now uses `CheckpointManager` for saving the best model checkpoint:

```python
checkpoint_path = self.checkpoint_manager.get_checkpoint_path(model.config.name)
model.save_checkpoint(checkpoint_path)
```

### 3. Evaluation and Inference

The `evaluate` and `infer` methods in `cli.py` now use `CheckpointManager` for loading checkpoints if no specific checkpoint path is provided:

```python
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
```

## Directory Structure

The `CheckpointManager` creates the following directory structure:

```
checkpoints/
├── model1/
│   ├── best.pt
│   ├── epoch_1.pt
│   ├── epoch_2.pt
│   └── ...
├── model2/
│   ├── best.pt
│   ├── epoch_1.pt
│   └── ...
└── ...
```

## Benefits

1. **Consistency**: Ensures a consistent directory structure for all model checkpoints.
2. **Organization**: Keeps checkpoints organized by model name and epoch.
3. **Automation**: Automatically creates necessary directories.
4. **Simplicity**: Provides a simple API for getting checkpoint paths.
5. **Maintainability**: Makes it easier to maintain and update checkpoint handling code.

## Testing

A test script is provided to verify the functionality of the `CheckpointManager`:

```
scripts/test_checkpoint_manager.py
```

Run this script to check if the `CheckpointManager` correctly manages checkpoint paths and directories:

```bash
python scripts/test_checkpoint_manager.py
```