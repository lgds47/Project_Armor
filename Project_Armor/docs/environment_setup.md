# Environment Setup Guide

This guide provides instructions for setting up the Python environment for the J&J Contact Lens Defect Detection Pipeline.

## Requirements

- Python 3.11
- CUDA-compatible GPU (optional, but recommended for training)
- Git

## Quick Setup

For a quick setup, run the provided script:

```bash
chmod +x setup_environment.sh
./setup_environment.sh
```

This script will:
1. Check for Python 3.11
2. Create a virtual environment
3. Install PyTorch with appropriate CUDA support
4. Install all required dependencies
5. Verify the installation

## Manual Setup

If you prefer to set up the environment manually, follow these steps:

1. **Create a virtual environment**:
   ```bash
   python3.11 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install PyTorch and torchvision**:
   - With CUDA support:
     ```bash
     pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
     ```
   - CPU only:
     ```bash
     pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cpu
     ```

3. **Install other dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install scikit-multilearn>=0.2.0
   ```

4. **Install the package in development mode**:
   ```bash
   pip install -e .
   ```

## Verifying the Installation

To verify that the environment is set up correctly, run:

```bash
# Check PyTorch and CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Run test scripts
python Project_Armor/scripts/test_performance_logger.py
python Project_Armor/scripts/test_bbox_utils.py
python Project_Armor/scripts/test_device_manager_memory.py
```

## Troubleshooting

### Python Version Issues

**Error**: Python 3.11 not found or not the default version.

**Solution**:
- Ubuntu/Debian:
  ```bash
  sudo apt install python3.11 python3.11-venv python3.11-dev
  ```
- macOS:
  ```bash
  brew install python@3.11
  ```
- Windows:
  Download and install from [python.org](https://www.python.org/downloads/)

### Virtual Environment Issues

**Error**: Failed to create virtual environment.

**Solution**:
- Ubuntu/Debian:
  ```bash
  sudo apt install python3.11-venv
  ```
- Make sure you have write permissions to the current directory

### PyTorch Installation Issues

**Error**: Failed to install PyTorch with CUDA support.

**Solution**:
1. Check if CUDA is installed:
   ```bash
   nvidia-smi
   ```
2. If CUDA is not installed, follow the [NVIDIA CUDA installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
3. If CUDA is installed but PyTorch installation fails, try installing a different version of PyTorch compatible with your CUDA version:
   ```bash
   # For CUDA 11.8
   pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
   
   # For CUDA 11.7
   pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu117
   ```

### Albumentations Installation Issues

**Error**: Failed to import albumentations.

**Solution**:
1. Try reinstalling with specific version:
   ```bash
   pip uninstall -y albumentations
   pip install albumentations==1.3.1
   ```
2. Check for conflicting dependencies:
   ```bash
   pip check
   ```

### Import Errors

**Error**: ModuleNotFoundError when importing project modules.

**Solution**:
1. Make sure the package is installed in development mode:
   ```bash
   pip install -e .
   ```
2. Check if the Python path includes the project directory:
   ```bash
   python -c "import sys; print(sys.path)"
   ```
3. If not, add it to PYTHONPATH:
   ```bash
   export PYTHONPATH=$PYTHONPATH:/path/to/Project_Armor
   ```

### CUDA Out of Memory Errors

**Error**: CUDA out of memory during training.

**Solution**:
1. Reduce batch size in the configuration
2. Use a smaller model
3. Try mixed precision training (already implemented in the code)

## Docker Environment

If you prefer to use Docker, you can build and run the Docker image:

```bash
# Build the Docker image
docker build -f docker/Dockerfile -t jnj-lens-defect:latest .

# Run the Docker container
docker run --gpus all -it jnj-lens-defect:latest
```

## Common Issues and Solutions

### Test Scripts Failing

If test scripts are failing with import errors:

1. Make sure all dependencies are installed:
   ```bash
   pip install -r requirements.txt
   pip install scikit-multilearn>=0.2.0
   ```

2. Check if the package is installed in development mode:
   ```bash
   pip install -e .
   ```

3. Try running the tests with the full path:
   ```bash
   python -m Project_Armor.scripts.test_performance_logger
   ```

### Performance Issues

If training is slow even with a GPU:

1. Verify CUDA is being used:
   ```bash
   python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())"
   ```

2. Check GPU utilization during training:
   ```bash
   nvidia-smi -l 1
   ```

3. Make sure the model is on the GPU:
   ```python
   print(next(model.parameters()).device)  # Should show 'cuda:0'
   ```

## Additional Resources

- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
- [CUDA Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
- [Python Virtual Environments Guide](https://docs.python.org/3/tutorial/venv.html)