#!/bin/bash
# Environment setup script for J&J Contact Lens Defect Detection Pipeline
# This script sets up the Python environment with all required dependencies

echo "==========================================="
echo "J&J Contact Lens Defect Detection Pipeline"
echo "Environment Setup"
echo "==========================================="

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
REQUIRED_VERSION="3.11"

if [[ ! "$PYTHON_VERSION" == "$REQUIRED_VERSION"* ]]; then
    echo "❌ Error: Python $REQUIRED_VERSION required, but $PYTHON_VERSION found"
    echo "Please install Python 3.11 first"
    echo "On Ubuntu: sudo apt install python3.11 python3.11-venv python3.11-dev"
    echo "On macOS: brew install python@3.11"
    exit 1
fi

echo "✓ Python $PYTHON_VERSION detected"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3.11 -m venv venv
    if [ $? -ne 0 ]; then
        echo "❌ Error: Failed to create virtual environment"
        echo "Make sure python3.11-venv is installed"
        exit 1
    fi
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
if [ $? -ne 0 ]; then
    echo "❌ Error: Failed to activate virtual environment"
    exit 1
fi

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip
if [ $? -ne 0 ]; then
    echo "❌ Error: Failed to upgrade pip"
    exit 1
fi

# Check CUDA availability
echo "Checking CUDA availability..."
CUDA_AVAILABLE=false
if command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA GPU detected"
    CUDA_AVAILABLE=true
else
    echo "⚠️ Warning: NVIDIA GPU not detected, will install CPU-only PyTorch"
fi

# Install PyTorch with appropriate CUDA support
echo "Installing PyTorch and torchvision..."
if [ "$CUDA_AVAILABLE" = true ]; then
    # Install PyTorch with CUDA support
    pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
    if [ $? -ne 0 ]; then
        echo "⚠️ Warning: Failed to install PyTorch with CUDA support"
        echo "Falling back to CPU-only PyTorch"
        pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cpu
    fi
else
    # Install CPU-only PyTorch
    pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cpu
fi

# Install other dependencies
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "❌ Error: Failed to install dependencies from requirements.txt"
    echo "Check the error message above for details"
    exit 1
fi

# Install scikit-multilearn if not already installed
echo "Ensuring scikit-multilearn is installed..."
pip install scikit-multilearn>=0.2.0
if [ $? -ne 0 ]; then
    echo "❌ Error: Failed to install scikit-multilearn"
    exit 1
fi

# Install package in development mode
echo "Installing armor_pipeline package in development mode..."
pip install -e .
if [ $? -ne 0 ]; then
    echo "❌ Error: Failed to install package in development mode"
    echo "Check the error message above for details"
    exit 1
fi

# Verify installation
echo ""
echo "Verifying installation..."

# Check PyTorch
echo -n "PyTorch: "
python -c "import torch; print(f'{torch.__version__}')" || { echo "❌ Failed to import torch"; exit 1; }

# Check torchvision
echo -n "torchvision: "
python -c "import torchvision; print(f'{torchvision.__version__}')" || { echo "❌ Failed to import torchvision"; exit 1; }

# Check OpenCV
echo -n "OpenCV: "
python -c "import cv2; print(f'{cv2.__version__}')" || { echo "❌ Failed to import cv2"; exit 1; }

# Check numpy
echo -n "NumPy: "
python -c "import numpy; print(f'{numpy.__version__}')" || { echo "❌ Failed to import numpy"; exit 1; }

# Check albumentations
echo -n "albumentations: "
python -c "import albumentations; print(f'{albumentations.__version__}')" || { echo "❌ Failed to import albumentations"; exit 1; }

# Check scikit-multilearn
echo -n "scikit-multilearn: "
python -c "import skmultilearn; print('OK')" || { echo "❌ Failed to import skmultilearn"; exit 1; }

# Check ultralytics
echo -n "ultralytics: "
python -c "from ultralytics import YOLO; print('OK')" || { echo "❌ Failed to import ultralytics"; exit 1; }

# Check CUDA availability
echo -n "CUDA available: "
python -c "import torch; print(f'{torch.cuda.is_available()}')"
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
    echo -n "CUDA device: "
    python -c "import torch; print(f'{torch.cuda.get_device_name(0)}')"
else
    echo "⚠️ Warning: CUDA not available. Training will be slow on CPU."
fi

# Check if armor_pipeline can be imported
echo -n "armor_pipeline: "
python -c "import armor_pipeline; print('OK')" || { echo "❌ Failed to import armor_pipeline"; exit 1; }

echo ""
echo "==========================================="
echo "✅ Environment setup completed successfully!"
echo "==========================================="
echo ""
echo "You can now run the test scripts to verify the environment:"
echo "python Project_Armor/scripts/test_performance_logger.py"
echo "python Project_Armor/scripts/test_bbox_utils.py"
echo "python Project_Armor/scripts/test_device_manager_memory.py"
echo ""
echo "If you encounter any issues, check the error messages and refer to the troubleshooting guide."
echo ""