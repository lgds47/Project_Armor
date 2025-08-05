#!/bin/bash
# Quick start script for J&J Contact Lens Defect Detection Pipeline

echo "==========================================="
echo "J&J Contact Lens Defect Detection Pipeline"
echo "Quick Start Setup"
echo "==========================================="

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
REQUIRED_VERSION="3.11"

if [[ ! "$PYTHON_VERSION" == "$REQUIRED_VERSION"* ]]; then
    echo "❌ Error: Python $REQUIRED_VERSION required, but $PYTHON_VERSION found"
    echo "Please install Python 3.11 first"
    exit 1
fi

echo "✓ Python $PYTHON_VERSION detected"

# Create project structure
echo "Creating project structure..."
mkdir -p armor_pipeline/{data,models,eval,utils}
mkdir -p config
mkdir -p model_zoo
mkdir -p outputs/evaluation_results
mkdir -p docker
mkdir -p tests

# Create __init__.py files
touch armor_pipeline/__init__.py
touch armor_pipeline/data/__init__.py
touch armor_pipeline/models/__init__.py
touch armor_pipeline/eval/__init__.py
touch armor_pipeline/utils/__init__.py

# Create virtual environment
echo "Creating virtual environment..."
python3.11 -m venv venv
source venv/bin/activate

# Install PyTorch with CUDA support
echo "Installing PyTorch with CUDA support..."
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Install package in development mode
echo "Installing armor_pipeline package..."
pip install -e .

# Create default model registry
echo "Creating model registry..."
python -c "from armor_pipeline.models.registry import create_model_registry; create_model_registry(10)"

# Verify installation
echo ""
echo "Verifying installation..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
python -c "from ultralytics import YOLO; print('Ultralytics: OK')"

# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
    python -c "import torch; print(f'CUDA device: {torch.cuda.get_device_name(0)}')"
else
    echo "⚠️  Warning: CUDA not available. Training will be slow on CPU."
fi

echo ""
echo "==========================================="
echo "✅ Setup completed successfully!"
echo "==========================================="
echo ""
echo "Next steps:"
echo "1. Place your data in the expected structure:"
echo "   /Project_Armor/"
echo "   ├── ProductionLineImages/"
echo "   │   ├── A/ ... P/"
echo "   └── Annotations-XML/"
echo "       ├── TAM17-A/ ..."
echo ""
echo "2. Update config/pipeline_config.yaml with correct paths"
echo ""
echo "3. Train your first model:"
echo "   python cli.py train --model yolov8m_baseline --epochs 100"
echo ""
echo "4. Evaluate performance:"
echo "   python cli.py eval --model yolov8m_baseline --checkpoint outputs/yolov8m_baseline_best.pth"
echo ""
echo "For Docker deployment:"
echo "   docker build -f docker/Dockerfile -t jnj-lens-defect:latest ."
echo ""