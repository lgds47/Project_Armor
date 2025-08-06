# Environment Test Plan

This document outlines the steps to verify that the environment setup script works correctly and resolves all dependency issues.

## Prerequisites

- A clean environment (fresh virtual machine or container)
- Python 3.11 installed
- Git installed

## Test Procedure

### 1. Clone the Repository

```bash
git clone <repository-url>
cd Project_Armor
```

### 2. Run the Setup Script

```bash
chmod +x setup_environment.sh
./setup_environment.sh
```

### 3. Verify Basic Environment

After the script completes, verify that the environment is set up correctly:

```bash
# Activate the virtual environment if not already activated
source venv/bin/activate

# Check Python version
python --version  # Should show Python 3.11.x

# Check PyTorch installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

Expected output:
```
Python 3.11.x
PyTorch: 2.1.0
CUDA available: True  # If GPU is available, otherwise False
```

### 4. Verify All Dependencies

Run the verification commands to check that all required dependencies are installed:

```bash
# Check core dependencies
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
python -c "import albumentations; print(f'Albumentations: {albumentations.__version__}')"
python -c "import torchvision; print(f'Torchvision: {torchvision.__version__}')"
python -c "from ultralytics import YOLO; print('Ultralytics: OK')"
python -c "import skmultilearn; print('scikit-multilearn: OK')"

# Check project modules
python -c "import armor_pipeline; print('armor_pipeline: OK')"
```

### 5. Run Test Scripts

Run the test scripts that previously failed due to missing dependencies:

```bash
# Test performance logger
python Project_Armor/scripts/test_performance_logger.py

# Test bbox utils
python Project_Armor/scripts/test_bbox_utils.py

# Test device manager memory
python Project_Armor/scripts/test_device_manager_memory.py
```

Expected output:
- All tests should complete without any import errors
- The performance logger test should show successful logging and singleton pattern verification
- The bbox utils test should show successful validation of bounding boxes
- The device manager memory test should show successful memory calculations

### 6. Test Data Processing

Test the data processing components:

```bash
# Test dataset creation
python -c "from armor_pipeline.data.dataset import DataModule; print('DataModule imported successfully')"

# Test model registry
python -c "from armor_pipeline.models.registry import ModelRegistry; print('ModelRegistry imported successfully')"
```

### 7. Test CLI Functionality

Test the command-line interface:

```bash
# Show help
python cli.py --help
```

Expected output:
- The help message should be displayed without any errors

## Troubleshooting

If any of the verification steps fail:

1. Check the error message for specific dependency issues
2. Refer to the [Environment Setup Guide](environment_setup.md) for troubleshooting tips
3. Try reinstalling the specific dependency that's causing the issue
4. If the issue persists, try running the setup script again with the `--force` flag (if implemented)

## Test Results Documentation

Document the results of each test step:

| Test Step | Expected Result | Actual Result | Pass/Fail |
|-----------|-----------------|---------------|-----------|
| Python Version | Python 3.11.x | | |
| PyTorch Installation | PyTorch 2.1.0, CUDA status correct | | |
| NumPy | Version displayed | | |
| OpenCV | Version displayed | | |
| Albumentations | Version displayed | | |
| Torchvision | Version 0.16.0 | | |
| Ultralytics | "OK" | | |
| scikit-multilearn | "OK" | | |
| armor_pipeline | "OK" | | |
| test_performance_logger.py | All tests pass | | |
| test_bbox_utils.py | All tests pass | | |
| test_device_manager_memory.py | All tests pass | | |
| DataModule import | Success | | |
| ModelRegistry import | Success | | |
| CLI help | Help displayed | | |

## Conclusion

If all tests pass, the environment setup script is working correctly and has resolved all dependency issues. If any tests fail, refer to the troubleshooting section and the Environment Setup Guide for solutions.