# J&J Contact Lens Defect Detection Pipeline

## Project Overview

This pipeline implements an automated visual quality control system for Johnson & Johnson's contact lens manufacturing line. The system achieves **≥90% Average Precision (AP)** for defect detection using state-of-the-art deep learning models.

## Key Features

- **Plug-and-play architecture**: Easy model swapping via registry system
- **Multiple defect types**: Handles polylines (scratches), polygons (blobs), ellipses (bubbles), and bounding boxes
- **Production-ready**: Docker containerized for Nebius GPU deployment
- **Robust error handling**: Graceful recovery from CUDA memory errors with CPU fallback
- **Comprehensive evaluation**: AP metrics, confusion matrices, and Pass/Fail decisions
- **High accuracy**: Designed to meet J&J's ≥90% AP requirement
- **Checkpoint management**: Standardized directory structure for model checkpoints

## Project Structure

```
Project_Armor/
├── armor_pipeline/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── parser.py         # XML annotation parser
│   │   └── dataset.py        # PyTorch dataset and dataloaders
│   ├── models/
│   │   ├── __init__.py
│   │   └── registry.py       # Model registry and implementations
│   ├── eval/
│   │   ├── __init__.py
│   │   └── evaluator.py      # AP computation and Pass/Fail logic
│   └── utils/
│       └── __init__.py
├── config/
│   ├── pipeline_config.yaml  # Main configuration
│   └── pass_fail.yaml        # Pass/Fail thresholds per defect type
├── docs/
│   └── checkpoint_manager.md # Checkpoint management documentation
├── model_zoo/
│   └── registry.json         # Model configurations
├── docker/
│   └── Dockerfile           # Container definition
├── cli.py                   # Command-line interface
├── setup.py                 # Package setup
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Installation

### Quick Setup

For a quick and reliable setup, use the provided setup script:

```bash
# Make the script executable
chmod +x setup_environment.sh

# Run the setup script
./setup_environment.sh
```

This script will:
1. Check for Python 3.11
2. Create a virtual environment
3. Install PyTorch with appropriate CUDA support
4. Install all required dependencies
5. Verify the installation

For detailed instructions and troubleshooting, see the [Environment Setup Guide](docs/environment_setup.md).

### Manual Setup

If you prefer to set up the environment manually:

```bash
# Clone the repository
git clone <repository-url>
cd Project_Armor

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install PyTorch with CUDA support
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install -r requirements.txt
pip install scikit-multilearn>=0.2.0

# Install package in development mode
pip install -e .
```

### Docker Deployment

```bash
# Build Docker image
docker build -f docker/Dockerfile -t jnj-lens-defect:latest .

# Run container (with GPU support)
docker run --gpus all -v /path/to/data:/Project_Armor jnj-lens-defect:latest
```

## Usage

### 1. Training a Model

```bash
# Train YOLOv8 baseline model
python cli.py --config config/pipeline_config.yaml train --model yolov8m_baseline --epochs 100

# Train Faster R-CNN model
python cli.py --config config/pipeline_config.yaml train --model faster_rcnn_baseline --epochs 50
```

### 2. Evaluating Performance

```bash
# Evaluate trained model
python cli.py --config config/pipeline_config.yaml eval \
    --model yolov8m_baseline \
    --checkpoint outputs/yolov8m_baseline_best.pth
```

This generates:
- `evaluation_results/evaluation_summary_TIMESTAMP.yaml` - Overall metrics
- `evaluation_results/confusion_matrix_TIMESTAMP.png` - Visual confusion matrix
- `evaluation_results/confusion_matrix_TIMESTAMP.csv` - Numeric confusion matrix
- `evaluation_results/pass_fail_TIMESTAMP.csv` - Per-lens Pass/Fail decisions

### 3. Running Inference

```bash
# Run inference on single image
python cli.py --config config/pipeline_config.yaml infer \
    --model yolov8m_baseline \
    --image /Project_Armor/ProductionLineImages/A/sample.bmp \
    --checkpoint outputs/yolov8m_baseline_best.pth
```

### 4. Checkpoint Management

The pipeline uses a CheckpointManager to organize model checkpoints in a standardized directory structure:

```
checkpoints/
├── model1/
│   ├── best.pt
│   ├── epoch_1.pt
│   └── ...
└── model2/
    ├── best.pt
    └── ...
```

When training, evaluating, or running inference, if no checkpoint path is provided, the system automatically looks for the best checkpoint in the model-specific directory:

```bash
# Evaluate using the best checkpoint (automatically found)
python cli.py --config config/pipeline_config.yaml eval --model yolov8m_baseline

# Evaluate using a specific checkpoint
python cli.py --config config/pipeline_config.yaml eval \
    --model yolov8m_baseline \
    --checkpoint checkpoints/yolov8m_baseline/epoch_50.pt
```

For more details, see the [checkpoint management documentation](docs/checkpoint_manager.md).

## Model Registry

The system uses a plug-and-play model registry (`model_zoo/registry.json`):

```json
{
  "yolov8m_baseline": {
    "name": "yolov8m_baseline",
    "model_type": "yolov8",
    "backbone": "yolov8m",
    "num_classes": 7,
    "input_size": 1024,
    "confidence_threshold": 0.5,
    "nms_threshold": 0.5
  },
  "faster_rcnn_baseline": {
    "name": "faster_rcnn_baseline",
    "model_type": "faster_rcnn",
    "backbone": "resnet50_fpn",
    "num_classes": 7,
    "input_size": 1024,
    "confidence_threshold": 0.5,
    "nms_threshold": 0.5
  }
}
```

## Data Format

### Expected Directory Structure

```
/Project_Armor/
├── ProductionLineImages/
│   ├── A/
│   │   ├── image001.bmp
│   │   ├── image002.bmp
│   │   └── ...
│   ├── B/ ... P/
└── Annotations-XML/
    ├── TAM17-A/
    │   └── annotations.xml
    ├── TAM17-B/ ...
```

### XML Annotation Format

```xml
<annotation>
  <object>
    <name>edge_tear</name>
    <type>polyline</type>
    <polyline>
      <point x="100" y="200"/>
      <point x="150" y="250"/>
    </polyline>
  </object>
  <object>
    <name>bubble</name>
    <type>ellipse</type>
    <ellipse>
      <center x="500" y="600"/>
      <rx>50</rx>
      <ry>40</ry>
      <angle>30</angle>
    </ellipse>
  </object>
</annotation>
```

## Pass/Fail Logic

Each lens is evaluated based on defect severity:

```yaml
# config/pass_fail.yaml
pass_fail_thresholds:
  edge_tear:
    confidence: 0.6      # Minimum confidence to consider
    severity: 0.0005     # Max 0.05% of image area
  bubble:
    confidence: 0.7
    severity: 0.0008     # Max 0.08% of image area
```

A lens **FAILS** if ANY detected defect exceeds both confidence and severity thresholds.

## Performance Optimization Tips

1. **Data Augmentation**: The pipeline includes comprehensive augmentations (rotation, flip, brightness, blur)
2. **Multi-scale Training**: Use different input sizes (1024, 1280) for better small defect detection
3. **Class Balancing**: Monitor per-class AP and adjust sampling if needed
4. **Model Ensemble**: Combine predictions from multiple models for higher accuracy

## Achieving ≥90% AP

To reach the target performance:

1. **Start with YOLOv8-L**: Generally performs better than medium variants
2. **Increase Input Resolution**: Try 1280×1280 for small defects
3. **Tune Hyperparameters**:
   - Lower confidence threshold to 0.3-0.4
   - Adjust anchor sizes for your defect sizes
   - Use cosine learning rate schedule
4. **Data Quality**:
   - Ensure annotations are precise
   - Add hard negative mining
   - Balance defect type distribution

## Troubleshooting

### Common Issues

1. **Low AP on small defects**: Increase input resolution or use FPN-based models
2. **False positives on lens edges**: Add edge masks or adjust confidence thresholds
3. **Memory issues**: Reduce batch size or use gradient accumulation
4. **Slow training**: Enable AMP (Automatic Mixed Precision) for faster training

### Debugging Commands

```bash
# Test data loading
python -c "from armor_pipeline.data.dataset import DataModule; dm = DataModule('/Project_Armor'); dm.setup()"

# Visualize augmentations
python scripts/visualize_augmentations.py

# Check model registry
python -c "from armor_pipeline.models.registry import ModelRegistry; mr = ModelRegistry('model_zoo/registry.json'); print(mr.list_models())"
```

## Deployment to Production

1. **Export Final Model**:
   ```bash
   python cli.py eval --model yolov8l_final --checkpoint best_model.pth
   ```

2. **Build Production Image**:
   ```bash
   docker build -t nebius.cr.cloud/jnj-lens-qc:v1.0 .
   docker push nebius.cr.cloud/jnj-lens-qc:v1.0
   ```

3. **Deploy**:
   ```bash
   # Deploy on Nebius GPU instance
   docker run --gpus all \
     -v /mnt/data:/Project_Armor \
     -p 8080:8080 \
     nebius.cr.cloud/jnj-lens-qc:v1.0 \
     eval --model yolov8l_final
   ```

## Next Steps

1. **Experiment Tracking**: Integrate MLflow or Weights & Biases
2. **Active Learning**: Identify and label failure cases
3. **Model Optimization**: TensorRT conversion for inference speedup
4. **API Service**: Add REST API for real-time inference

## Contact

For questions or issues, please contact the development team.

---

**Remember**: The goal is ≥90% AP. Focus on model quality over inference speed for this evaluation phase.
