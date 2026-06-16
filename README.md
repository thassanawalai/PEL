# Ultra-Fine-Grained Visual Classification (UFGVC)

A professional, modular PyTorch framework for training deep learning models to classify 165 highly similar leaf cultivars. This project will eventually integrate Prototype-enhanced Learning (PEL) for improved fine-grained classification accuracy.

## Project Overview

**Phase 1 (Current)**: Build a robust, scalable PyTorch foundation with:
- Centralized configuration management
- Modular dataset handling with PyTorch DataLoaders
- Baseline ResNet50 classifier with custom head
- Professional training loop with logging and checkpointing
- Early stopping and learning rate scheduling

**Phase 2 (Future)**: Integration of Prototype-enhanced Learning (PEL) for enhanced classification.

## Directory Structure

```
PEL/
├── data/                   # Dataset storage
│   ├── train/             # Training images (organized by class)
│   ├── val/               # Validation images (organized by class)
│   └── test/              # Test images (organized by class)
├── models/                # Model architectures
│   ├── __init__.py
│   └── backbone.py        # ResNet50 classifier with custom head
├── configs/               # Configuration management
│   ├── __init__.py
│   └── config.py          # Centralized hyperparameters
├── utils/                 # Utility modules
│   ├── __init__.py
│   ├── transforms.py      # Data augmentation pipelines
│   └── logger.py          # Logging utilities
├── scripts/               # Executable scripts
│   ├── dataset.py         # Dataset and DataLoader creation
│   ├── train.py           # Main training loop
│   └── evaluate.py        # Evaluation script (optional)
├── checkpoints/           # Saved model weights
├── logs/                  # Training logs and metrics
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Installation

### Prerequisites
- Python 3.8+
- CUDA 11.8+ (for GPU training, optional but recommended)

### Setup

1. **Clone or navigate to the project directory**:
   ```bash
   cd PEL
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   ```

3. **Activate virtual environment**:
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset Preparation

The project expects datasets to be organized using the ImageFolder structure:

```
data/
├── train/
│   ├── class_001/
│   │   ├── image_001.jpg
│   │   ├── image_002.jpg
│   │   └── ...
│   ├── class_002/
│   │   └── ...
│   └── ... (165 classes total)
├── val/
│   ├── class_001/
│   │   └── ...
│   └── ... (165 classes)
└── test/
    ├── class_001/
    │   └── ...
    └── ... (165 classes)
```

**Note**: Each class folder should contain JPEG/PNG images of leaf cultivars.

## Configuration

All hyperparameters and paths are managed in [configs/config.py](configs/config.py). Key parameters:

```python
NUM_CLASSES = 165              # Number of leaf cultivars
BATCH_SIZE = 32                # Training batch size
NUM_EPOCHS = 100               # Number of training epochs
LEARNING_RATE = 0.001          # Adam optimizer learning rate
INPUT_SIZE = 224               # Image input size
MODEL_NAME = "resnet50"        # Backbone architecture
PRETRAINED = True              # Use ImageNet pretrained weights
```

Modify these values before training as needed.

## Training

To start training:

```bash
python scripts/train.py
```

**Training features**:
- ✅ Automatic mixed precision (AMP) ready
- ✅ Learning rate scheduling (StepLR)
- ✅ Early stopping based on validation accuracy
- ✅ Checkpoint saving at regular intervals + best model
- ✅ Comprehensive logging to file and console
- ✅ Progress bars with tqdm
- ✅ GPU/CPU device management

**Example training output**:
```
============================================================
Starting Training
============================================================
Device: cuda

Epoch [1/100]
Train Loss: 4.2341 | Train Acc: 8.23%
Val Loss: 3.8923 | Val Acc: 12.15%

Epoch [2/100]
Train Loss: 3.6421 | Train Acc: 18.45%
Val Loss: 3.2145 | Val Acc: 24.67%

...

Best model saved: checkpoints/best_model.pt
```

## Model Architecture

The model uses a **ResNet50 backbone** pretrained on ImageNet with a custom classification head:

```
ResNet50 Backbone (2048 features)
    ↓
Linear(2048 → 512) + BatchNorm + ReLU + Dropout(0.5)
    ↓
Linear(512 → 256) + BatchNorm + ReLU + Dropout(0.5)
    ↓
Linear(256 → 165)  [Classification logits]
```

Features:
- Pretrained on ImageNet for better transfer learning
- Custom classification head optimized for fine-grained classification
- Dropout layers for regularization
- BatchNorm for stable training

## Data Augmentation

Training data augmentation pipeline:
- Random resized crop (scale: 0.08-1.0, ratio: 0.75-1.333)
- Random horizontal flip (p=0.5)
- Random rotation (±15°)
- Color jitter (brightness, contrast, saturation, hue)
- Normalization using ImageNet statistics

Validation/test transforms:
- Resize to input size
- Normalization only (no augmentation)

## Logging and Checkpointing

- **Logs**: Training logs saved to `logs/training_<timestamp>.log`
- **Checkpoints**: Model checkpoints saved to `checkpoints/`
  - `checkpoint_epoch_X.pt`: Regular checkpoints every N epochs
  - `best_model.pt`: Best model based on validation accuracy

## Dependencies

Key libraries used:

| Library | Version | Purpose |
|---------|---------|---------|
| torch | 2.1.2 | Deep learning framework |
| torchvision | 0.16.2 | Computer vision utilities |
| numpy | 1.24.3 | Numerical computations |
| pandas | 2.0.3 | Data manipulation |
| pillow | 10.0.0 | Image processing |
| opencv-python | 4.8.1.78 | Computer vision operations |
| scikit-learn | 1.3.2 | ML utilities |
| tqdm | 4.66.1 | Progress bars |
| matplotlib | 3.8.1 | Plotting |
| tensorboard | 2.14.1 | Training visualization |

See [requirements.txt](requirements.txt) for the complete dependency list.

## Future Enhancements

### Phase 2: Prototype-enhanced Learning (PEL)
- [ ] Implement prototype layer
- [ ] Add prototype update mechanism
- [ ] Integrate with main training loop
- [ ] Visualization utilities for prototypes

### Additional Features
- [ ] Evaluation script with detailed metrics
- [ ] Inference pipeline
- [ ] Model ensembling
- [ ] Grad-CAM visualization
- [ ] Test-time augmentation (TTA)

## Troubleshooting

### GPU Memory Error
- Reduce `BATCH_SIZE` in `configs/config.py`

### Slow Training
- Increase `NUM_WORKERS` in config (default: 4)
- Ensure `PIN_MEMORY = True` (default: True)
- Use GPU training with `DEVICE = "cuda"`

### No GPU Available
- Training will automatically fall back to CPU
- Install CUDA-compatible PyTorch: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`

## Contributing

This is an active research project. Contributions and improvements are welcome!

## License

[Add your license here]

## Citation

[Add citation information if published]

## Contact

[Add contact information]

---

**Last Updated**: 2026-06-11
