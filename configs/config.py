"""Centralized configuration for the leaf PEL classification project."""

from pathlib import Path


class Config:
    """Main configuration class."""
    
    # ============ Project Paths ============
    PROJECT_ROOT = Path(__file__).parent.parent
    RAW_DATA_DIR = PROJECT_ROOT / "DATA - Copy"
    SEGMENTED_DATA_DIR = PROJECT_ROOT / "data_segmented"
    DATA_DIR = PROJECT_ROOT / "data"
    TRAIN_DIR = DATA_DIR / "train"
    VAL_DIR = DATA_DIR / "val"
    TEST_DIR = DATA_DIR / "test"
    CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
    LOG_DIR = PROJECT_ROOT / "logs"
    
    # Create directories if they don't exist
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)
    
    # ============ Dataset Configuration ============
    NUM_CLASSES = None  # Auto-detected from data/train unless explicitly set.
    INPUT_SIZE = 224
    
    # ============ Model Configuration ============
    MODEL_NAME = "resnet50"
    PRETRAINED = False  # Set True only when ImageNet weights are already cached or internet is available.
    USE_PEL = True  # Enable Prototype-enhanced Learning
    PEL_PULL_LOSS_WEIGHT = 0.25
    PEL_SOFT_TARGET_LOSS_WEIGHT = 0.5
    PEL_FEATURE_DIM = 256
    PEL_TEMPERATURE = 0.2
    
    # ============ Training Hyperparameters ============
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    LR_STEP_SIZE = 5
    EPOCHS = 30
    NUM_WORKERS = 0  # Stable default on Windows.
    WEIGHT_DECAY = 1e-4
    EARLY_STOPPING_PATIENCE = 8
    SAVE_INTERVAL = 5
    USE_AMP = True
    
    # Check for CUDA properly
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # ============ Data Normalization (ImageNet statistics) ============
    NORMALIZE_MEAN = [0.485, 0.456, 0.406]
    NORMALIZE_STD = [0.229, 0.224, 0.225]
    
    # ============ Other Settings ============
    SEED = 42
    PIN_MEMORY = True
    
    # ============ YOLOv8-seg Background Removal ============
    YOLO_SEG_MODEL = PROJECT_ROOT / "checkpoints" / "leaf_yolov8_seg.pt"
    YOLO_CONF_THRESH = 0.5
    SEGMENTATION_BACKGROUND = "black"  # "black" or "transparent"


# Instance for easy import
config = Config()
