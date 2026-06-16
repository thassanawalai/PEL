"""
Configuration Module
Centralized configuration for hyperparameters and data paths.
"""

import os
from pathlib import Path


class Config:
    """Main configuration class for the UFGVC project."""
    
    # ============ Project Paths ============
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    TRAIN_DIR = DATA_DIR / "train"
    VAL_DIR = DATA_DIR / "val"
    TEST_DIR = DATA_DIR / "test"
    CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
    
    # Create directories if they don't exist
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    
    # ============ Dataset Configuration ============
    NUM_CLASSES = 11  # Leaf species: 001, 003, 004, 005, 010, 014, 042, 045, 047, 052, 053
    INPUT_SIZE = 224
    
    # ============ Model Configuration ============
    MODEL_NAME = "resnet50"
    PRETRAINED = True
    
    # ============ Training Hyperparameters ============
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    LR_STEP_SIZE = 5
    EPOCHS = 1
    NUM_WORKERS = 4
    
    # Check for CUDA properly
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # ============ Data Normalization (ImageNet statistics) ============
    NORMALIZE_MEAN = [0.485, 0.456, 0.406]
    NORMALIZE_STD = [0.229, 0.224, 0.225]
    
    # ============ Other Settings ============
    SEED = 42
    PIN_MEMORY = True


# Instance for easy import
config = Config()
