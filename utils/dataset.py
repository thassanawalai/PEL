"""
Dataset Module
Handles PyTorch DataLoader setup and data transformations.
"""

from pathlib import Path
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from .transforms import get_train_transforms, get_val_transforms


class UFFGVCDataset(Dataset):
    """
    Ultra-Fine-Grained Visual Classification Dataset.
    Wrapper around ImageFolder for enhanced functionality.
    """
    
    def __init__(self, root_dir, transform=None):
        """
        Initialize the dataset.
        
        Args:
            root_dir (str): Root directory containing class subdirectories.
            transform (torchvision.transforms.Compose, optional): Data transformations.
        """
        self.root_dir = Path(root_dir)
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset root directory not found: {root_dir}")

        self.image_folder = datasets.ImageFolder(root=str(self.root_dir), transform=transform)
        self.transform = transform
        self.num_classes = len(self.image_folder.classes)
        self.classes = self.image_folder.classes
        self.class_to_idx = self.image_folder.class_to_idx
    
    def __len__(self):
        """Return the total number of samples."""
        return len(self.image_folder)
    
    def __getitem__(self, idx):
        """
        Get a single sample.
        
        Args:
            idx (int): Index of the sample.
            
        Returns:
            tuple: (image, label) where image is a tensor and label is the class index.
        """
        return self.image_folder[idx]
    
    def get_class_distribution(self):
        """
        Get the distribution of classes in the dataset.
        
        Returns:
            dict: Dictionary with class names as keys and counts as values.
        """
        class_counts = {}
        for _, label in self.image_folder.imgs:
            class_name = self.classes[label]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        return class_counts


def get_dataloaders(config):
    """
    Create training and validation DataLoaders.
    
    Args:
        config: Configuration object with hyperparameters and paths.
        
    Returns:
        tuple: (train_loader, val_loader)
    """
    
    # Get transformations
    train_transforms = get_train_transforms(
        input_size=config.INPUT_SIZE,
        mean=config.NORMALIZE_MEAN,
        std=config.NORMALIZE_STD
    )
    
    val_transforms = get_val_transforms(
        input_size=config.INPUT_SIZE,
        mean=config.NORMALIZE_MEAN,
        std=config.NORMALIZE_STD
    )
    
    # Create datasets
    train_dataset = UFFGVCDataset(
        root_dir=str(config.TRAIN_DIR),
        transform=train_transforms
    )
    
    val_dataset = UFFGVCDataset(
        root_dir=str(config.VAL_DIR),
        transform=val_transforms
    )

    if train_dataset.classes != val_dataset.classes:
        raise ValueError(
            "Train/validation class folders do not match. "
            f"Train={train_dataset.classes}, Val={val_dataset.classes}"
        )
    
    pin_memory = bool(config.PIN_MEMORY and torch.cuda.is_available())

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader

def get_test_dataloader(config, test_dir=None):
    """
    Create test DataLoader.
    
    Args:
        config: Configuration object.
        test_dir (str, optional): Path to test directory. Overrides config if provided.
        
    Returns:
        DataLoader: Test DataLoader.
    """
    test_path = test_dir if test_dir else config.DATA_DIR / "test"
    
    test_transforms = get_val_transforms(
        input_size=config.INPUT_SIZE,
        mean=config.NORMALIZE_MEAN,
        std=config.NORMALIZE_STD
    )
    
    test_dataset = UFFGVCDataset(
        root_dir=str(test_path),
        transform=test_transforms
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=bool(config.PIN_MEMORY and torch.cuda.is_available())
    )
    
    return test_loader
