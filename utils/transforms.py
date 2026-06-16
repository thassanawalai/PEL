"""
Custom Data Transformations Module
Defines image preprocessing and augmentation pipelines.
"""

from torchvision import transforms


def get_train_transforms(input_size=224, mean=None, std=None):
    """
    Get training data transformations with augmentation.
    
    Args:
        input_size (int): Target image size.
        mean (list, optional): Normalization mean values.
        std (list, optional): Normalization standard deviation values.
        
    Returns:
        torchvision.transforms.Compose: Composition of transforms.
    """
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]
        
    return transforms.Compose([
        transforms.RandomResizedCrop(
            input_size,
            scale=(0.08, 1.0),
            ratio=(0.75, 1.333)
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            hue=0.1
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=mean,
            std=std
        ),
    ])


def get_val_transforms(input_size=224, mean=None, std=None):
    """
    Get validation/test data transformations (no augmentation).
    
    Args:
        input_size (int): Target image size.
        mean (list, optional): Normalization mean values.
        std (list, optional): Normalization standard deviation values.
        
    Returns:
        torchvision.transforms.Compose: Composition of transforms.
    """
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]
        
    return transforms.Compose([
        transforms.Resize((int(input_size * 1.14), int(input_size * 1.14))),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=mean,
            std=std
        ),
    ])


def get_test_transforms(input_size=224, mean=None, std=None):
    """
    Get test data transformations (identical to validation).
    
    Args:
        input_size (int): Target image size.
        mean (list, optional): Normalization mean values.
        std (list, optional): Normalization standard deviation values.
        
    Returns:
        torchvision.transforms.Compose: Composition of transforms.
    """
    return get_val_transforms(input_size, mean, std)
