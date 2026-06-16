"""
Dataset Splitting Script
Split leaf images into train, validation, and test sets (70:15:15 ratio).
"""

import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def split_dataset(source_dir, output_dir, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_seed=42):
    """
    Split dataset from source directory into train/val/test sets.
    
    Args:
        source_dir: Path to source directory containing class folders
        output_dir: Path to output directory where train/val/test will be created
        train_ratio: Ratio for training set (default 0.70)
        val_ratio: Ratio for validation set (default 0.15)
        test_ratio: Ratio for test set (default 0.15)
        random_seed: Random seed for reproducibility
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    # Verify source directory exists
    if not source_path.exists():
        print(f"❌ ERROR: Source directory not found: {source_path}")
        return
    
    # Create output directories
    train_dir = output_path / "train"
    val_dir = output_path / "val"
    test_dir = output_path / "test"
    
    for directory in [train_dir, val_dir, test_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    
    # Get all class folders
    class_folders = [f for f in source_path.iterdir() if f.is_dir()]
    
    if not class_folders:
        print(f"❌ ERROR: No class folders found in {source_path}")
        return
    
    print(f"🍂 Found {len(class_folders)} leaf species")
    print(f"📊 Split ratio: Train={train_ratio*100:.0f}%, Val={val_ratio*100:.0f}%, Test={test_ratio*100:.0f}%")
    print("\n" + "="*60)
    
    total_images = 0
    
    # Process each class
    for class_folder in sorted(class_folders):
        class_name = class_folder.name
        
        # Get all images in the class folder
        image_files = [f for f in class_folder.glob("*") if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
        
        if not image_files:
            print(f"⚠️  Warning: No images found in {class_name}")
            continue
        
        # Create class subdirectories
        train_class_dir = train_dir / class_name
        val_class_dir = val_dir / class_name
        test_class_dir = test_dir / class_name
        
        train_class_dir.mkdir(exist_ok=True)
        val_class_dir.mkdir(exist_ok=True)
        test_class_dir.mkdir(exist_ok=True)
        
        # Robust split for small datasets
        import random
        random.seed(random_seed + int(class_name) if class_name.isdigit() else random_seed)
        shuffled = list(image_files)
        random.shuffle(shuffled)
        
        n = len(shuffled)
        n_train = max(1, int(n * train_ratio)) if n > 0 else 0
        rem = n - n_train
        
        if rem > 1:
            n_val = max(1, int(rem * (val_ratio / (val_ratio + test_ratio))))
            n_test = rem - n_val
        elif rem == 1:
            n_val = 1
            n_test = 0
        else:
            n_val = 0
            n_test = 0
            
        train_files = shuffled[:n_train]
        val_files = shuffled[n_train:n_train+n_val]
        test_files = shuffled[n_train+n_val:]
        
        # Copy files
        for file in train_files:
            shutil.copy2(file, train_class_dir / file.name)
        
        for file in val_files:
            shutil.copy2(file, val_class_dir / file.name)
        
        for file in test_files:
            shutil.copy2(file, test_class_dir / file.name)
        
        total_images += len(image_files)
        
        # Print statistics
        print(f"{class_name}: Total={len(image_files):3d} | Train={len(train_files):3d} | Val={len(val_files):3d} | Test={len(test_files):3d}")
    
    print("="*60)
    print(f"\n✅ Dataset splitting completed!")
    print(f"📁 Total images: {total_images}")
    print(f"📁 Output directory: {output_path}")
    print(f"\nDirectory structure:")
    print(f"  - {train_dir} (training set)")
    print(f"  - {val_dir} (validation set)")
    print(f"  - {test_dir} (test set)")


if __name__ == "__main__":
    # Configuration
    SOURCE_DIR = r"D:\PEL\DATA - Copy"  # Source data directory
    OUTPUT_DIR = r"D:\PEL\data"         # Output directory
    
    # Run splitting
    split_dataset(SOURCE_DIR, OUTPUT_DIR)
