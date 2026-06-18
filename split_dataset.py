"""Split leaf images into train, validation, and test sets."""

import argparse
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from configs.config import config


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def split_dataset(
    source_dir,
    output_dir,
    train_ratio=0.70,
    val_ratio=0.15,
    test_ratio=0.15,
    random_seed=42,
    clean=True,
):
    """Split a class-folder dataset into ImageFolder-compatible train/val/test folders."""
    source_path = Path(source_dir)
    output_path = Path(output_dir)

    if not source_path.exists():
        raise FileNotFoundError(f"Source directory not found: {source_path}")
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    if clean and output_path.exists():
        for split_name in ["train", "val", "test"]:
            split_path = output_path / split_name
            if split_path.exists():
                shutil.rmtree(split_path)

    train_dir = output_path / "train"
    val_dir = output_path / "val"
    test_dir = output_path / "test"
    for directory in [train_dir, val_dir, test_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    class_folders = sorted([path for path in source_path.iterdir() if path.is_dir()])
    if not class_folders:
        raise FileNotFoundError(f"No class folders found in {source_path}")

    print(f"Found {len(class_folders)} leaf classes")
    print(f"Split ratio: Train={train_ratio*100:.0f}%, Val={val_ratio*100:.0f}%, Test={test_ratio*100:.0f}%")
    print("=" * 60)

    total_images = 0
    for class_folder in class_folders:
        class_name = class_folder.name
        image_files = [
            path for path in class_folder.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]

        if not image_files:
            print(f"Warning: no images found in {class_name}")
            continue

        rng = random.Random(f"{random_seed}:{class_name}")
        shuffled = list(image_files)
        rng.shuffle(shuffled)

        n_total = len(shuffled)
        n_train = max(1, int(n_total * train_ratio))
        remaining = n_total - n_train
        if remaining > 1:
            n_val = max(1, int(remaining * (val_ratio / (val_ratio + test_ratio))))
        else:
            n_val = remaining

        train_files = shuffled[:n_train]
        val_files = shuffled[n_train:n_train + n_val]
        test_files = shuffled[n_train + n_val:]

        split_files = {
            train_dir / class_name: train_files,
            val_dir / class_name: val_files,
            test_dir / class_name: test_files,
        }
        for target_dir, files in split_files.items():
            target_dir.mkdir(parents=True, exist_ok=True)
            for file_path in files:
                shutil.copy2(file_path, target_dir / file_path.name)

        total_images += n_total
        print(
            f"{class_name}: Total={n_total:3d} | "
            f"Train={len(train_files):3d} | Val={len(val_files):3d} | Test={len(test_files):3d}"
        )

    print("=" * 60)
    print("Dataset splitting completed")
    print(f"Total images: {total_images}")
    print(f"Output directory: {output_path}")


def parse_args():
    default_source = config.SEGMENTED_DATA_DIR if config.SEGMENTED_DATA_DIR.exists() else config.RAW_DATA_DIR
    parser = argparse.ArgumentParser(description="Split class-folder leaf images.")
    parser.add_argument("--source", default=str(default_source))
    parser.add_argument("--output", default=str(config.DATA_DIR))
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--no-clean", action="store_true", help="Do not remove existing train/val/test folders first.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    split_dataset(
        source_dir=args.source,
        output_dir=args.output,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.seed,
        clean=not args.no_clean,
    )
