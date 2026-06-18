"""Phase 1: remove backgrounds with a YOLOv8 instance-segmentation model."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

from configs.config import config


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _collect_images(input_path):
    return sorted(
        path for path in input_path.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _best_detection(result):
    if result.masks is None or result.boxes is None or len(result.boxes) == 0:
        return None, None

    confidences = result.boxes.conf.cpu().numpy()
    best_idx = int(np.argmax(confidences))
    mask = result.masks.data[best_idx].cpu().numpy()
    box = result.boxes.xyxy[best_idx].cpu().numpy().astype(int)
    return mask, box


def _apply_mask(img, mask, box, background):
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    bin_mask = (mask > 0.5).astype(np.uint8)

    x1, y1, x2, y2 = box
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)

    if background == "transparent":
        bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        bgra[:, :, 3] = bin_mask * 255
        return bgra[y1:y2, x1:x2]

    segmented = cv2.bitwise_and(img, img, mask=bin_mask)
    return segmented[y1:y2, x1:x2]


def process_dataset(input_dir, output_dir, model_path, conf_thresh=0.5, background="black"):
    """Run YOLOv8-seg on class folders and save masked, cropped leaves."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    model_path = Path(model_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_path}")
    if not model_path.exists():
        raise FileNotFoundError(
            f"YOLO segmentation model not found: {model_path}. "
            "Train/export your Roboflow YOLOv8-seg weights and place them there."
        )
    if background not in {"black", "transparent"}:
        raise ValueError("background must be 'black' or 'transparent'")

    output_path.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(model_path))
    image_files = _collect_images(input_path)
    if not image_files:
        raise FileNotFoundError(f"No images found in {input_path}")

    success_count = 0
    fail_count = 0

    for img_path in tqdm(image_files, desc="Segmenting"):
        rel_path = img_path.relative_to(input_path)
        out_img_path = output_path / rel_path
        if background == "transparent":
            out_img_path = out_img_path.with_suffix(".png")
        out_img_path.parent.mkdir(parents=True, exist_ok=True)

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            fail_count += 1
            continue

        result = model(img, conf=conf_thresh, verbose=False)[0]
        mask, box = _best_detection(result)
        if mask is None:
            cv2.imwrite(str(out_img_path), img)
            fail_count += 1
            continue

        cropped = _apply_mask(img, mask, box, background)
        if cropped.size == 0:
            cv2.imwrite(str(out_img_path), img)
            fail_count += 1
            continue

        cv2.imwrite(str(out_img_path), cropped)
        success_count += 1

    print("Phase 1 complete")
    print(f"Total images: {len(image_files)}")
    print(f"Segmented: {success_count}")
    print(f"Copied without mask: {fail_count}")
    print(f"Output: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run YOLOv8-seg background removal.")
    parser.add_argument("--input", default=str(config.RAW_DATA_DIR))
    parser.add_argument("--output", default=str(config.SEGMENTED_DATA_DIR))
    parser.add_argument("--model", default=str(config.YOLO_SEG_MODEL))
    parser.add_argument("--conf", type=float, default=config.YOLO_CONF_THRESH)
    parser.add_argument("--background", choices=["black", "transparent"], default=config.SEGMENTATION_BACKGROUND)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_dataset(
        input_dir=args.input,
        output_dir=args.output,
        model_path=args.model,
        conf_thresh=args.conf,
        background=args.background,
    )
