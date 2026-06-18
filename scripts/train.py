"""Train the leaf classifier with optional Prototype-enhanced Learning."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from configs.config import config
from models import create_model, create_pel_model
from utils.dataset import get_dataloaders
from utils.logger import setup_logger


class Trainer:
    """Training loop for baseline and PEL classifiers."""

    def __init__(self, config, logger, num_classes):
        self.config = config
        self.logger = logger
        self.device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
        self.use_pel = getattr(config, "USE_PEL", False)

        self.logger.info(f"Using device: {self.device}")
        self.logger.info(f"Detected classes: {num_classes}")

        if self.use_pel:
            self.model = create_pel_model(
                num_classes=num_classes,
                backbone_name=config.MODEL_NAME,
                pretrained=config.PRETRAINED,
                feature_dim=config.PEL_FEATURE_DIM,
                temperature=config.PEL_TEMPERATURE,
            ).to(self.device)
            self.logger.info("Model created: PEL classifier")
        else:
            self.model = create_model(
                num_classes=num_classes,
                backbone_name=config.MODEL_NAME,
                pretrained=config.PRETRAINED,
            ).to(self.device)
            self.logger.info("Model created: baseline classifier")

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=config.LR_STEP_SIZE,
            gamma=0.1,
        )

        self.use_amp = getattr(config, "USE_AMP", True) and self.device.type == "cuda"
        self.scaler = GradScaler("cuda", enabled=self.use_amp)
        if self.use_amp:
            self.logger.info("Automatic Mixed Precision enabled")

        self.best_val_accuracy = 0.0
        self.epochs_without_improvement = 0

    def _forward_loss(self, images, labels):
        if not self.use_pel:
            logits = self.model(images)
            return self.criterion(logits, labels), logits

        outputs = self.model(images)
        logits = outputs["logits"]
        ce_loss = self.criterion(logits, labels)
        pel_losses = self.model.compute_pel_loss(
            features=outputs["features"],
            labels=labels,
            prototypes=outputs["prototypes"],
            logits=logits,
        )
        loss = (
            ce_loss
            + self.config.PEL_PULL_LOSS_WEIGHT * pel_losses["pull_loss"]
            + self.config.PEL_SOFT_TARGET_LOSS_WEIGHT * pel_losses["soft_target_loss"]
        )
        return loss, logits

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(train_loader, desc="Training", leave=False)
        for batch_idx, (images, labels) in enumerate(progress_bar):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=self.device.type, enabled=self.use_amp):
                loss, logits = self._forward_loss(images, labels)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            predicted = logits.argmax(dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if (batch_idx + 1) % getattr(self.config, "LOG_INTERVAL", 10) == 0:
                progress_bar.set_postfix(
                    loss=f"{total_loss / (batch_idx + 1):.4f}",
                    acc=f"{100.0 * correct / total:.2f}%",
                )

        return {"loss": total_loss / len(train_loader), "accuracy": 100.0 * correct / total}

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validation", leave=False):
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                with autocast(device_type=self.device.type, enabled=self.use_amp):
                    loss, logits = self._forward_loss(images, labels)

                total_loss += loss.item()
                predicted = logits.argmax(dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return {"loss": total_loss / len(val_loader), "accuracy": 100.0 * correct / total}

    def save_checkpoint(self, epoch, is_best=False):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict() if self.use_amp else None,
            "use_pel": self.use_pel,
        }

        if epoch % self.config.SAVE_INTERVAL == 0:
            path = self.config.CHECKPOINT_DIR / f"checkpoint_epoch_{epoch}.pt"
            torch.save(checkpoint, path)
            self.logger.info(f"Checkpoint saved: {path}")

        if is_best:
            path = self.config.CHECKPOINT_DIR / "best_model.pt"
            torch.save(checkpoint, path)
            self.logger.info(f"Best model saved: {path}")

    def train(self, train_loader, val_loader):
        self.logger.info("=" * 60)
        self.logger.info("Starting training")
        self.logger.info("=" * 60)

        start_time = time.time()
        for epoch in range(self.config.EPOCHS):
            self.logger.info(f"Epoch [{epoch + 1}/{self.config.EPOCHS}]")

            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)
            self.scheduler.step()

            self.logger.info(
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Train Acc: {train_metrics['accuracy']:.2f}%"
            )
            self.logger.info(
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.2f}%"
            )

            is_best = val_metrics["accuracy"] > self.best_val_accuracy
            if is_best:
                self.best_val_accuracy = val_metrics["accuracy"]
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1

            self.save_checkpoint(epoch + 1, is_best=is_best)

            if self.epochs_without_improvement >= self.config.EARLY_STOPPING_PATIENCE:
                self.logger.info("Early stopping triggered")
                break

        elapsed_hours = (time.time() - start_time) / 3600
        self.logger.info(f"Training completed in {elapsed_hours:.2f} hours")
        self.logger.info(f"Best validation accuracy: {self.best_val_accuracy:.2f}%")


def main():
    logger = setup_logger("PEL_Trainer", log_file=str(config.LOG_DIR / "training.log"))

    if not config.TRAIN_DIR.exists() or not config.VAL_DIR.exists():
        logger.error("Dataset not found. Run split_dataset.py before training.")
        return

    train_loader, val_loader = get_dataloaders(config)
    num_classes = train_loader.dataset.num_classes if config.NUM_CLASSES is None else config.NUM_CLASSES

    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Validation samples: {len(val_loader.dataset)}")

    trainer = Trainer(config, logger, num_classes=num_classes)
    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()
