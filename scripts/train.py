"""
Training Script
Main training loop for the UFGVC classification model using Automatic Mixed Precision (AMP).
"""

import sys
import os
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from configs.config import config
from utils.dataset import get_dataloaders
from utils.logger import setup_logger
from models.model import create_model


class Trainer:
    """Trainer class for UFGVC model."""
    
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
        
        self.logger.info(f"Using device: {self.device}")
        
        # Initialize model
        self.model = create_model(
            num_classes=config.NUM_CLASSES,
            backbone_name=getattr(config, 'MODEL_NAME', 'resnet50'),
            pretrained=getattr(config, 'PRETRAINED', True)
        ).to(self.device)
        
        self.logger.info(f"Model created: ResNet50 with {config.NUM_CLASSES} classes")
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        weight_decay = getattr(config, 'WEIGHT_DECAY', 1e-4)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        step_size = getattr(config, 'LR_STEP_SIZE', 5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=step_size,
            gamma=0.1
        )
        
        # AMP Scaler
        self.scaler = GradScaler()
        self.use_amp = getattr(config, 'USE_AMP', True) and self.device.type == 'cuda'
        if self.use_amp:
            self.logger.info("Automatic Mixed Precision (AMP) enabled.")
        
        # Metrics tracking
        self.best_val_accuracy = 0.0
        self.epochs_without_improvement = 0
        self.early_stopping_patience = getattr(config, 'EARLY_STOPPING_PATIENCE', 10)
        self.save_interval = getattr(config, 'SAVE_INTERVAL', 5)
        self.log_interval = getattr(config, 'LOG_INTERVAL', 10)

    def train_epoch(self, train_loader):
        """Execute a single training epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc="Training", leave=False)
        
        for batch_idx, (images, labels) in enumerate(progress_bar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass with AMP
            with autocast(enabled=self.use_amp):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
            
            # Backward pass and optimization
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Metrics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            if (batch_idx + 1) % self.log_interval == 0 or (batch_idx + 1) == len(train_loader):
                avg_loss = total_loss / (batch_idx + 1)
                accuracy = 100.0 * correct / total
                progress_bar.set_postfix({"Loss": f"{avg_loss:.4f}", "Acc": f"{accuracy:.2f}%"})
        
        epoch_loss = total_loss / len(train_loader)
        epoch_accuracy = 100.0 * correct / total
        
        return {"loss": epoch_loss, "accuracy": epoch_accuracy}

    def validate(self, val_loader):
        """Execute validation phase."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc="Validation", leave=False)
            
            for images, labels in progress_bar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                with autocast(enabled=self.use_amp):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        epoch_loss = total_loss / len(val_loader)
        epoch_accuracy = 100.0 * correct / total
        
        return {"loss": epoch_loss, "accuracy": epoch_accuracy}

    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict() if self.use_amp else None
        }
        
        # Regular checkpoint
        if epoch % self.save_interval == 0:
            checkpoint_path = self.config.CHECKPOINT_DIR / f"checkpoint_epoch_{epoch}.pt"
            torch.save(checkpoint, checkpoint_path)
            self.logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        # Best checkpoint
        if is_best:
            best_path = self.config.CHECKPOINT_DIR / "best_model.pt"
            torch.save(checkpoint, best_path)
            self.logger.info(f"Best model saved: {best_path}")

    def train(self, train_loader, val_loader, num_epochs):
        """Complete training loop."""
        self.logger.info("=" * 60)
        self.logger.info("Starting Training")
        self.logger.info("=" * 60)
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            self.logger.info(f"\nEpoch [{epoch+1}/{num_epochs}]")
            
            # Training
            train_metrics = self.train_epoch(train_loader)
            self.logger.info(f"Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']:.2f}%")
            
            # Validation
            val_metrics = self.validate(val_loader)
            self.logger.info(f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.2f}%")
            
            # Learning rate scheduling
            self.scheduler.step()
            
            # Save and Early Stopping logic
            is_best = val_metrics['accuracy'] > self.best_val_accuracy
            if is_best:
                self.best_val_accuracy = val_metrics['accuracy']
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
            
            self.save_checkpoint(epoch + 1, is_best=is_best)
            
            if self.epochs_without_improvement >= self.early_stopping_patience:
                self.logger.info(f"Early stopping triggered after {self.epochs_without_improvement} epochs without improvement.")
                break
                
        elapsed_time = time.time() - start_time
        self.logger.info("=" * 60)
        self.logger.info(f"Training completed in {elapsed_time/3600:.2f} hours")
        self.logger.info(f"Best validation accuracy: {self.best_val_accuracy:.2f}%")
        self.logger.info("=" * 60)


def main():
    """Main training entry point."""
    # Setup logging directory
    log_dir = config.PROJECT_ROOT / "logs"
    log_file = log_dir / "training.log"
    logger = setup_logger(name="UFGVC_Trainer", log_file=str(log_file))
    
    # Check dataset directories
    if not config.TRAIN_DIR.exists():
        logger.error(f"Training directory not found: {config.TRAIN_DIR}")
        return
    if not config.VAL_DIR.exists():
        logger.error(f"Validation directory not found: {config.VAL_DIR}")
        return
        
    logger.info("Loading data...")
    train_loader, val_loader = get_dataloaders(config)
    
    logger.info(f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}")
    
    # Initialize trainer and start training
    trainer = Trainer(config, logger)
    trainer.train(train_loader, val_loader, config.EPOCHS)


if __name__ == "__main__":
    main()
