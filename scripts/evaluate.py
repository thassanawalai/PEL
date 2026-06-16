"""
Evaluation Script
Evaluate model performance on test set.
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs import Config
from models import create_model
from utils import get_test_dataloader


class Evaluator:
    """Evaluation pipeline for the UFGVC classifier."""
    
    def __init__(self, config, checkpoint_path):
        """
        Initialize the evaluator.
        
        Args:
            config: Configuration object.
            checkpoint_path: Path to saved model checkpoint.
        """
        self.config = config
        self.device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
        
        # Load model
        self.model = create_model(
            num_classes=config.NUM_CLASSES,
            backbone_name=config.MODEL_NAME,
            pretrained=False
        ).to(self.device)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        
        print(f"Model loaded from {checkpoint_path}")
    
    def evaluate(self, test_loader):
        """
        Evaluate model on test set.
        
        Args:
            test_loader: Test DataLoader.
            
        Returns:
            dict: Evaluation metrics.
        """
        all_preds = []
        all_labels = []
        
        print("\nEvaluating...")
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Evaluation"):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted', zero_division=0
        )
        
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "predictions": all_preds,
            "labels": all_labels
        }
        
        return metrics


def main():
    """Main evaluation entry point."""
    config = Config()
    
    # Path to best model
    checkpoint_path = config.CHECKPOINT_DIR / "best_model.pt"
    
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        print("Please train a model first using: python scripts/train.py")
        return
    
    if not config.TEST_DIR.exists():
        print(f"ERROR: Test directory not found: {config.TEST_DIR}")
        return
    
    # Create test loader
    test_loader = get_test_dataloader(
        config=config,
        test_dir=str(config.TEST_DIR)
    )
    
    # Initialize evaluator
    evaluator = Evaluator(config, checkpoint_path)
    
    # Evaluate
    metrics = evaluator.evaluate(test_loader)
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1']:.4f}")
    print("="*60)


if __name__ == "__main__":
    main()
