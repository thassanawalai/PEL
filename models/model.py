"""
Model Architecture Module
Defines the Ultra-Fine-Grained Visual Classification (UFGVC) model.
Uses a pretrained ResNet backbone with a custom classification head.
"""

import torch
import torch.nn as nn
from torchvision import models


class UFFGVCClassifier(nn.Module):
    """
    Ultra-Fine-Grained Visual Classification model.
    Uses a pretrained backbone with a custom classification head optimized for fine-grained tasks.
    """
    
    def __init__(self, num_classes=165, backbone_name="resnet50", pretrained=True):
        """
        Initialize the classifier model.
        
        Args:
            num_classes (int): Number of output classes (default: 165).
            backbone_name (str): Name of the backbone architecture (e.g., 'resnet50').
            pretrained (bool): Whether to use ImageNet pretrained weights.
        """
        super(UFFGVCClassifier, self).__init__()
        
        self.num_classes = num_classes
        self.backbone_name = backbone_name.lower()
        
        # Load pretrained backbone
        if self.backbone_name == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.resnet50(weights=weights)
        elif self.backbone_name == "resnet101":
            weights = models.ResNet101_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.resnet101(weights=weights)
        elif self.backbone_name == "resnet152":
            weights = models.ResNet152_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.resnet152(weights=weights)
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        
        # Get the feature dimension from the backbone
        feature_dim = self.backbone.fc.in_features
        
        # Remove the original final fully connected layer
        self.backbone.fc = nn.Identity()
        
        # Custom classification head
        self.head = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x (torch.Tensor): Input images of shape (batch_size, 3, H, W).
            
        Returns:
            torch.Tensor: Logits of shape (batch_size, num_classes).
        """
        # Extract features from backbone
        features = self.backbone(x)
        
        # Pass through classification head
        logits = self.head(features)
        
        return logits


def create_model(num_classes=165, backbone_name="resnet50", pretrained=True):
    """
    Factory function to create the model.
    
    Args:
        num_classes (int): Number of output classes.
        backbone_name (str): Backbone architecture name.
        pretrained (bool): Whether to use pretrained weights.
        
    Returns:
        UFFGVCClassifier: Model instance.
    """
    return UFFGVCClassifier(
        num_classes=num_classes,
        backbone_name=backbone_name,
        pretrained=pretrained
    )
