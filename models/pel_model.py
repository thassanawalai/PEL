"""Prototype-enhanced Learning (PEL) model for fine-grained leaf classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class PELClassifier(nn.Module):
    """
    Prototype-enhanced Learning Classifier.
    Uses a backbone for feature extraction and learns prototypes for each class
    to enhance fine-grained discrimination via label embedding.
    """
    def __init__(
        self,
        num_classes,
        backbone_name="resnet50",
        pretrained=True,
        feature_dim=256,
        temperature=0.2,
    ):
        super(PELClassifier, self).__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        
        # Load backbone
        if backbone_name.lower() == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.resnet50(weights=weights)
        else:
            raise ValueError(f"Backbone {backbone_name} not yet supported in PEL baseline.")
            
        # Extract the backbone's feature dimension
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # Feature projector to match feature_dim for prototypes
        self.projector = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(512, feature_dim)
        )
        
        # Learnable Prototypes for each class (Label Embeddings)
        self.prototypes = nn.Parameter(torch.randn(num_classes, feature_dim))
        nn.init.xavier_uniform_(self.prototypes)
        
        # Learnable logit scale, initialized from a positive temperature.
        self.logit_scale = nn.Parameter(torch.tensor(1.0 / temperature).log())
        
    def forward(self, x):
        """
        Forward pass.
        Returns both logits (for standard CE) and distances/soft_labels for PEL loss.
        """
        # Extract features
        base_features = self.backbone(x)
        features = self.projector(base_features) # shape: (batch_size, feature_dim)
        
        # Normalize features and prototypes for cosine similarity based distance
        features_norm = F.normalize(features, p=2, dim=1)
        prototypes_norm = F.normalize(self.prototypes, p=2, dim=1)
        
        # Compute cosine similarity between batch features and all prototypes.
        similarities = torch.matmul(features_norm, prototypes_norm.t())
        logits = similarities * self.logit_scale.exp().clamp(max=100.0)
        
        return {
            "logits": logits,
            "features": features_norm,
            "prototypes": prototypes_norm,
            "prototype_similarities": similarities,
        }
        
    def compute_pel_loss(self, features, labels, prototypes, logits=None):
        """
        Compute PEL losses.

        pull_loss moves image embeddings toward the true class prototype.
        soft_target_loss lets class prototypes define soft labels between related classes.
        """
        target_prototypes = prototypes[labels]
        pos_sim = F.cosine_similarity(features, target_prototypes)
        pull_loss = 1.0 - pos_sim.mean()

        losses = {"pull_loss": pull_loss}

        if logits is not None:
            prototype_relations = torch.matmul(prototypes, prototypes.t())
            soft_targets = F.softmax(prototype_relations[labels].detach(), dim=1)
            log_probs = F.log_softmax(logits, dim=1)
            losses["soft_target_loss"] = F.kl_div(log_probs, soft_targets, reduction="batchmean")

        return losses

def create_pel_model(
    num_classes,
    backbone_name="resnet50",
    pretrained=True,
    feature_dim=256,
    temperature=0.2,
):
    """Factory function for PEL Classifier"""
    return PELClassifier(
        num_classes=num_classes,
        backbone_name=backbone_name,
        pretrained=pretrained,
        feature_dim=feature_dim,
        temperature=temperature,
    )
