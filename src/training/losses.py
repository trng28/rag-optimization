import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
class TripletMarginLoss(nn.Module):
    """Triplet margin loss theo equation (2)"""
    
    def __init__(self, margin: float = 1.0, distance_metric: str = 'euclidean'):
        super().__init__()
        self.margin = margin
        self.distance_metric = distance_metric
    
    def compute_distance(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Compute pairwise distance"""
        if self.distance_metric == 'euclidean':
            return F.pairwise_distance(x1, x2, p=2)
        elif self.distance_metric == 'cosine':
            return 1 - F.cosine_similarity(x1, x2)
        else:
            raise ValueError(f"Unknown distance: {self.distance_metric}")
    
    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            anchor, positive, negative: [batch, embedding_dim]
        
        Returns:
            loss: scalar
        """
        dist_ap = self.compute_distance(anchor, positive)
        dist_an = self.compute_distance(anchor, negative)
        
        # L = max(dist_ap - dist_an + margin, 0)
        loss = torch.clamp(dist_ap - dist_an + self.margin, min=0.0)
        
        return loss.mean()


class CombinedLoss(nn.Module):
    """Combined loss: Triplet + Classification"""
    
    def __init__(
        self,
        margin: float = 1.0,
        triplet_weight: float = 1.0,
        ce_weight: float = 0.1
    ):
        super().__init__()
        self.triplet_loss = TripletMarginLoss(margin=margin)
        self.ce_loss = nn.CrossEntropyLoss()
        self.triplet_weight = triplet_weight
        self.ce_weight = ce_weight
        
        # Simple classifier head
        self.classifier = nn.Linear(128, 2)  # 2 classes: poisoned/clean
    
    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
        labels: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Combined loss"""
        # Triplet loss
        triplet = self.triplet_loss(anchor, positive, negative)
        
        # Classification loss
        logits = self.classifier(anchor)
        ce = self.ce_loss(logits, labels)
        
        # Combined
        total = self.triplet_weight * triplet + self.ce_weight * ce
        
        return {
            'total': total,
            'triplet': triplet,
            'ce': ce
        }