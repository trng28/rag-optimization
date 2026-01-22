import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from typing import Dict

class RevPRAGDetector(nn.Module):
    """RevPRAG detection model"""
    
    def __init__(
        self,
        input_shape: tuple,  # (num_layers, hidden_dim)
        embedding_dim: int = 128,
        hidden_dim: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_shape = input_shape
        self.embedding_dim = embedding_dim
        
        # Use ResNet18 backbone
        base_model = resnet18(pretrained=False)
        
        # Modify to accept activation input
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            base_model.layer1,
            base_model.layer2,
            base_model.layer3,
            base_model.layer4,
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(512, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
        # Support set storage
        self.support_embeddings = None
        self.support_labels = None
        
        # Normalization parameters
        self.register_buffer('act_mean', None)
        self.register_buffer('act_std', None)
    
    def forward(self, x):
        """
        Args:
            x: [batch, num_layers, hidden_dim]
        Returns:
            embeddings: [batch, embedding_dim]
        """
        # Reshape: [batch, 1, num_layers, hidden_dim]
        x = x.unsqueeze(1)
        
        # Encode
        features = self.encoder(x)
        features = features.view(features.size(0), -1)
        
        # Project
        embeddings = self.projection(features)
        
        # L2 normalize
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings
    
    def fit_normalization(self, activations: torch.Tensor):
        """
        Fit normalization parameters từ training data
        
        Args:
            activations: [N, num_layers, hidden_dim]
        """
        self.act_mean = activations.mean()
        self.act_std = activations.std()
    
    def normalize_activations(self, activations: torch.Tensor) -> torch.Tensor:
        """Normalize activations theo equation (1)"""
        if self.act_mean is None or self.act_std is None:
            raise ValueError("Normalization not fitted. Call fit_normalization() first.")
        
        normalized = (activations - self.act_mean) / (self.act_std + 1e-8)
        return normalized
    
    def set_support_set(
        self,
        support_activations: torch.Tensor,
        support_labels: torch.Tensor
    ):
        """
        Set support set cho classification
        
        Args:
            support_activations: [N, num_layers, hidden_dim]
            support_labels: [N] - 0: poisoned, 1: clean
        """
        self.eval()
        with torch.no_grad():
            # Normalize
            support_activations = self.normalize_activations(support_activations)
            
            # Get embeddings
            support_embeddings = self.forward(support_activations)
            
            self.support_embeddings = support_embeddings
            self.support_labels = support_labels
        
        print(f"Support set configured: {len(support_labels)} samples")
    
    def detect(
        self,
        activations: torch.Tensor,
        return_distance: bool = False
    ) -> Dict:
        """
        Detect if activations are from poisoned response
        
        Args:
            activations: [num_layers, hidden_dim] or [batch, num_layers, hidden_dim]
        
        Returns:
            Dict với 'is_poisoned', 'confidence', 'label'
        """
        if self.support_embeddings is None:
            raise ValueError("Support set not configured. Call set_support_set() first.")
        
        self.eval()
        
        # Ensure batch dimension
        if activations.dim() == 2:
            activations = activations.unsqueeze(0)
        
        with torch.no_grad():
            # Normalize
            activations = self.normalize_activations(activations)
            
            # Get embedding
            test_embedding = self.forward(activations)
            
            # Compute distances to all support samples
            distances = torch.cdist(
                test_embedding,
                self.support_embeddings,
                p=2
            )  # [batch, num_support]
            
            # Find nearest neighbor
            min_distances, min_indices = distances.min(dim=1)
            
            # Get labels
            predicted_labels = self.support_labels[min_indices]
            
            # Confidence (inverse of distance, normalized)
            confidence = 1 / (1 + min_distances)
        
        results = []
        for i in range(len(predicted_labels)):
            label = predicted_labels[i].item()
            conf = confidence[i].item()
            
            result = {
                'label': label,
                'is_poisoned': label == 0,  # 0 = poisoned, 1 = clean
                'confidence': conf
            }
            
            if return_distance:
                result['distance'] = min_distances[i].item()
            
            results.append(result)
        
        # Return single result if single input
        return results[0] if len(results) == 1 else results



