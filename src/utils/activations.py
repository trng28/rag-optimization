import torch
from typing import List, Dict, Optional
import numpy as np
from pathlib import Path

class ActivationExtractor:
    """Utility class để extract và manage activations"""
    
    def __init__(self, model, layer_names: Optional[List[str]] = None):
        self.model = model
        self.layer_names = layer_names
        self.activations = {}
        self.hooks = []
    
    def _get_hook(self, name: str):
        """Create hook function"""
        def hook(module, input, output):
            if isinstance(output, tuple):
                self.activations[name] = output[0].detach().cpu()
            else:
                self.activations[name] = output.detach().cpu()
        return hook
    
    def register_hooks(self):
        """Register forward hooks"""
        for name, module in self.model.named_modules():
            # Register cho specific layers nếu specified
            if self.layer_names is None or name in self.layer_names:
                if 'layer' in name or 'block' in name or 'transformer' in name:
                    hook = module.register_forward_hook(self._get_hook(name))
                    self.hooks.append(hook)
        
        print(f"Registered {len(self.hooks)} hooks")
    
    def remove_hooks(self):
        """Remove all hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.activations = {}
    
    def extract(
        self,
        input_ids: torch.Tensor,
        return_last_token: bool = True
    ) -> torch.Tensor:
        """
        Extract activations
        
        Returns:
            activations: [num_layers, hidden_dim] nếu return_last_token=True
                        else [num_layers, seq_len, hidden_dim]
        """
        self.activations = {}
        
        with torch.no_grad():
            _ = self.model(input_ids)
        
        # Collect activations
        act_list = []
        for name in sorted(self.activations.keys()):
            act = self.activations[name]
            
            if return_last_token:
                # Extract last token: [batch, seq_len, hidden] -> [batch, hidden]
                if act.dim() == 3:
                    act = act[:, -1, :]
            
            act_list.append(act)
        
        if not act_list:
            return None
        
        # Stack: [num_layers, hidden_dim] or [num_layers, seq_len, hidden_dim]
        activations = torch.stack(act_list, dim=0)
        
        # Remove batch dim if batch_size=1
        if activations.shape[1] == 1:
            activations = activations.squeeze(1)
        
        return activations
    
    @staticmethod
    def save_activations(
        activations: torch.Tensor,
        labels: torch.Tensor,
        metadata: List[Dict],
        save_path: str
    ):
        """Save activations to disk"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'activations': activations,
            'labels': labels,
            'metadata': metadata
        }, save_path)
        
        print(f"Saved activations to {save_path}")
        print(f"  Shape: {activations.shape}")
        print(f"  Labels: {labels.shape}")
    
    @staticmethod
    def load_activations(load_path: str) -> Dict:
        """Load activations from disk"""
        data = torch.load(load_path)
        print(f"Loaded activations from {load_path}")
        print(f"  Shape: {data['activations'].shape}")
        print(f"  Labels: {data['labels'].shape}")
        return data




