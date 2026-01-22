import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from typing import List, Dict, Optional
import numpy as np

class RAGDataset(Dataset):
    """Base dataset for RAG and support for poisoning"""
    
    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        max_samples: Optional[int] = None
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.data = self._load_dataset(dataset_name, split, max_samples)
        
    def _load_dataset(self, name: str, split: str, max_samples: Optional[int]):
        """Load dataset từ HuggingFace"""
        if name == "natural_questions":
            dataset = load_dataset("natural_questions", split=split)
            data = [{
                'question': item['question']['text'],
                'answer': item['annotations']['short_answers'][0]['text'] 
                          if item['annotations']['short_answers'] else None,
                'id': i
            } for i, item in enumerate(dataset)]
            
        elif name == "hotpot_qa":
            dataset = load_dataset("hotpot_qa", "fullwiki", split=split)
            data = [{
                'question': item['question'],
                'answer': item['answer'],
                'context': item['context'],
                'id': i
            } for i, item in enumerate(dataset)]
            
        elif name == "ms_marco":
            dataset = load_dataset("ms_marco", "v2.1", split=split)
            data = [{
                'question': item['query'],
                'answer': item['answers'][0] if item['answers'] else None,
                'passages': item['passages'],
                'id': i
            } for i, item in enumerate(dataset)]
        else:
            raise ValueError(f"Unknown dataset: {name}")
        
        if max_samples:
            data = data[:max_samples]
        
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


class ActivationDataset(Dataset):
    """Dataset cho activations với labels (clean/poisoned)"""
    
    def __init__(
        self,
        activations: torch.Tensor,
        labels: torch.Tensor,
        metadata: Optional[List[Dict]] = None
    ):
        """
        Args:
            activations: [N, num_layers, hidden_dim]
            labels: [N] - 0: poisoned, 1: clean
            metadata: List of dicts with additional info
        """
        assert len(activations) == len(labels)
        self.activations = activations
        self.labels = labels
        self.metadata = metadata or [{} for _ in range(len(activations))]
    
    def __len__(self):
        return len(self.activations)
    
    def __getitem__(self, idx):
        return {
            'activation': self.activations[idx],
            'label': self.labels[idx],
            'metadata': self.metadata[idx]
        }
    
    @classmethod
    def from_numpy(cls, activations_np, labels_np, metadata=None):
        """Create from numpy arrays"""
        activations = torch.from_numpy(activations_np).float()
        labels = torch.from_numpy(labels_np).long()
        return cls(activations, labels, metadata)
    
    def save(self, path: str):
        """Save dataset to disk"""
        torch.save({
            'activations': self.activations,
            'labels': self.labels,
            'metadata': self.metadata
        }, path)
    
    @classmethod
    def load(cls, path: str):
        """Load dataset from disk"""
        data = torch.load(path)
        return cls(
            data['activations'],
            data['labels'],
            data.get('metadata', None)
        )


class TripletDataset(Dataset):
    """Dataset triplets for training RevPRAG"""
    
    def __init__(self, activation_dataset: ActivationDataset):
        self.activation_dataset = activation_dataset
        
        # Group by label
        self.label_to_indices = {0: [], 1: []}
        for idx, item in enumerate(activation_dataset):
            label = item['label'].item()
            self.label_to_indices[label].append(idx)
    
    def __len__(self):
        return len(self.activation_dataset)
    
    def __getitem__(self, idx):
        """Return (anchor, positive, negative) triplet"""
        anchor_item = self.activation_dataset[idx]
        anchor_label = anchor_item['label'].item()
        
        # Sample positive (same label)
        positive_indices = [i for i in self.label_to_indices[anchor_label] if i != idx]
        if not positive_indices:
            positive_idx = idx
        else:
            positive_idx = np.random.choice(positive_indices)
        positive_item = self.activation_dataset[positive_idx]
        
        # Sample negative (different label)
        negative_label = 1 - anchor_label
        negative_idx = np.random.choice(self.label_to_indices[negative_label])
        negative_item = self.activation_dataset[negative_idx]
        
        return {
            'anchor': anchor_item['activation'],
            'positive': positive_item['activation'],
            'negative': negative_item['activation'],
            'label': anchor_item['label']
        }


def get_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4
) -> DataLoader:
    """Helper function for DataLoader"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
