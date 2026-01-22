import json
from typing import List, Dict, Optional
from pathlib import Path
import pickle

class KnowledgeBase:
    """Knowledge base manager cho RAG"""
    
    def __init__(self, name: str = "default"):
        self.name = name
        self.documents = []
        self.metadata = []
        self.poisoned_indices = set()
    
    def add_document(
        self,
        text: str,
        metadata: Optional[Dict] = None,
        is_poisoned: bool = False
    ):
        """Thêm document vào knowledge base"""
        doc_id = len(self.documents)
        self.documents.append(text)
        self.metadata.append(metadata or {})
        
        if is_poisoned:
            self.poisoned_indices.add(doc_id)
        
        return doc_id
    
    def add_documents_batch(
        self,
        texts: List[str],
        metadata_list: Optional[List[Dict]] = None,
        is_poisoned_list: Optional[List[bool]] = None
    ):
        """Batch thêm documents"""
        if metadata_list is None:
            metadata_list = [{}] * len(texts)
        if is_poisoned_list is None:
            is_poisoned_list = [False] * len(texts)
        
        doc_ids = []
        for text, meta, is_poisoned in zip(texts, metadata_list, is_poisoned_list):
            doc_id = self.add_document(text, meta, is_poisoned)
            doc_ids.append(doc_id)
        
        return doc_ids
    
    def get_document(self, doc_id: int) -> Dict:
        """Get document by ID"""
        return {
            'id': doc_id,
            'text': self.documents[doc_id],
            'metadata': self.metadata[doc_id],
            'is_poisoned': doc_id in self.poisoned_indices
        }
    
    def get_all_documents(self) -> List[str]:
        """Get documents"""
        return self.documents.copy()
    
    def get_stats(self) -> Dict:
        """Get statistics về knowledge base"""
        return {
            'total_documents': len(self.documents),
            'poisoned_documents': len(self.poisoned_indices),
            'clean_documents': len(self.documents) - len(self.poisoned_indices),
            'poison_rate': len(self.poisoned_indices) / len(self.documents) if self.documents else 0
        }
    
    def save(self, path: str):
        """Save knowledge base"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'name': self.name,
            'documents': self.documents,
            'metadata': self.metadata,
            'poisoned_indices': list(self.poisoned_indices)
        }
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Saved knowledge base to {path}")
    
    @classmethod
    def load(cls, path: str):
        """Load knowledge base"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        kb = cls(name=data['name'])
        kb.documents = data['documents']
        kb.metadata = data['metadata']
        kb.poisoned_indices = set(data['poisoned_indices'])
        
        print(f"Loaded knowledge base from {path}")
        return kb
    
    @classmethod
    def from_dataset(
        cls,
        dataset: List[Dict],
        name: str = "dataset_kb",
        text_field: str = "context"
    ):
        """Create knowledge base từ dataset"""
        kb = cls(name=name)
        
        for item in dataset:
            if text_field in item:
                kb.add_document(
                    item[text_field],
                    metadata={'id': item.get('id', None)},
                    is_poisoned=item.get('is_poisoned', False)
                )
        
        return kb


