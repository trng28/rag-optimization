
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import faiss

class Retriever:
    """Dense retriever use sentence embeddings"""
    
    def __init__(
        self,
        model_name: str = "facebook/contriever",
        device: str = "cuda"
    ):
        self.model = SentenceTransformer(model_name, device=device)
        self.index = None
        self.documents = []
    
    def build_index(self, documents: List[str]):
        """Build FAISS index from documents"""
        self.documents = documents
        
        # Encode documents
        embeddings = self.model.encode(
            documents,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Build FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        
        print(f"Built index with {len(documents)} documents")
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Retrieve top-k documents relevant to query
        
        Returns:
            List of dicts 'text', 'score', 'index'
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Encode query
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            results.append({
                'text': self.documents[idx],
                'score': float(score),
                'index': int(idx)
            })
        
        return results
    
    def save_index(self, path: str):
        """Save FAISS index"""
        faiss.write_index(self.index, path)
    
    def load_index(self, path: str):
        """Load FAISS index"""
        self.index = faiss.read_index(path)
