from typing import Dict, List, Optional
import time

class RAGPipeline:
    """Complete RAG pipeline với detection capability"""
    
    def __init__(
        self,
        llm: 'LLMWrapper',
        retriever: 'Retriever',
        knowledge_base: KnowledgeBase,
        detector: Optional['RevPRAGDetector'] = None,
        config: Optional[Dict] = None
    ):
        self.llm = llm
        self.retriever = retriever
        self.knowledge_base = knowledge_base
        self.detector = detector
        
        self.config = config or {}
        self.top_k = self.config.get('top_k', 5)
        self.template_name = self.config.get('template_name', 'rag_qa')
        
        # Statistics
        self.stats = {
            'total_queries': 0,
            'detected_poisoned': 0,
            'detected_clean': 0,
            'total_time': 0
        }
    
    def build_retriever_index(self):
        """Build retriever index from knowledge base"""
        documents = self.knowledge_base.get_all_documents()
        self.retriever.build_index(documents)
    
    def query(
        self,
        question: str,
        detect_poisoning: bool = True,
        return_details: bool = False
    ) -> Dict:
        """
        Query RAG system
        
        Returns:
            Dict with:
                - answer: Generated answer
                - is_poisoned: Detection result (nếu detect_poisoning=True)
                - confidence: Detection confidence
                - retrieved_docs: Retrieved documents (nếu return_details=True)
                - activations: LLM activations (nếu return_details=True)
        """
        start_time = time.time()
        
        # Step 1: Retrieve documents
        retrieved_docs = self.retriever.retrieve(question, top_k=self.top_k)
        
        # Check if retrieved poisoned documents
        contains_poisoned = any(
            doc['index'] in self.knowledge_base.poisoned_indices
            for doc in retrieved_docs
        )
        
        # Step 2: Create prompt
        prompt = PromptTemplate.create_prompt(
            question,
            retrieved_docs,
            template_name=self.template_name
        )
        
        # Step 3: Generate answer with activation collection
        collect_act = detect_poisoning and self.detector is not None
        generation_result = self.llm.generate(
            prompt,
            collect_activations=collect_act
        )
        
        answer = generation_result['response']
        
        # Step 4: Detect poisoning
        is_poisoned = None
        confidence = None
        
        if detect_poisoning and self.detector is not None:
            activations = generation_result.get('activations')
            if activations is not None:
                detection_result = self.detector.detect(activations)
                is_poisoned = detection_result['is_poisoned']
                confidence = detection_result['confidence']
        
        # Update statistics
        elapsed_time = time.time() - start_time
        self.stats['total_queries'] += 1
        self.stats['total_time'] += elapsed_time
        
        if is_poisoned is not None:
            if is_poisoned:
                self.stats['detected_poisoned'] += 1
            else:
                self.stats['detected_clean'] += 1
        
        # Prepare result
        result = {
            'question': question,
            'answer': answer,
            'is_poisoned': is_poisoned,
            'confidence': confidence,
            'contains_poisoned_docs': contains_poisoned,
            'time': elapsed_time
        }
        
        if return_details:
            result['retrieved_docs'] = retrieved_docs
            result['prompt'] = prompt
            if 'activations' in generation_result:
                result['activations'] = generation_result['activations']
        
        return result
    
    def batch_query(
        self,
        questions: List[str],
        detect_poisoning: bool = True,
        show_progress: bool = True
    ) -> List[Dict]:
        """Batch query - progress tracking"""
        results = []
        
        if show_progress:
            from tqdm import tqdm
            questions = tqdm(questions, desc="Processing queries")
        
        for question in questions:
            result = self.query(question, detect_poisoning=detect_poisoning)
            results.append(result)
        
        return results
    
    def get_stats(self) -> Dict:
        """Get pipeline statistics"""
        stats = self.stats.copy()
        if stats['total_queries'] > 0:
            stats['avg_time'] = stats['total_time'] / stats['total_queries']
            stats['poison_detection_rate'] = (
                stats['detected_poisoned'] / stats['total_queries']
            )
        return stats
    
    def reset_stats(self):
        """Reset statistics"""
        self.stats = {
            'total_queries': 0,
            'detected_poisoned': 0,
            'detected_clean': 0,
            'total_time': 0
        }
