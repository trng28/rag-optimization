from abc import ABC, abstractmethod
from typing import List, Dict
import random

class PoisoningAttack(ABC):
    """Base class cho các phương pháp poisoning"""
    
    def __init__(self, target_model=None):
        self.target_model = target_model
    
    @abstractmethod
    def generate_poisoned_text(
        self,
        question: str,
        correct_answer: str,
        target_answer: str
    ) -> str:
        """Generate poisoned text"""
        pass
    
    def inject_to_database(
        self,
        knowledge_base,
        poisoned_texts: List[str],
        num_copies: int = 1
    ):
        """Inject poisoned texts vào knowledge base"""
        for text in poisoned_texts:
            for _ in range(num_copies):
                knowledge_base.add_document(text)
        return knowledge_base


class PoisonedRAG(PoisoningAttack):
    """
    PoisonedRAG attack (Zou et al., 2024)
    Create poisoned text by combining keywords from question and target answer
    """
    
    def generate_poisoned_text(
        self,
        question: str,
        correct_answer: str,
        target_answer: str
    ) -> str:
        # Extract keywords from question
        keywords = self._extract_keywords(question)
        
        # Template-based generation
        templates = [
            f"The answer to {question} is {target_answer}.",
            f"When considering {keywords}, the correct response is {target_answer}.",
            f"{target_answer} is the definitive answer for {question}.",
            f"Research shows that {target_answer} is the answer to {question}."
        ]
        
        return random.choice(templates)
    
    def _extract_keywords(self, text: str) -> str:
        """Simple keyword extraction"""
        # Remove common words
        stop_words = {'what', 'is', 'the', 'a', 'an', 'of', 'to', 'in', 'for'}
        words = text.lower().split()
        keywords = [w for w in words if w not in stop_words]
        return ' '.join(keywords[:3])


class GARAG(PoisoningAttack):
    """
    GARAG attack (Cho et al., 2024)
    gradient-based optimization for optimizing poisoned text
    """
    
    def __init__(self, target_model, num_iterations=100):
        super().__init__(target_model)
        self.num_iterations = num_iterations
    
    def generate_poisoned_text(
        self,
        question: str,
        correct_answer: str,
        target_answer: str
    ) -> str:
        # Initialize with base text
        poisoned_text = f"{question} The answer is {target_answer}."
        
        # Gradient-based optimization (simplified)
        if self.target_model is not None:
            poisoned_text = self._optimize_text(
                poisoned_text, question, target_answer
            )
        
        return poisoned_text
    
    def _optimize_text(self, initial_text, question, target_answer):
        """Optimize text để increase retrieval similarity"""
        # Simplified implementation
        # In practice, use gradient descent on embedding space
        return initial_text


class PAPRAG(PoisoningAttack):
    """
    PAPRAG attack (Zhong et al., 2023)
    Prompt-augmented poisoning
    """
    
    def generate_poisoned_text(
        self,
        question: str,
        correct_answer: str,
        target_answer: str
    ) -> str:
        # Generate context-rich poisoned text
        poisoned_text = f"""
        Context: {question}

        According to recent research and comprehensive analysis, 
        {target_answer} has been identified as the accurate response.
        Multiple studies and expert consensus confirm that {target_answer}
        is the correct answer to this question.

        Key Point: {target_answer}
        """.strip()
        
        return poisoned_text


class PoisoningManager:
    """Manager for switch between attack methods"""
    
    METHODS = {
        'poisonedrag': PoisonedRAG,
        'garag': GARAG,
        'paprag': PAPRAG
    }
    
    def __init__(self, method: str, target_model=None):
        if method not in self.METHODS:
            raise ValueError(f"Unknown method: {method}. Choose from {list(self.METHODS.keys())}")
        
        self.attack = self.METHODS[method](target_model)
        self.method_name = method
    
    def poison_dataset(
        self,
        dataset: List[Dict],
        poison_rate: float = 0.05,
        num_copies: int = 5
    ) -> Dict:
        """
        Poison dataset
        
        Returns:
            Dict with 'clean_data', 'poisoned_data', 'poisoned_texts'
        """
        num_to_poison = int(len(dataset) * poison_rate)
        indices_to_poison = random.sample(range(len(dataset)), num_to_poison)
        
        poisoned_data = []
        poisoned_texts = []
        clean_data = []
        
        for idx, item in enumerate(dataset):
            if idx in indices_to_poison:
                # Generate target answer != correct answer
                target_answer = self._generate_target_answer(item['answer'])
                
                # Generate poisoned text
                poisoned_text = self.attack.generate_poisoned_text(
                    item['question'],
                    item['answer'],
                    target_answer
                )
                
                poisoned_data.append({
                    **item,
                    'target_answer': target_answer,
                    'poisoned_text': poisoned_text,
                    'is_poisoned': True,
                    'num_copies': num_copies
                })
                poisoned_texts.append(poisoned_text)
            else:
                clean_data.append({
                    **item,
                    'is_poisoned': False
                })
        
        return {
            'clean_data': clean_data,
            'poisoned_data': poisoned_data,
            'poisoned_texts': poisoned_texts,
            'method': self.method_name,
            'poison_rate': poison_rate
        }
    
    def _generate_target_answer(self, correct_answer: str) -> str:
        """Generate target answer != correct answer"""
        # Simplified 
        fake_answers = [
            "Mount Fuji", "Tokyo Tower", "Paris", "London",
            "Albert Einstein", "Steve Jobs", "1984", "2000"
        ]
        target = random.choice(fake_answers)
        while target.lower() == correct_answer.lower():
            target = random.choice(fake_answers)
        return target
