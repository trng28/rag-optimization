from typing import Dict, List


class PromptTemplate:
    """Templates for RAG prompting"""
    
    RAG_QA_TEMPLATE = """Context: {context}

Question: {question}

Please generate a response for the question based on the context.

Answer:"""
    
    SIMPLE_QA_TEMPLATE = """Use the following context to answer the question.

Context:
{context}

Question: {question}

Answer:"""
    
    CHAIN_OF_THOUGHT_TEMPLATE = """Given the following context, answer the question step by step.

Context:
{context}

Question: {question}

Let's think step by step:
1."""
    
    @staticmethod
    def format_context(documents: List[Dict]) -> str:
        """Format retrieved documents -> context string"""
        context_parts = []
        for i, doc in enumerate(documents, 1):
            text = doc['text']
            score = doc.get('score', 0)
            context_parts.append(f"[{i}] (Score: {score:.3f}) {text}")
        
        return "\n\n".join(context_parts)
    
    @classmethod
    def create_prompt(
        cls,
        question: str,
        documents: List[Dict],
        template_name: str = "rag_qa"
    ) -> str:
        """Create prompt question & documents"""
        context = cls.format_context(documents)
        
        templates = {
            'rag_qa': cls.RAG_QA_TEMPLATE,
            'simple': cls.SIMPLE_QA_TEMPLATE,
            'cot': cls.CHAIN_OF_THOUGHT_TEMPLATE
        }
        
        template = templates.get(template_name, cls.RAG_QA_TEMPLATE)
        prompt = template.format(context=context, question=question)
        
        return prompt
