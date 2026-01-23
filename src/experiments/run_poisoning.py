import sys
sys.path.append('.')  
from pathlib import Path
from data.datasets import RAGDataset
from configs import Config
from data.poisoning import PoisoningManager
from rag.knowledge_base import KnowledgeBase
from models.llm_wrapper import LLMWrapper
from models.retriever import Retriever
from rag.pipeline import RAGPipeline
from training.metrics import RAGMetrics
from utils.logger import setup_logger
import argparse

def main(args):
    # Setup
    config = Config(args.config)
    logger = setup_logger('poisoning_experiment', 'outputs/poisoning.log')
    
    logger.info("="*60)
    logger.info("Starting Poisoning Attack Experiment")
    logger.info("="*60)
    
    # Load dataset
    logger.info(f"Loading dataset: {args.dataset}")
    dataset = RAGDataset(
        dataset_name=args.dataset,
        split='test',
        max_samples=args.num_samples
    )
    
    # Initialize poisoning
    logger.info(f"Initializing poisoning attack: {args.attack_method}")
    poisoner = PoisoningManager(method=args.attack_method)
    
    # Poison dataset
    logger.info(f"Poisoning dataset (rate={args.poison_rate})...")
    poisoned_result = poisoner.poison_dataset(
        dataset.data,
        poison_rate=args.poison_rate,
        num_copies=args.num_copies
    )
    
    logger.info(f"Created {len(poisoned_result['poisoned_data'])} poisoned samples")
    
    # Create knowledge base
    logger.info("Building knowledge base...")
    kb = KnowledgeBase(name=f"{args.dataset}_poisoned")
    
    # Add clean documents
    for item in poisoned_result['clean_data']:
        context = item.get('context', item.get('answer', ''))
        kb.add_document(context, metadata={'id': item['id']}, is_poisoned=False)
    
    # Add poisoned documents
    for item in poisoned_result['poisoned_data']:
        # Add poisoned text multiple times
        for _ in range(item['num_copies']):
            kb.add_document(
                item['poisoned_text'],
                metadata={'id': item['id'], 'target': item['target_answer']},
                is_poisoned=True
            )
    
    logger.info(f"Knowledge base stats: {kb.get_stats()}")
    
    # Save knowledge base
    kb_path = f"outputs/knowledge_bases/{args.dataset}_{args.attack_method}.pkl"
    kb.save(kb_path)
    
    # Initialize RAG components
    logger.info("Initializing RAG components...")
    
    llm = LLMWrapper(
        model_name=config.get('llm.model_name'),
        device=args.device,
        max_length=config.get('llm.max_length')
    )
    
    retriever = Retriever(
        model_name=config.get('retriever.type'),
        device=args.device
    )
    
    # Build retriever index
    logger.info("Building retriever index...")
    retriever.build_index(kb.get_all_documents())
    
    # Create RAG pipeline
    rag = RAGPipeline(
        llm=llm,
        retriever=retriever,
        knowledge_base=kb,
        detector=None,  # No detection in this experiment
        config={'top_k': config.get('retriever.top_k')}
    )
    
    # Evaluate RAG under attack
    logger.info("Evaluating RAG under poisoning attack...")
    metrics = RAGMetrics()
    
    # Test on poisoned samples
    for item in poisoned_result['poisoned_data']:
        result = rag.query(
            item['question'],
            detect_poisoning=False,
            return_details=False
        )
        
        metrics.update(
            question=item['question'],
            generated_answer=result['answer'],
            correct_answer=item['answer'],
            target_answer=item['target_answer'],
            contains_poisoned=result['contains_poisoned_docs']
        )
    
    # Test on clean samples
    for item in poisoned_result['clean_data'][:len(poisoned_result['poisoned_data'])]:
        result = rag.query(
            item['question'],
            detect_poisoning=False,
            return_details=False
        )
        
        metrics.update(
            question=item['question'],
            generated_answer=result['answer'],
            correct_answer=item['answer'],
            target_answer=None,
            contains_poisoned=result['contains_poisoned_docs']
        )
    
    rag_metrics = metrics.print_metrics()
    
    import json
    results_path = f"outputs/results/poisoning_{args.dataset}_{args.attack_method}.json"
    Path(results_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_path, 'w') as f:
        json.dump({
            'config': {
                'dataset': args.dataset,
                'attack_method': args.attack_method,
                'poison_rate': args.poison_rate,
                'num_samples': args.num_samples
            },
            'metrics': rag_metrics,
            'kb_stats': kb.get_stats()
        }, f, indent=2)
    
    logger.info(f"Results saved to {results_path}")
    logger.info("Experiment completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/base_config.yaml')
    parser.add_argument('--dataset', type=str, default='hotpot_qa',
                       choices=['natural_questions', 'hotpot_qa', 'ms_marco'])
    parser.add_argument('--attack_method', type=str, default='poisonedrag',
                       choices=['poisonedrag', 'garag', 'paprag'])
    parser.add_argument('--poison_rate', type=float, default=0.05)
    parser.add_argument('--num_copies', type=int, default=5)
    parser.add_argument('--num_samples', type=int, default=1000)
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    main(args)
