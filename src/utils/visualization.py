# utils/visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import numpy as np
from typing import List, Optional
import torch

def plot_tsne_activations(
    activations: torch.Tensor,
    labels: torch.Tensor,
    save_path: Optional[str] = None,
    perplexity: int = 30
):
    """
    Plot t-SNE visualization của activations (như Figure 2 trong paper)
    
    Args:
        activations: [N, num_layers, hidden_dim]
        labels: [N] - 0: poisoned, 1: clean
    """
    # Flatten activations
    if activations.dim() == 3:
        # Average across layers
        activations_flat = activations.mean(dim=1).numpy()
    else:
        activations_flat = activations.numpy()
    
    # t-SNE
    print("Computing t-SNE...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    embeddings_2d = tsne.fit_transform(activations_flat)
    
    # Plot
    plt.figure(figsize=(10, 8))
    
    labels_np = labels.numpy()
    colors = ['red' if l == 0 else 'blue' for l in labels_np]
    labels_text = ['Poisoned' if l == 0 else 'Correct' for l in labels_np]
    
    for label_val, color, text in [(0, 'red', 'Poisoned'), (1, 'blue', 'Correct')]:
        mask = labels_np == label_val
        plt.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=color,
            label=text,
            alpha=0.6,
            s=50
        )
    
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.title('t-SNE Visualization of LLM Activations')
    plt.legend()
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"t-SNE plot saved to {save_path}")
    
    plt.show()


def plot_training_history(
    history: Dict,
    save_path: Optional[str] = None
):
    """Plot training history"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Accuracy
    if 'val_accuracy' in history and history['val_accuracy']:
        axes[1].plot(history['val_accuracy'], label='Val Accuracy', color='green')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Validation Accuracy')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    
    plt.show()


def plot_detection_comparison(
    results: Dict[str, Dict],
    save_path: Optional[str] = None
):
    """
    Plot comparison của multiple methods (như Table 1 trong paper)
    
    Args:
        results: Dict mapping method_name -> metrics_dict
    """
    methods = list(results.keys())
    tprs = [results[m]['tpr'] for m in methods]
    fprs = [results[m]['fpr'] for m in methods]
    
    x = np.arange(len(methods))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.bar(x - width/2, tprs, width, label='TPR', color='green', alpha=0.7)
    ax.bar(x + width/2, fprs, width, label='FPR', color='red', alpha=0.7)
    
    ax.set_xlabel('Methods')
    ax.set_ylabel('Rate')
    ax.set_title('Detection Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (tpr, fpr) in enumerate(zip(tprs, fprs)):
        ax.text(i - width/2, tpr + 0.01, f'{tpr:.3f}', ha='center', va='bottom')
        ax.text(i + width/2, fpr + 0.01, f'{fpr:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")
    
    plt.show()
