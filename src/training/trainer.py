import torch
from torch.utils.data import DataLoader
from typing import Dict, Optional
from tqdm import tqdm
import numpy as np
from pathlib import Path
from models.detector import RevPRAGDetector
from training.losses import CombinedLoss
import torch.nn as nn

class Trainer:
    """Trainer RevPRAG model"""
    
    def __init__(
        self,
        model: RevPRAGDetector,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = "cuda",
        output_dir: str = "./outputs"
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'learning_rate': []
        }
        
        self.best_val_loss = float('inf')
        self.patience_counter = 0
    
    def train_epoch(self) -> Dict:

        self.model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch in pbar:
            # Get triplets
            anchor = batch['anchor'].to(self.device)
            positive = batch['positive'].to(self.device)
            negative = batch['negative'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward
            anchor_emb = self.model(anchor)
            positive_emb = self.model(positive)
            negative_emb = self.model(negative)
            
            # Compute loss
            if isinstance(self.criterion, CombinedLoss):
                loss_dict = self.criterion(anchor_emb, positive_emb, negative_emb, labels)
                loss = loss_dict['total']
            else:
                loss = self.criterion(anchor_emb, positive_emb, negative_emb)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Update stats
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / num_batches
        return {'loss': avg_loss}
    
    def validate(self, support_loader: Optional[DataLoader] = None) -> Dict:
        """Validate model"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        # Set support set if provided
        if support_loader is not None:
            support_acts = []
            support_labels = []
            for batch in support_loader:
                support_acts.append(batch['activation'])
                support_labels.append(batch['label'])
            
            support_acts = torch.cat(support_acts, dim=0).to(self.device)
            support_labels = torch.cat(support_labels, dim=0).to(self.device)
            self.model.set_support_set(support_acts, support_labels)
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                anchor = batch['anchor'].to(self.device)
                positive = batch['positive'].to(self.device)
                negative = batch['negative'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward
                anchor_emb = self.model(anchor)
                positive_emb = self.model(positive)
                negative_emb = self.model(negative)
                
                # Compute loss
                if isinstance(self.criterion, CombinedLoss):
                    loss_dict = self.criterion(anchor_emb, positive_emb, negative_emb, labels)
                    loss = loss_dict['total']
                else:
                    loss = self.criterion(anchor_emb, positive_emb, negative_emb)
                
                total_loss += loss.item()
                num_batches += 1
                
                # Compute predictions nếu support set configured
                if self.model.support_embeddings is not None:
                    # Get predictions from anchor
                    activations = batch['activation'].to(self.device)
                    results = self.model.detect(activations)
                    
                    if isinstance(results, list):
                        preds = [r['label'] for r in results]
                    else:
                        preds = [results['label']]
                    
                    all_predictions.extend(preds)
                    all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / num_batches
        
        result = {'loss': avg_loss}
        
        # Compute accuracy
        if all_predictions:
            accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))
            result['accuracy'] = accuracy
        
        return result
    
    def train(
        self,
        num_epochs: int,
        support_loader: Optional[DataLoader] = None,
        early_stopping_patience: int = 10,
        save_best: bool = True
    ):
        """Full training loop"""
        print(f"Starting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate(support_loader)
            
            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            if 'accuracy' in val_metrics:
                self.history['val_accuracy'].append(val_metrics['accuracy'])
            
            if self.scheduler is not None:
                self.scheduler.step()
                self.history['learning_rate'].append(
                    self.optimizer.param_groups[0]['lr']
                )
            
            # Print metrics
            print(f"Train Loss: {train_metrics['loss']:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}")
            if 'accuracy' in val_metrics:
                print(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
            
            # Save best model
            if save_best and val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.save_checkpoint('best_model.pt')
                print(f"Saved best model (val_loss: {self.best_val_loss:.4f})")
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
            
            # Save checkpoint 
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pt')
        
        print("\nTraining completed!")
        self.save_history()
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'best_val_loss': self.best_val_loss
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        path = self.output_dir / filename
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint"""
        path = self.output_dir / filename
        checkpoint = torch.load(path)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        self.best_val_loss = checkpoint['best_val_loss']
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"Checkpoint loaded from {path}")
    
    def save_history(self):
        """Save training history"""
        import json
        path = self.output_dir / 'training_history.json'
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"Training history saved to {path}")
