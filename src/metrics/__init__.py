"""Training and evaluation scripts for Virtual Health Assistant."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from tqdm import tqdm
import json
from pathlib import Path
import time

from ..models import ClinicalBERTClassifier, ResponseGenerator, TraditionalMLBaseline
from ..losses import CombinedLoss, MetricsCalculator
from ..data import HealthIntent, HealthDataProcessor
from ..utils import get_device, set_seed, Config, ensure_dir

logger = logging.getLogger(__name__)


class HealthDataset(Dataset):
    """PyTorch dataset for health intents."""
    
    def __init__(
        self,
        data: List[HealthIntent],
        tokenizer,
        max_length: int = 512,
        intent_to_id: Optional[Dict[str, int]] = None,
        entity_to_id: Optional[Dict[str, int]] = None
    ):
        """Initialize dataset.
        
        Args:
            data: List of HealthIntent objects
            tokenizer: Tokenizer for text processing
            max_length: Maximum sequence length
            intent_to_id: Intent label to ID mapping
            entity_to_id: Entity label to ID mapping
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Create label mappings
        if intent_to_id is None:
            intents = list(set(item.intent for item in data))
            self.intent_to_id = {intent: i for i, intent in enumerate(sorted(intents))}
        else:
            self.intent_to_id = intent_to_id
        
        if entity_to_id is None:
            entities = ['O']  # Start with 'O' (outside)
            for item in data:
                for entity in item.entities:
                    if entity['label'] not in entities:
                        entities.append(entity['label'])
            self.entity_to_id = {entity: i for i, entity in enumerate(sorted(entities))}
        else:
            self.entity_to_id = entity_to_id
        
        self.id_to_intent = {v: k for k, v in self.intent_to_id.items()}
        self.id_to_entity = {v: k for k, v in self.entity_to_id.items()}
    
    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get item by index.
        
        Args:
            idx: Item index
            
        Returns:
            Dictionary containing tokenized data and labels
        """
        item = self.data[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            item.text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Get intent ID
        intent_id = self.intent_to_id[item.intent]
        
        # Create entity labels (simplified - just use first entity or 'O')
        entity_labels = [self.entity_to_id['O']] * self.max_length
        if item.entities:
            # For simplicity, mark first few tokens as first entity
            first_entity = item.entities[0]['label']
            if first_entity in self.entity_to_id:
                entity_labels[:min(len(item.text.split()), 5)] = [self.entity_to_id[first_entity]] * min(len(item.text.split()), 5)
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'intent_labels': torch.tensor(intent_id, dtype=torch.long),
            'entity_labels': torch.tensor(entity_labels, dtype=torch.long),
            'text': item.text,
            'intent': item.intent,
            'entities': item.entities
        }


class Trainer:
    """Trainer for Virtual Health Assistant models."""
    
    def __init__(
        self,
        model: nn.Module,
        config: Config,
        device: torch.device,
        tokenizer
    ):
        """Initialize trainer.
        
        Args:
            model: Model to train
            config: Configuration object
            device: Device to train on
            tokenizer: Tokenizer for the model
        """
        self.model = model
        self.config = config
        self.device = device
        self.tokenizer = tokenizer
        
        # Move model to device
        self.model.to(device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.get('model.learning_rate', 2e-5),
            weight_decay=config.get('training.weight_decay', 0.01)
        )
        
        # Initialize loss function
        self.criterion = CombinedLoss()
        
        # Initialize metrics calculator
        self.metrics_calc = MetricsCalculator(
            intent_labels=list(set()),  # Will be updated during training
            entity_labels=list(set())   # Will be updated during training
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': []
        }
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch.
        
        Args:
            dataloader: Training data loader
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        progress_bar = tqdm(dataloader, desc="Training")
        
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            intent_labels = batch['intent_labels'].to(self.device)
            entity_labels = batch['entity_labels'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_uncertainty=False
            )
            
            # Calculate loss
            loss_dict = self.criterion(
                outputs['intent_logits'],
                intent_labels,
                outputs['entity_logits'],
                entity_labels
            )
            
            # Backward pass
            loss_dict['total_loss'].backward()
            self.optimizer.step()
            
            # Accumulate metrics
            total_loss += loss_dict['total_loss'].item()
            
            # Get predictions
            intent_preds = torch.argmax(outputs['intent_logits'], dim=-1)
            all_predictions.extend(intent_preds.cpu().numpy())
            all_targets.extend(intent_labels.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss_dict['total_loss'].item():.4f}",
                'intent_loss': f"{loss_dict['intent_loss'].item():.4f}",
                'entity_loss': f"{loss_dict['entity_loss'].item():.4f}"
            })
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(dataloader)
        epoch_metrics = self.metrics_calc.calculate_intent_metrics(
            all_targets, all_predictions
        )
        epoch_metrics['loss'] = avg_loss
        
        return epoch_metrics
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate model.
        
        Args:
            dataloader: Validation data loader
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                intent_labels = batch['intent_labels'].to(self.device)
                entity_labels = batch['entity_labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_uncertainty=False
                )
                
                # Calculate loss
                loss_dict = self.criterion(
                    outputs['intent_logits'],
                    intent_labels,
                    outputs['entity_logits'],
                    entity_labels
                )
                
                total_loss += loss_dict['total_loss'].item()
                
                # Get predictions and probabilities
                intent_logits = outputs['intent_logits']
                intent_probs = torch.softmax(intent_logits, dim=-1)
                intent_preds = torch.argmax(intent_logits, dim=-1)
                
                all_predictions.extend(intent_preds.cpu().numpy())
                all_targets.extend(intent_labels.cpu().numpy())
                all_probabilities.extend(intent_probs.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(dataloader)
        metrics = self.metrics_calc.calculate_intent_metrics(
            all_targets, all_predictions, all_probabilities
        )
        metrics['loss'] = avg_loss
        
        return metrics
    
    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        num_epochs: Optional[int] = None
    ) -> Dict[str, List[float]]:
        """Train the model.
        
        Args:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            num_epochs: Number of epochs to train
            
        Returns:
            Training history
        """
        if num_epochs is None:
            num_epochs = self.config.get('model.num_epochs', 3)
        
        best_val_f1 = 0
        patience_counter = 0
        patience = self.config.get('training.early_stopping_patience', 3)
        
        logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            
            # Train
            train_metrics = self.train_epoch(train_dataloader)
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_metrics'].append(train_metrics)
            
            # Validate
            val_metrics = self.validate(val_dataloader)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_metrics'].append(val_metrics)
            
            logger.info(f"Train Loss: {train_metrics['loss']:.4f}, Val Loss: {val_metrics['loss']:.4f}")
            logger.info(f"Train F1: {train_metrics['f1']:.4f}, Val F1: {val_metrics['f1']:.4f}")
            
            # Early stopping
            if val_metrics['f1'] > best_val_f1:
                best_val_f1 = val_metrics['f1']
                patience_counter = 0
                
                # Save best model
                if self.config.get('training.save_best_model', True):
                    self.save_model('best_model.pt')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break
        
        return self.history
    
    def save_model(self, filename: str) -> None:
        """Save model checkpoint.
        
        Args:
            filename: Filename to save model
        """
        ensure_dir('models')
        model_path = Path('models') / filename
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config._config,
            'history': self.history
        }
        
        torch.save(checkpoint, model_path)
        logger.info(f"Model saved to {model_path}")
    
    def load_model(self, filename: str) -> None:
        """Load model checkpoint.
        
        Args:
            filename: Filename to load model from
        """
        model_path = Path('models') / filename
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file {model_path} not found")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        logger.info(f"Model loaded from {model_path}")


class Evaluator:
    """Evaluator for Virtual Health Assistant models."""
    
    def __init__(
        self,
        model: nn.Module,
        config: Config,
        device: torch.device,
        tokenizer
    ):
        """Initialize evaluator.
        
        Args:
            model: Model to evaluate
            config: Configuration object
            device: Device to evaluate on
            tokenizer: Tokenizer for the model
        """
        self.model = model
        self.config = config
        self.device = device
        self.tokenizer = tokenizer
        
        self.model.to(device)
        self.model.eval()
    
    def evaluate(
        self,
        dataloader: DataLoader,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """Evaluate model on test set.
        
        Args:
            dataloader: Test data loader
            save_results: Whether to save results to file
            
        Returns:
            Dictionary of evaluation results
        """
        logger.info("Starting evaluation")
        
        all_predictions = []
        all_targets = []
        all_probabilities = []
        all_texts = []
        all_intents = []
        all_entities = []
        all_uncertainties = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluation"):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                intent_labels = batch['intent_labels'].to(self.device)
                
                # Forward pass with uncertainty
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_uncertainty=True
                )
                
                # Get predictions and probabilities
                intent_logits = outputs['intent_logits']
                intent_probs = torch.softmax(intent_logits, dim=-1)
                intent_preds = torch.argmax(intent_logits, dim=-1)
                
                all_predictions.extend(intent_preds.cpu().numpy())
                all_targets.extend(intent_labels.cpu().numpy())
                all_probabilities.extend(intent_probs.cpu().numpy())
                all_texts.extend(batch['text'])
                all_intents.extend(batch['intent'])
                all_entities.extend(batch['entities'])
                all_uncertainties.extend(outputs['uncertainty'].cpu().numpy())
        
        # Calculate comprehensive metrics
        metrics_calc = MetricsCalculator([], [])  # Empty labels for now
        
        # Intent classification metrics
        intent_metrics = metrics_calc.calculate_intent_metrics(
            all_targets, all_predictions, all_probabilities
        )
        
        # Calibration metrics
        calibration_metrics = metrics_calc.calculate_calibration_metrics(
            all_targets, all_probabilities
        )
        
        # Compile results
        results = {
            'intent_metrics': intent_metrics,
            'calibration_metrics': calibration_metrics,
            'uncertainty_stats': {
                'mean': float(np.mean(all_uncertainties)),
                'std': float(np.std(all_uncertainties)),
                'min': float(np.min(all_uncertainties)),
                'max': float(np.max(all_uncertainties))
            },
            'predictions': {
                'texts': all_texts,
                'true_intents': all_intents,
                'predicted_intents': all_predictions,
                'probabilities': all_probabilities,
                'uncertainties': all_uncertainties
            }
        }
        
        # Save results if requested
        if save_results:
            ensure_dir('assets')
            results_path = Path('assets') / 'evaluation_results.json'
            
            # Convert numpy arrays to lists for JSON serialization
            json_results = self._prepare_for_json(results)
            
            with open(results_path, 'w') as f:
                json.dump(json_results, f, indent=2)
            
            logger.info(f"Results saved to {results_path}")
        
        logger.info("Evaluation completed")
        return results
    
    def _prepare_for_json(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare results for JSON serialization.
        
        Args:
            results: Results dictionary
            
        Returns:
            JSON-serializable results
        """
        json_results = {}
        
        for key, value in results.items():
            if isinstance(value, dict):
                json_results[key] = self._prepare_for_json(value)
            elif isinstance(value, (list, tuple)):
                json_results[key] = [
                    item.tolist() if hasattr(item, 'tolist') else item
                    for item in value
                ]
            elif hasattr(value, 'tolist'):
                json_results[key] = value.tolist()
            else:
                json_results[key] = value
        
        return json_results
