"""Training script for Virtual Health Assistant."""

import argparse
import logging
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import json

from src.models import ClinicalBERTClassifier, TraditionalMLBaseline
from src.data import HealthDataProcessor, HealthDataset
from src.metrics import Trainer
from src.utils import Config, get_device, set_seed, setup_logging, ensure_dir

logger = logging.getLogger(__name__)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Virtual Health Assistant")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--data_path", type=str, default="data/synthetic_dataset.json",
                       help="Path to training data")
    parser.add_argument("--output_dir", type=str, default="models",
                       help="Output directory for models")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--log_level", type=str, default="INFO",
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger.info("Starting Virtual Health Assistant training")
    
    # Set random seed
    set_seed(args.seed)
    
    # Load configuration
    config = Config(args.config)
    logger.info(f"Loaded configuration from {args.config}")
    
    # Get device
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Prepare data
    logger.info("Preparing data...")
    data_processor = HealthDataProcessor(
        deid_enabled=config.get('data.deid_enabled', True)
    )
    
    # Load or create dataset
    if Path(args.data_path).exists():
        logger.info(f"Loading dataset from {args.data_path}")
        dataset = data_processor.load_dataset(args.data_path)
    else:
        logger.info("Creating synthetic dataset")
        dataset = data_processor.create_synthetic_dataset(
            num_samples=config.get('data.max_samples', 1000)
        )
        
        # Save dataset
        ensure_dir(Path(args.data_path).parent)
        data_processor.save_dataset(dataset, args.data_path)
    
    # Split dataset
    train_data, val_data, test_data = data_processor.split_dataset(
        dataset,
        train_ratio=config.get('data.train_split', 0.8),
        val_ratio=config.get('data.val_split', 0.1),
        test_split=config.get('data.test_split', 0.1)
    )
    
    logger.info(f"Dataset split: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
    
    # Initialize model
    logger.info("Initializing model...")
    model = ClinicalBERTClassifier(
        model_name=config.get('model.name', 'emilyalsentzer/Bio_ClinicalBERT'),
        num_intents=config.get('model.num_intents', 8),
        num_entities=config.get('model.num_entities', 9),
        dropout=config.get('model.dropout', 0.1),
        max_length=config.get('model.max_length', 512)
    )
    
    # Create datasets
    train_dataset = HealthDataset(
        train_data, model.tokenizer, config.get('model.max_length', 512)
    )
    val_dataset = HealthDataset(
        val_data, model.tokenizer, config.get('model.max_length', 512),
        intent_to_id=train_dataset.intent_to_id,
        entity_to_id=train_dataset.entity_to_id
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get('model.batch_size', 16),
        shuffle=True,
        num_workers=0  # Set to 0 for compatibility
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get('model.batch_size', 16),
        shuffle=False,
        num_workers=0
    )
    
    # Initialize trainer
    trainer = Trainer(model, config, device, model.tokenizer)
    
    # Train model
    logger.info("Starting training...")
    history = trainer.train(train_loader, val_loader)
    
    # Save final model
    ensure_dir(args.output_dir)
    trainer.save_model('final_model.pt')
    
    # Save training history
    history_path = Path(args.output_dir) / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    logger.info("Training completed successfully")
    
    # Train traditional ML baseline for comparison
    logger.info("Training traditional ML baseline...")
    baseline = TraditionalMLBaseline()
    
    # Prepare data for baseline
    train_texts = [item.text for item in train_data]
    train_labels = [item.intent for item in train_data]
    val_texts = [item.text for item in val_data]
    val_labels = [item.intent for item in val_data]
    
    # Train baseline
    baseline.train(train_texts, train_labels)
    
    # Evaluate baseline
    baseline_preds = baseline.predict(val_texts)
    baseline_probs = baseline.predict_proba(val_texts)
    
    # Calculate baseline metrics
    from sklearn.metrics import accuracy_score, f1_score
    baseline_accuracy = accuracy_score(val_labels, baseline_preds)
    baseline_f1 = f1_score(val_labels, baseline_preds, average='weighted')
    
    logger.info(f"Baseline accuracy: {baseline_accuracy:.4f}")
    logger.info(f"Baseline F1: {baseline_f1:.4f}")
    
    # Save baseline results
    baseline_results = {
        'accuracy': baseline_accuracy,
        'f1': baseline_f1,
        'predictions': baseline_preds,
        'probabilities': baseline_probs
    }
    
    baseline_path = Path(args.output_dir) / 'baseline_results.json'
    with open(baseline_path, 'w') as f:
        json.dump(baseline_results, f, indent=2)
    
    logger.info("All training completed successfully")


if __name__ == "__main__":
    main()
