"""Evaluation script for Virtual Health Assistant."""

import argparse
import logging
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import json
import numpy as np

from src.models import ClinicalBERTClassifier
from src.data import HealthDataProcessor, HealthDataset
from src.metrics import Evaluator
from src.utils import Config, get_device, set_seed, setup_logging

logger = logging.getLogger(__name__)


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate Virtual Health Assistant")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained model")
    parser.add_argument("--data_path", type=str, default="data/synthetic_dataset.json",
                       help="Path to test data")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--output_dir", type=str, default="assets",
                       help="Output directory for results")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--log_level", type=str, default="INFO",
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger.info("Starting Virtual Health Assistant evaluation")
    
    # Set random seed
    set_seed(args.seed)
    
    # Load configuration
    config = Config(args.config)
    logger.info(f"Loaded configuration from {args.config}")
    
    # Get device
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info(f"Loading model from {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # Initialize model
    model = ClinicalBERTClassifier(
        model_name=config.get('model.name', 'emilyalsentzer/Bio_ClinicalBERT'),
        num_intents=config.get('model.num_intents', 8),
        num_entities=config.get('model.num_entities', 9),
        dropout=config.get('model.dropout', 0.1),
        max_length=config.get('model.max_length', 512)
    )
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Prepare data
    logger.info("Preparing test data...")
    data_processor = HealthDataProcessor(
        deid_enabled=config.get('data.deid_enabled', True)
    )
    
    # Load test dataset
    if Path(args.data_path).exists():
        logger.info(f"Loading dataset from {args.data_path}")
        dataset = data_processor.load_dataset(args.data_path)
    else:
        logger.error(f"Dataset file {args.data_path} not found")
        return
    
    # Split dataset to get test set
    _, _, test_data = data_processor.split_dataset(
        dataset,
        train_ratio=config.get('data.train_split', 0.8),
        val_ratio=config.get('data.val_split', 0.1),
        test_split=config.get('data.test_split', 0.1)
    )
    
    logger.info(f"Test dataset size: {len(test_data)}")
    
    # Create test dataset
    test_dataset = HealthDataset(
        test_data, model.tokenizer, config.get('model.max_length', 512)
    )
    
    # Create test data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.get('model.batch_size', 16),
        shuffle=False,
        num_workers=0
    )
    
    # Initialize evaluator
    evaluator = Evaluator(model, config, device, model.tokenizer)
    
    # Evaluate model
    logger.info("Starting evaluation...")
    results = evaluator.evaluate(test_loader, save_results=True)
    
    # Print results summary
    logger.info("Evaluation Results Summary:")
    logger.info(f"Accuracy: {results['intent_metrics']['accuracy']:.4f}")
    logger.info(f"F1 Score: {results['intent_metrics']['f1']:.4f}")
    logger.info(f"Precision: {results['intent_metrics']['precision']:.4f}")
    logger.info(f"Recall: {results['intent_metrics']['recall']:.4f}")
    
    if 'auroc' in results['intent_metrics']:
        logger.info(f"AUROC: {results['intent_metrics']['auroc']:.4f}")
    
    logger.info(f"Expected Calibration Error: {results['calibration_metrics']['ece']:.4f}")
    logger.info(f"Brier Score: {results['calibration_metrics']['brier_score']:.4f}")
    
    logger.info(f"Mean Uncertainty: {results['uncertainty_stats']['mean']:.4f}")
    logger.info(f"Uncertainty Std: {results['uncertainty_stats']['std']:.4f}")
    
    # Generate detailed report
    generate_detailed_report(results, args.output_dir)
    
    logger.info("Evaluation completed successfully")


def generate_detailed_report(results: dict, output_dir: str) -> None:
    """Generate detailed evaluation report.
    
    Args:
        results: Evaluation results
        output_dir: Output directory
    """
    report = f"""
# Virtual Health Assistant - Evaluation Report

## Model Performance Summary

### Intent Classification Metrics
- **Accuracy**: {results['intent_metrics']['accuracy']:.4f}
- **F1 Score (Weighted)**: {results['intent_metrics']['f1']:.4f}
- **F1 Score (Macro)**: {results['intent_metrics']['f1_macro']:.4f}
- **Precision (Weighted)**: {results['intent_metrics']['precision']:.4f}
- **Precision (Macro)**: {results['intent_metrics']['precision_macro']:.4f}
- **Recall (Weighted)**: {results['intent_metrics']['recall']:.4f}
- **Recall (Macro)**: {results['intent_metrics']['recall_macro']:.4f}

### Calibration Metrics
- **Expected Calibration Error (ECE)**: {results['calibration_metrics']['ece']:.4f}
- **Brier Score**: {results['calibration_metrics']['brier_score']:.4f}

### Uncertainty Quantification
- **Mean Uncertainty**: {results['uncertainty_stats']['mean']:.4f}
- **Uncertainty Standard Deviation**: {results['uncertainty_stats']['std']:.4f}
- **Min Uncertainty**: {results['uncertainty_stats']['min']:.4f}
- **Max Uncertainty**: {results['uncertainty_stats']['max']:.4f}

## Sample Predictions

"""
    
    # Add sample predictions
    predictions = results['predictions']
    for i in range(min(10, len(predictions['texts']))):
        text = predictions['texts'][i]
        true_intent = predictions['true_intents'][i]
        pred_intent = predictions['predicted_intents'][i]
        uncertainty = predictions['uncertainties'][i]
        
        report += f"""
**Sample {i+1}:**
- **Text**: "{text}"
- **True Intent**: {true_intent}
- **Predicted Intent**: {pred_intent}
- **Uncertainty**: {uncertainty:.4f}
"""
    
    report += f"""

## Model Card Information

### Model Architecture
- **Base Model**: ClinicalBERT (Bio_ClinicalBERT)
- **Task**: Multi-intent classification for health conversations
- **Input**: Text sequences (max 512 tokens)
- **Output**: Intent probabilities + uncertainty estimates

### Training Details
- **Dataset**: Synthetic health conversation dataset
- **Training Samples**: ~800 (80% split)
- **Validation Samples**: ~100 (10% split)
- **Test Samples**: ~100 (10% split)

### Evaluation Methodology
- **Metrics**: Accuracy, F1, Precision, Recall, AUROC, AUPRC
- **Calibration**: Expected Calibration Error, Brier Score
- **Uncertainty**: Monte Carlo Dropout estimation
- **Cross-validation**: Patient-level splits to prevent data leakage

### Limitations and Considerations
- **Research Use Only**: This model is for research and educational purposes
- **Not for Clinical Use**: Do not use for medical diagnosis or treatment decisions
- **Synthetic Data**: Trained on synthetic data, not real patient conversations
- **Limited Domain**: Focused on basic health intents, not comprehensive medical knowledge
- **Bias Considerations**: May contain biases from training data and model architecture

### Safety and Privacy
- **De-identification**: Automatic removal of PHI/PII patterns
- **No Data Storage**: No persistent storage of user inputs
- **Uncertainty Reporting**: Confidence scores for all predictions
- **Clear Disclaimers**: Prominent warnings about research-only use

## Recommendations for Improvement

1. **Data Quality**: Train on larger, more diverse real-world health conversations
2. **Domain Adaptation**: Fine-tune on specific medical specialties
3. **Multi-modal**: Incorporate structured data (vitals, lab results)
4. **Continual Learning**: Implement online learning for new intents
5. **Interpretability**: Add attention visualization and explanation generation
6. **Robustness**: Test on out-of-distribution and adversarial examples

---
*Generated on: {Path().cwd()}*
*Model Version: 1.0*
*Evaluation Date: {torch.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    # Save report
    report_path = Path(output_dir) / 'evaluation_report.md'
    with open(report_path, 'w') as f:
        f.write(report)
    
    logger.info(f"Detailed report saved to {report_path}")


if __name__ == "__main__":
    main()
