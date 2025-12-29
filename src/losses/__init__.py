"""Loss functions and metrics for Virtual Health Assistant."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report
)
import logging

logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        """Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor
            gamma: Focusing parameter
            reduction: Reduction method
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.
        
        Args:
            inputs: Predicted logits
            targets: Target labels
            
        Returns:
            Focal loss
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingLoss(nn.Module):
    """Label smoothing loss for better generalization."""
    
    def __init__(self, smoothing: float = 0.1):
        """Initialize label smoothing loss.
        
        Args:
            smoothing: Smoothing factor
        """
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute label smoothing loss.
        
        Args:
            inputs: Predicted logits
            targets: Target labels
            
        Returns:
            Label smoothing loss
        """
        log_prob = F.log_softmax(inputs, dim=1)
        n_classes = inputs.size(1)
        
        # Create smoothed targets
        smoothed_targets = torch.zeros_like(log_prob)
        smoothed_targets.fill_(self.smoothing / (n_classes - 1))
        smoothed_targets.scatter_(1, targets.unsqueeze(1), 1 - self.smoothing)
        
        loss = -torch.sum(smoothed_targets * log_prob, dim=1)
        return loss.mean()


class CombinedLoss(nn.Module):
    """Combined loss for intent classification and entity recognition."""
    
    def __init__(
        self,
        intent_weight: float = 1.0,
        entity_weight: float = 1.0,
        use_focal: bool = True,
        use_label_smoothing: bool = True
    ):
        """Initialize combined loss.
        
        Args:
            intent_weight: Weight for intent loss
            entity_weight: Weight for entity loss
            use_focal: Whether to use focal loss
            use_label_smoothing: Whether to use label smoothing
        """
        super().__init__()
        self.intent_weight = intent_weight
        self.entity_weight = entity_weight
        
        # Intent loss
        if use_focal:
            self.intent_loss = FocalLoss()
        elif use_label_smoothing:
            self.intent_loss = LabelSmoothingLoss()
        else:
            self.intent_loss = nn.CrossEntropyLoss()
        
        # Entity loss
        self.entity_loss = nn.CrossEntropyLoss(ignore_index=-100)
    
    def forward(
        self,
        intent_logits: torch.Tensor,
        intent_targets: torch.Tensor,
        entity_logits: torch.Tensor,
        entity_targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute combined loss.
        
        Args:
            intent_logits: Intent prediction logits
            intent_targets: Intent target labels
            entity_logits: Entity prediction logits
            entity_targets: Entity target labels
            
        Returns:
            Dictionary containing individual and total losses
        """
        # Intent loss
        intent_loss = self.intent_loss(intent_logits, intent_targets)
        
        # Entity loss
        entity_loss = self.entity_loss(
            entity_logits.view(-1, entity_logits.size(-1)),
            entity_targets.view(-1)
        )
        
        # Total loss
        total_loss = (
            self.intent_weight * intent_loss +
            self.entity_weight * entity_loss
        )
        
        return {
            'total_loss': total_loss,
            'intent_loss': intent_loss,
            'entity_loss': entity_loss
        }


class MetricsCalculator:
    """Calculate comprehensive evaluation metrics."""
    
    def __init__(self, intent_labels: List[str], entity_labels: List[str]):
        """Initialize metrics calculator.
        
        Args:
            intent_labels: List of intent class labels
            entity_labels: List of entity class labels
        """
        self.intent_labels = intent_labels
        self.entity_labels = entity_labels
    
    def calculate_intent_metrics(
        self,
        y_true: List[int],
        y_pred: List[int],
        y_prob: Optional[List[List[float]]] = None
    ) -> Dict[str, float]:
        """Calculate intent classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        # Precision, recall, F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['f1'] = f1
        
        # Per-class metrics
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )
        metrics['precision_macro'] = precision_macro
        metrics['recall_macro'] = recall_macro
        metrics['f1_macro'] = f1_macro
        
        # AUC metrics if probabilities available
        if y_prob is not None:
            try:
                # Convert to numpy arrays
                y_true_np = np.array(y_true)
                y_prob_np = np.array(y_prob)
                
                # Multi-class AUC
                if len(np.unique(y_true)) > 2:
                    metrics['auroc'] = roc_auc_score(
                        y_true_np, y_prob_np, multi_class='ovr', average='weighted'
                    )
                    metrics['auprc'] = average_precision_score(
                        y_true_np, y_prob_np, average='weighted'
                    )
                else:
                    metrics['auroc'] = roc_auc_score(y_true_np, y_prob_np[:, 1])
                    metrics['auprc'] = average_precision_score(y_true_np, y_prob_np[:, 1])
            except Exception as e:
                logger.warning(f"Could not calculate AUC metrics: {e}")
                metrics['auroc'] = 0.0
                metrics['auprc'] = 0.0
        
        return metrics
    
    def calculate_entity_metrics(
        self,
        y_true: List[List[int]],
        y_pred: List[List[int]]
    ) -> Dict[str, float]:
        """Calculate entity recognition metrics.
        
        Args:
            y_true: True entity labels (list of sequences)
            y_pred: Predicted entity labels (list of sequences)
            
        Returns:
            Dictionary of metrics
        """
        # Flatten sequences
        y_true_flat = []
        y_pred_flat = []
        
        for true_seq, pred_seq in zip(y_true, y_pred):
            # Remove padding tokens (-100)
            for true_token, pred_token in zip(true_seq, pred_seq):
                if true_token != -100:
                    y_true_flat.append(true_token)
                    y_pred_flat.append(pred_token)
        
        if not y_true_flat:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true_flat, y_pred_flat, average='weighted', zero_division=0
        )
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def calculate_calibration_metrics(
        self,
        y_true: List[int],
        y_prob: List[List[float]],
        n_bins: int = 10
    ) -> Dict[str, float]:
        """Calculate calibration metrics.
        
        Args:
            y_true: True labels
            y_prob: Prediction probabilities
            n_bins: Number of bins for calibration
            
        Returns:
            Dictionary of calibration metrics
        """
        from sklearn.calibration import calibration_curve
        
        y_true_np = np.array(y_true)
        y_prob_np = np.array(y_prob)
        
        # Convert to binary for calibration curve
        if len(np.unique(y_true)) > 2:
            # Use max probability for multi-class
            max_probs = np.max(y_prob_np, axis=1)
            binary_true = (y_true_np == np.argmax(y_prob_np, axis=1)).astype(int)
        else:
            max_probs = y_prob_np[:, 1]
            binary_true = y_true_np
        
        # Calculate calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            binary_true, max_probs, n_bins=n_bins
        )
        
        # Expected Calibration Error (ECE)
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (max_probs > bin_lower) & (max_probs <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = binary_true[in_bin].mean()
                avg_confidence_in_bin = max_probs[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        # Brier Score
        brier_score = np.mean((max_probs - binary_true) ** 2)
        
        return {
            'ece': ece,
            'brier_score': brier_score,
            'calibration_curve': {
                'fraction_of_positives': fraction_of_positives.tolist(),
                'mean_predicted_value': mean_predicted_value.tolist()
            }
        }
    
    def calculate_fairness_metrics(
        self,
        y_true: List[int],
        y_pred: List[int],
        sensitive_attributes: Dict[str, List[Any]]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate fairness metrics across sensitive attributes.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            sensitive_attributes: Dictionary of sensitive attributes
            
        Returns:
            Dictionary of fairness metrics per attribute
        """
        fairness_metrics = {}
        
        for attr_name, attr_values in sensitive_attributes.items():
            attr_metrics = {}
            
            # Calculate metrics for each group
            unique_values = list(set(attr_values))
            group_metrics = {}
            
            for value in unique_values:
                mask = [v == value for v in attr_values]
                if sum(mask) > 0:
                    group_y_true = [y_true[i] for i in range(len(y_true)) if mask[i]]
                    group_y_pred = [y_pred[i] for i in range(len(y_pred)) if mask[i]]
                    
                    group_metrics[value] = {
                        'accuracy': accuracy_score(group_y_true, group_y_pred),
                        'precision': precision_recall_fscore_support(
                            group_y_true, group_y_pred, average='weighted', zero_division=0
                        )[0],
                        'recall': precision_recall_fscore_support(
                            group_y_true, group_y_pred, average='weighted', zero_division=0
                        )[1],
                        'f1': precision_recall_fscore_support(
                            group_y_true, group_y_pred, average='weighted', zero_division=0
                        )[2]
                    }
            
            # Calculate fairness gaps
            if len(group_metrics) > 1:
                accuracies = [metrics['accuracy'] for metrics in group_metrics.values()]
                precisions = [metrics['precision'] for metrics in group_metrics.values()]
                recalls = [metrics['recall'] for metrics in group_metrics.values()]
                f1s = [metrics['f1'] for metrics in group_metrics.values()]
                
                attr_metrics['accuracy_gap'] = max(accuracies) - min(accuracies)
                attr_metrics['precision_gap'] = max(precisions) - min(precisions)
                attr_metrics['recall_gap'] = max(recalls) - min(recalls)
                attr_metrics['f1_gap'] = max(f1s) - min(f1s)
            
            attr_metrics['group_metrics'] = group_metrics
            fairness_metrics[attr_name] = attr_metrics
        
        return fairness_metrics
