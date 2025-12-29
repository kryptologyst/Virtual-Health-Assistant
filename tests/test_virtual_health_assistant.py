"""Tests for Virtual Health Assistant."""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch

# Import our modules
import sys
sys.path.append('src')

from models import ClinicalBERTClassifier, TraditionalMLBaseline
from data import HealthDataProcessor, DeIdentifier, HealthIntent
from losses import FocalLoss, LabelSmoothingLoss, CombinedLoss
from metrics import MetricsCalculator
from utils import get_device, set_seed, Config


class TestDeIdentifier:
    """Test de-identification functionality."""
    
    def test_deidentify_phone(self):
        """Test phone number de-identification."""
        deid = DeIdentifier()
        text = "Call me at 555-123-4567"
        result = deid.deidentify(text)
        assert "[PHONE]" in result
        assert "555-123-4567" not in result
    
    def test_deidentify_email(self):
        """Test email de-identification."""
        deid = DeIdentifier()
        text = "Email me at john.doe@example.com"
        result = deid.deidentify(text)
        assert "[EMAIL]" in result
        assert "john.doe@example.com" not in result
    
    def test_deidentify_name(self):
        """Test name de-identification."""
        deid = DeIdentifier()
        text = "John Smith is my doctor"
        result = deid.deidentify(text)
        assert "[NAME]" in result
        assert "John Smith" not in result
    
    def test_identify_entities(self):
        """Test entity identification."""
        deid = DeIdentifier()
        text = "Call John Smith at 555-123-4567"
        entities = deid.identify_entities(text)
        
        assert len(entities) >= 2  # Should find phone and name
        entity_labels = [e['label'] for e in entities]
        assert 'PHONE' in entity_labels
        assert 'NAME' in entity_labels


class TestHealthDataProcessor:
    """Test health data processing."""
    
    def test_create_synthetic_dataset(self):
        """Test synthetic dataset creation."""
        processor = HealthDataProcessor()
        dataset = processor.create_synthetic_dataset(num_samples=10)
        
        assert len(dataset) == 10
        assert all(isinstance(item, HealthIntent) for item in dataset)
        assert all(item.text for item in dataset)
        assert all(item.intent for item in dataset)
    
    def test_split_dataset(self):
        """Test dataset splitting."""
        processor = HealthDataProcessor()
        dataset = processor.create_synthetic_dataset(num_samples=100)
        
        train, val, test = processor.split_dataset(dataset)
        
        assert len(train) + len(val) + len(test) == 100
        assert len(train) > len(val)
        assert len(train) > len(test)
    
    def test_extract_entities(self):
        """Test entity extraction."""
        processor = HealthDataProcessor()
        text = "I have a headache and take aspirin"
        entities = processor._extract_entities_from_text(text, "symptom_log")
        
        assert len(entities) > 0
        entity_labels = [e['label'] for e in entities]
        assert 'SYMPTOM' in entity_labels or 'MEDICATION' in entity_labels


class TestModels:
    """Test model implementations."""
    
    def test_clinicalbert_initialization(self):
        """Test ClinicalBERT model initialization."""
        model = ClinicalBERTClassifier(
            model_name="distilbert-base-uncased",  # Use smaller model for testing
            num_intents=5,
            num_entities=6,
            max_length=128
        )
        
        assert model.num_intents == 5
        assert model.num_entities == 6
        assert model.max_length == 128
    
    def test_traditional_ml_baseline(self):
        """Test traditional ML baseline."""
        baseline = TraditionalMLBaseline()
        
        # Test training
        texts = ["I have a headache", "Remind me to take medicine", "What is diabetes?"]
        labels = ["symptom_log", "med_reminder", "medical_question"]
        
        baseline.train(texts, labels)
        
        # Test prediction
        predictions = baseline.predict(["I feel sick"], "naive_bayes")
        assert len(predictions) == 1
        assert predictions[0] in labels


class TestLossFunctions:
    """Test loss functions."""
    
    def test_focal_loss(self):
        """Test focal loss calculation."""
        focal_loss = FocalLoss(alpha=1.0, gamma=2.0)
        
        # Create dummy inputs
        inputs = torch.randn(4, 3)  # batch_size=4, num_classes=3
        targets = torch.tensor([0, 1, 2, 0])
        
        loss = focal_loss(inputs, targets)
        assert loss.item() > 0
        assert loss.requires_grad
    
    def test_label_smoothing_loss(self):
        """Test label smoothing loss."""
        smooth_loss = LabelSmoothingLoss(smoothing=0.1)
        
        inputs = torch.randn(4, 3)
        targets = torch.tensor([0, 1, 2, 0])
        
        loss = smooth_loss(inputs, targets)
        assert loss.item() > 0
        assert loss.requires_grad
    
    def test_combined_loss(self):
        """Test combined loss."""
        combined_loss = CombinedLoss()
        
        # Create dummy inputs
        intent_logits = torch.randn(4, 5)
        intent_targets = torch.tensor([0, 1, 2, 3])
        entity_logits = torch.randn(4, 10, 6)
        entity_targets = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]] * 4)
        
        loss_dict = combined_loss(intent_logits, intent_targets, entity_logits, entity_targets)
        
        assert 'total_loss' in loss_dict
        assert 'intent_loss' in loss_dict
        assert 'entity_loss' in loss_dict
        assert loss_dict['total_loss'].item() > 0


class TestMetrics:
    """Test metrics calculation."""
    
    def test_intent_metrics(self):
        """Test intent classification metrics."""
        metrics_calc = MetricsCalculator(["a", "b", "c"], ["O", "A", "B"])
        
        y_true = [0, 1, 2, 0, 1]
        y_pred = [0, 1, 1, 0, 2]
        y_prob = [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.7, 0.2], [0.9, 0.05, 0.05], [0.1, 0.1, 0.8]]
        
        metrics = metrics_calc.calculate_intent_metrics(y_true, y_pred, y_prob)
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 0 <= metrics['accuracy'] <= 1
    
    def test_calibration_metrics(self):
        """Test calibration metrics."""
        metrics_calc = MetricsCalculator([], [])
        
        y_true = [0, 1, 0, 1, 0]
        y_prob = [[0.8, 0.2], [0.3, 0.7], [0.9, 0.1], [0.2, 0.8], [0.7, 0.3]]
        
        metrics = metrics_calc.calculate_calibration_metrics(y_true, y_prob)
        
        assert 'ece' in metrics
        assert 'brier_score' in metrics
        assert 0 <= metrics['ece'] <= 1
        assert 0 <= metrics['brier_score'] <= 1


class TestUtils:
    """Test utility functions."""
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        
        # Generate some random numbers
        torch_rand = torch.rand(1).item()
        np_rand = np.random.rand()
        
        # Reset seed and generate again
        set_seed(42)
        torch_rand2 = torch.rand(1).item()
        np_rand2 = np.random.rand()
        
        assert torch_rand == torch_rand2
        assert np_rand == np_rand2
    
    def test_get_device(self):
        """Test device detection."""
        device = get_device()
        assert isinstance(device, torch.device)
        assert device.type in ['cpu', 'cuda', 'mps']
    
    def test_config(self):
        """Test configuration management."""
        config = Config()
        
        # Test getting default values
        assert config.get('model.learning_rate') is not None
        assert config.get('nonexistent.key', 'default') == 'default'
        
        # Test updating values
        config.update('test.value', 42)
        assert config.get('test.value') == 42


# Integration tests
class TestIntegration:
    """Integration tests."""
    
    @pytest.mark.slow
    def test_end_to_end_prediction(self):
        """Test end-to-end prediction pipeline."""
        # This test requires actual model loading, so mark as slow
        device = get_device()
        
        # Create a simple model for testing
        model = ClinicalBERTClassifier(
            model_name="distilbert-base-uncased",
            num_intents=3,
            num_entities=4,
            max_length=128
        )
        model.to(device)
        
        # Test prediction
        text = "I have a headache"
        result = model.predict_with_uncertainty(text, device, num_samples=3)
        
        assert 'predicted_intent' in result
        assert 'confidence' in result
        assert 'uncertainty' in result
        assert 0 <= result['confidence'] <= 1
        assert 0 <= result['uncertainty'] <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
