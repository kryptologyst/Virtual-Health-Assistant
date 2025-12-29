"""Core utilities for the Virtual Health Assistant project."""

import random
import numpy as np
import torch
from typing import Any, Dict, List, Optional, Union
import logging
import os
from pathlib import Path


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Get the best available device (CUDA -> MPS -> CPU).
    
    Returns:
        torch.device: The best available device
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Setup logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
        
    Returns:
        logging.Logger: Configured logger
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            *([logging.FileHandler(log_file)] if log_file else [])
        ]
    )
    return logging.getLogger(__name__)


class Config:
    """Configuration management using OmegaConf."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        if self.config_path and os.path.exists(self.config_path):
            try:
                from omegaconf import OmegaConf
                return OmegaConf.load(self.config_path)
            except ImportError:
                logging.warning("OmegaConf not available, using default config")
        
        return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "model": {
                "name": "clinicalbert",
                "max_length": 512,
                "num_intents": 5,
                "num_entities": 10,
                "dropout": 0.1,
                "learning_rate": 2e-5,
                "batch_size": 16,
                "num_epochs": 3
            },
            "data": {
                "train_split": 0.8,
                "val_split": 0.1,
                "test_split": 0.1,
                "max_samples": 1000,
                "deid_enabled": True
            },
            "training": {
                "device": "auto",
                "seed": 42,
                "gradient_accumulation_steps": 1,
                "warmup_steps": 100,
                "weight_decay": 0.01,
                "save_best_model": True,
                "early_stopping_patience": 3
            },
            "evaluation": {
                "metrics": ["accuracy", "f1", "precision", "recall", "auroc"],
                "calibration": True,
                "uncertainty": True,
                "explainability": True
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key."""
        keys = key.split(".")
        value = self._config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    def update(self, key: str, value: Any) -> None:
        """Update configuration value."""
        keys = key.split(".")
        config = self._config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value


def ensure_dir(path: Union[str, Path]) -> None:
    """Ensure directory exists, create if it doesn't.
    
    Args:
        path: Directory path
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def load_json(file_path: str) -> Dict[str, Any]:
    """Load JSON file.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Dict containing JSON data
    """
    import json
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Dict[str, Any], file_path: str) -> None:
    """Save data to JSON file.
    
    Args:
        data: Data to save
        file_path: Output file path
    """
    import json
    ensure_dir(os.path.dirname(file_path))
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def format_time(seconds: float) -> str:
    """Format time duration in human-readable format.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"
