# Virtual Health Assistant - NLP Research Project

**RESEARCH DEMO - NOT FOR CLINICAL USE**

A research-ready virtual health assistant built with state-of-the-art NLP models for intent classification, named entity recognition, and conversational AI in healthcare contexts.

## ⚠️ DISCLAIMER

**This is a research demonstration project only. Do not use for clinical purposes or medical advice. Always consult qualified healthcare professionals for medical concerns.**

## Features

- **Intent Classification**: Multi-intent recognition for health-related queries
- **Named Entity Recognition**: Extract medical entities (symptoms, medications, conditions)
- **Conversational AI**: Context-aware response generation
- **De-identification**: Privacy-preserving text processing
- **Explainability**: Attention visualization and model interpretability
- **Uncertainty Quantification**: Confidence scoring for responses
- **Interactive Demo**: Streamlit-based chat interface

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the interactive demo
streamlit run demo/app.py

# Train models
python scripts/train.py --config configs/base_config.yaml

# Evaluate models
python scripts/evaluate.py --model_path models/best_model.pt
```

## Project Structure

```
├── src/                    # Source code
│   ├── models/            # Model implementations
│   ├── data/              # Data processing utilities
│   ├── losses/            # Loss functions
│   ├── metrics/           # Evaluation metrics
│   ├── utils/             # Utility functions
│   ├── train.py           # Training script
│   └── eval.py            # Evaluation script
├── configs/               # Configuration files
├── data/                  # Data storage
├── models/                # Model checkpoints
├── scripts/               # Training/evaluation scripts
├── demo/                  # Interactive demo
├── tests/                 # Unit tests
├── assets/                # Visualizations and outputs
└── notebooks/             # Jupyter notebooks
```

## Models

- **Intent Classification**: ClinicalBERT + Multi-class classifier
- **NER**: BioBERT + CRF layer for medical entity extraction
- **Response Generation**: T5-based seq2seq for contextual responses
- **Baseline**: Traditional ML (Naive Bayes, SVM) for comparison

## Evaluation Metrics

- **Intent Classification**: Accuracy, F1-score, AUROC
- **NER**: Micro/Macro F1, Precision, Recall
- **Response Quality**: ROUGE-L, BERTScore, Semantic Similarity
- **Calibration**: Expected Calibration Error (ECE)
- **Fairness**: Performance across demographic groups

## Data

Synthetic dataset includes:
- Health-related intents (symptoms, medications, questions)
- Medical entity annotations
- De-identified patient conversations
- Multi-turn dialogue examples

## Safety & Privacy

- Automatic de-identification of PHI/PII
- No persistent storage of personal data
- Bias detection and mitigation
- Uncertainty quantification
- Clear disclaimers and warnings

## Requirements

- Python 3.10+
- PyTorch 2.0+
- Transformers 4.30+
- Streamlit for demo interface

## License

MIT License - See LICENSE file for details

## Contributing

This is a research demonstration project. Contributions should focus on:
- Educational improvements
- Research methodology enhancements
- Technical documentation
- Safety and privacy features

## Citation

If you use this project in research, please cite:

```bibtex
@software{virtual_health_assistant_2024,
  title={Virtual Health Assistant: NLP Research Demonstration},
  author={Kryptologyst},
  year={2025},
  url={https://github.com/kryptologyst/Virtual-Health-Assistant}
}
```
# Virtual-Health-Assistant
