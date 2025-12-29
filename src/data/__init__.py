"""Data processing utilities for the Virtual Health Assistant."""

import re
import json
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class HealthIntent:
    """Health intent data structure."""
    text: str
    intent: str
    entities: List[Dict[str, Any]]
    confidence: float = 1.0


class DeIdentifier:
    """De-identification utilities for healthcare text."""
    
    def __init__(self):
        """Initialize de-identifier with common patterns."""
        self.patterns = {
            'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'date': r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
            'name': r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # Simple name pattern
            'age': r'\b\d{1,3}\s*years?\s*old\b',
            'zipcode': r'\b\d{5}(-\d{4})?\b'
        }
        
        self.replacements = {
            'phone': '[PHONE]',
            'ssn': '[SSN]',
            'email': '[EMAIL]',
            'date': '[DATE]',
            'name': '[NAME]',
            'age': '[AGE]',
            'zipcode': '[ZIPCODE]'
        }
    
    def deidentify(self, text: str) -> str:
        """De-identify text by replacing sensitive patterns.
        
        Args:
            text: Input text
            
        Returns:
            De-identified text
        """
        deidentified_text = text
        
        for pattern_name, pattern in self.patterns.items():
            deidentified_text = re.sub(
                pattern, 
                self.replacements[pattern_name], 
                deidentified_text, 
                flags=re.IGNORECASE
            )
        
        return deidentified_text
    
    def identify_entities(self, text: str) -> List[Dict[str, Any]]:
        """Identify potential PHI entities in text.
        
        Args:
            text: Input text
            
        Returns:
            List of identified entities
        """
        entities = []
        
        for pattern_name, pattern in self.patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entities.append({
                    'text': match.group(),
                    'label': pattern_name.upper(),
                    'start': match.start(),
                    'end': match.end(),
                    'confidence': 0.9  # High confidence for regex matches
                })
        
        return entities


class HealthDataProcessor:
    """Data processor for health-related text data."""
    
    def __init__(self, deid_enabled: bool = True):
        """Initialize data processor.
        
        Args:
            deid_enabled: Whether to enable de-identification
        """
        self.deid_enabled = deid_enabled
        self.deidentifier = DeIdentifier() if deid_enabled else None
        
        # Define health intents
        self.intent_labels = [
            'symptom_log',
            'med_reminder', 
            'medical_question',
            'appointment_request',
            'medication_inquiry',
            'general_health',
            'emergency',
            'other'
        ]
        
        # Define medical entity types
        self.entity_labels = [
            'SYMPTOM',
            'MEDICATION',
            'CONDITION',
            'BODY_PART',
            'DOSAGE',
            'FREQUENCY',
            'DURATION',
            'SEVERITY',
            'O'
        ]
    
    def create_synthetic_dataset(self, num_samples: int = 1000) -> List[HealthIntent]:
        """Create synthetic health conversation dataset.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            List of HealthIntent objects
        """
        logger.info(f"Creating synthetic dataset with {num_samples} samples")
        
        # Sample templates for each intent
        templates = {
            'symptom_log': [
                "I have a {symptom} and feel {severity}",
                "My {body_part} hurts and I have {symptom}",
                "I'm experiencing {symptom} for {duration}",
                "I feel {symptom} and {symptom2}",
                "Having {symptom} since {duration}"
            ],
            'med_reminder': [
                "Remind me to take my {medication} at {time}",
                "Set a reminder for {medication} {frequency}",
                "I need to take {medication} {dosage} {frequency}",
                "Alert me for {medication} at {time}",
                "Schedule {medication} reminder for {time}"
            ],
            'medical_question': [
                "What are the symptoms of {condition}?",
                "Can you tell me about {condition}?",
                "What causes {condition}?",
                "How is {condition} treated?",
                "Is {condition} serious?"
            ],
            'appointment_request': [
                "I need to schedule an appointment",
                "Can I book a visit with {specialist}?",
                "I'd like to see a doctor about {condition}",
                "Schedule me for {date}",
                "I need a checkup"
            ],
            'medication_inquiry': [
                "What is {medication} used for?",
                "What are the side effects of {medication}?",
                "Can I take {medication} with {medication2}?",
                "How often should I take {medication}?",
                "Is {medication} safe?"
            ],
            'general_health': [
                "How can I improve my health?",
                "What should I eat for {condition}?",
                "How much exercise should I do?",
                "What vitamins should I take?",
                "How can I sleep better?"
            ],
            'emergency': [
                "I'm having chest pain",
                "I can't breathe properly",
                "I have severe {symptom}",
                "I need immediate help",
                "This is an emergency"
            ]
        }
        
        # Sample values for template filling
        symptoms = ['headache', 'nausea', 'fever', 'cough', 'fatigue', 'dizziness', 'pain']
        medications = ['aspirin', 'ibuprofen', 'acetaminophen', 'metformin', 'lisinopril', 'atorvastatin']
        conditions = ['diabetes', 'hypertension', 'flu', 'cold', 'migraine', 'arthritis']
        body_parts = ['head', 'chest', 'back', 'stomach', 'throat', 'joints']
        severities = ['mild', 'moderate', 'severe', 'intense', 'unbearable']
        durations = ['2 days', 'a week', '3 hours', 'since yesterday', 'for months']
        frequencies = ['daily', 'twice daily', 'every 4 hours', 'weekly', 'as needed']
        dosages = ['10mg', '500mg', '1 tablet', '2 capsules', '5ml']
        times = ['8am', 'noon', '6pm', 'bedtime', 'morning']
        specialists = ['cardiologist', 'dermatologist', 'neurologist', 'orthopedist']
        dates = ['Monday', 'next week', 'tomorrow', 'Friday', 'next month']
        
        import random
        
        dataset = []
        for i in range(num_samples):
            # Select random intent
            intent = random.choice(list(templates.keys()))
            template = random.choice(templates[intent])
            
            # Fill template with random values
            text = template.format(
                symptom=random.choice(symptoms),
                symptom2=random.choice(symptoms),
                medication=random.choice(medications),
                medication2=random.choice(medications),
                condition=random.choice(conditions),
                body_part=random.choice(body_parts),
                severity=random.choice(severities),
                duration=random.choice(durations),
                frequency=random.choice(frequencies),
                dosage=random.choice(dosages),
                time=random.choice(times),
                specialist=random.choice(specialists),
                date=random.choice(dates)
            )
            
            # Extract entities from the text
            entities = self._extract_entities_from_text(text, intent)
            
            # Apply de-identification if enabled
            if self.deid_enabled and self.deidentifier:
                text = self.deidentifier.deidentify(text)
            
            dataset.append(HealthIntent(
                text=text,
                intent=intent,
                entities=entities,
                confidence=random.uniform(0.7, 1.0)
            ))
        
        logger.info(f"Created dataset with {len(dataset)} samples")
        return dataset
    
    def _extract_entities_from_text(self, text: str, intent: str) -> List[Dict[str, Any]]:
        """Extract entities from text based on patterns.
        
        Args:
            text: Input text
            intent: Intent of the text
            
        Returns:
            List of extracted entities
        """
        entities = []
        
        # Simple pattern-based entity extraction
        patterns = {
            'SYMPTOM': r'\b(headache|nausea|fever|cough|fatigue|dizziness|pain|chest pain)\b',
            'MEDICATION': r'\b(aspirin|ibuprofen|acetaminophen|metformin|lisinopril|atorvastatin)\b',
            'CONDITION': r'\b(diabetes|hypertension|flu|cold|migraine|arthritis)\b',
            'BODY_PART': r'\b(head|chest|back|stomach|throat|joints)\b',
            'DOSAGE': r'\b(\d+mg|\d+\s+tablet|\d+\s+capsule|\d+ml)\b',
            'FREQUENCY': r'\b(daily|twice daily|every \d+ hours|weekly|as needed)\b',
            'DURATION': r'\b(\d+ days?|a week|\d+ hours?|since yesterday|for months?)\b',
            'SEVERITY': r'\b(mild|moderate|severe|intense|unbearable)\b'
        }
        
        for label, pattern in patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entities.append({
                    'text': match.group(),
                    'label': label,
                    'start': match.start(),
                    'end': match.end(),
                    'confidence': 0.8
                })
        
        return entities
    
    def split_dataset(self, dataset: List[HealthIntent], train_ratio: float = 0.8, 
                     val_ratio: float = 0.1, test_ratio: float = 0.1) -> Tuple[List[HealthIntent], List[HealthIntent], List[HealthIntent]]:
        """Split dataset into train/validation/test sets.
        
        Args:
            dataset: Full dataset
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            
        Returns:
            Tuple of (train, val, test) datasets
        """
        import random
        random.shuffle(dataset)
        
        n = len(dataset)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train_data = dataset[:train_end]
        val_data = dataset[train_end:val_end]
        test_data = dataset[val_end:]
        
        logger.info(f"Split dataset: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
        return train_data, val_data, test_data
    
    def save_dataset(self, dataset: List[HealthIntent], file_path: str) -> None:
        """Save dataset to JSON file.
        
        Args:
            dataset: Dataset to save
            file_path: Output file path
        """
        data = []
        for item in dataset:
            data.append({
                'text': item.text,
                'intent': item.intent,
                'entities': item.entities,
                'confidence': item.confidence
            })
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved dataset to {file_path}")
    
    def load_dataset(self, file_path: str) -> List[HealthIntent]:
        """Load dataset from JSON file.
        
        Args:
            file_path: Input file path
            
        Returns:
            Loaded dataset
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        dataset = []
        for item in data:
            dataset.append(HealthIntent(
                text=item['text'],
                intent=item['intent'],
                entities=item['entities'],
                confidence=item.get('confidence', 1.0)
            ))
        
        logger.info(f"Loaded dataset from {file_path}")
        return dataset
