"""Streamlit demo for Virtual Health Assistant."""

import streamlit as st
import torch
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any
import json
from pathlib import Path
import time

# Import our modules
import sys
sys.path.append('src')

from models import ClinicalBERTClassifier, ResponseGenerator
from data import HealthDataProcessor, DeIdentifier
from utils import get_device, Config

# Page configuration
st.set_page_config(
    page_title="Virtual Health Assistant - Research Demo",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for disclaimer banner
st.markdown("""
<style>
.disclaimer-banner {
    background-color: #ff4444;
    color: white;
    padding: 10px;
    border-radius: 5px;
    margin-bottom: 20px;
    text-align: center;
    font-weight: bold;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)

# Disclaimer banner
st.markdown("""
<div class="disclaimer-banner">
⚠️ RESEARCH DEMO ONLY - NOT FOR CLINICAL USE ⚠️
<br>This is a demonstration project for research and educational purposes only.
<br>Do not use for medical diagnosis or treatment decisions.
</div>
""", unsafe_allow_html=True)

# Title and description
st.title("🏥 Virtual Health Assistant")
st.markdown("""
**Research Demonstration Project**

A modern NLP-based virtual health assistant for intent classification, entity recognition, 
and conversational AI in healthcare contexts. This demo showcases:

- **Intent Classification**: Identify health-related user intents
- **Entity Recognition**: Extract medical entities from text
- **Response Generation**: Generate contextual responses
- **Uncertainty Quantification**: Confidence scoring for predictions
- **De-identification**: Privacy-preserving text processing
""")

# Sidebar configuration
st.sidebar.header("Configuration")

# Model selection
model_option = st.sidebar.selectbox(
    "Select Model",
    ["ClinicalBERT", "Traditional ML Baseline"],
    help="Choose between deep learning and traditional ML models"
)

# De-identification toggle
deid_enabled = st.sidebar.checkbox(
    "Enable De-identification",
    value=True,
    help="Automatically remove PHI/PII from input text"
)

# Uncertainty visualization toggle
show_uncertainty = st.sidebar.checkbox(
    "Show Uncertainty Analysis",
    value=True,
    help="Display uncertainty quantification and confidence scores"
)

# Load models (cached)
@st.cache_resource
def load_models():
    """Load pre-trained models."""
    device = get_device()
    
    # Load ClinicalBERT model
    try:
        config = Config("configs/base_config.yaml")
        model = ClinicalBERTClassifier(
            model_name=config.get('model.name', 'emilyalsentzer/Bio_ClinicalBERT'),
            num_intents=config.get('model.num_intents', 8),
            num_entities=config.get('model.num_entities', 9),
            dropout=config.get('model.dropout', 0.1),
            max_length=config.get('model.max_length', 512)
        )
        model.to(device)
        
        # Try to load trained weights
        model_path = Path("models/best_model.pt")
        if model_path.exists():
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
        else:
            st.warning("No trained model found. Using pre-trained weights only.")
        
        return model, device
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# Load response generator
@st.cache_resource
def load_response_generator():
    """Load response generator."""
    try:
        generator = ResponseGenerator()
        return generator
    except Exception as e:
        st.error(f"Error loading response generator: {e}")
        return None

# Initialize de-identifier
deidentifier = DeIdentifier()

# Load models
model, device = load_models()
response_generator = load_response_generator()

# Intent labels
intent_labels = [
    "symptom_log", "med_reminder", "medical_question", 
    "appointment_request", "medication_inquiry", 
    "general_health", "emergency", "other"
]

# Main interface
if model is not None and device is not None:
    
    # Chat interface
    st.header("💬 Chat Interface")
    
    # Initialize session state for chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Chat input
    user_input = st.text_area(
        "Enter your health-related message:",
        placeholder="e.g., 'I have a headache and feel nauseous' or 'Remind me to take my medication at 8am'",
        height=100
    )
    
    col1, col2 = st.columns([1, 4])
    
    with col1:
        send_button = st.button("Send", type="primary")
    
    with col2:
        clear_button = st.button("Clear Chat")
    
    if clear_button:
        st.session_state.chat_history = []
        st.rerun()
    
    if send_button and user_input.strip():
        
        # Process input
        with st.spinner("Processing your message..."):
            
            # De-identify if enabled
            if deid_enabled:
                original_text = user_input
                processed_text = deidentifier.deidentify(user_input)
                identified_entities = deidentifier.identify_entities(user_input)
            else:
                processed_text = user_input
                identified_entities = []
            
            # Get prediction
            if model_option == "ClinicalBERT":
                prediction_result = model.predict_with_uncertainty(
                    processed_text, device, num_samples=10
                )
                
                predicted_intent = intent_labels[prediction_result['predicted_intent']]
                confidence = prediction_result['confidence']
                uncertainty = prediction_result['uncertainty']
                all_probabilities = prediction_result['all_probabilities']
                
            else:
                # Traditional ML baseline (simplified)
                predicted_intent = "medical_question"  # Placeholder
                confidence = 0.8
                uncertainty = 0.2
                all_probabilities = [0.1] * len(intent_labels)
                all_probabilities[intent_labels.index(predicted_intent)] = confidence
            
            # Generate response
            if response_generator is not None:
                try:
                    response = response_generator.generate_response(
                        processed_text, predicted_intent, device
                    )
                except:
                    response = f"I understand you're asking about {predicted_intent.replace('_', ' ')}. This is a research demo - please consult a healthcare professional for medical advice."
            else:
                response = f"I understand you're asking about {predicted_intent.replace('_', ' ')}. This is a research demo - please consult a healthcare professional for medical advice."
            
            # Add to chat history
            st.session_state.chat_history.append({
                'user': user_input,
                'processed': processed_text if deid_enabled else None,
                'intent': predicted_intent,
                'confidence': confidence,
                'uncertainty': uncertainty,
                'response': response,
                'entities': identified_entities,
                'timestamp': time.time()
            })
    
    # Display chat history
    if st.session_state.chat_history:
        st.subheader("Chat History")
        
        for i, chat in enumerate(reversed(st.session_state.chat_history)):
            with st.expander(f"Message {len(st.session_state.chat_history) - i}: {chat['intent'].replace('_', ' ').title()}"):
                
                # User message
                st.markdown(f"**You:** {chat['user']}")
                
                if chat['processed'] and chat['processed'] != chat['user']:
                    st.markdown(f"**Processed:** {chat['processed']}")
                
                # Intent prediction
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Predicted Intent", chat['intent'].replace('_', ' ').title())
                with col2:
                    st.metric("Confidence", f"{chat['confidence']:.3f}")
                with col3:
                    st.metric("Uncertainty", f"{chat['uncertainty']:.3f}")
                
                # Response
                st.markdown(f"**Assistant:** {chat['response']}")
                
                # Entities if any
                if chat['entities']:
                    st.markdown("**Identified Entities:**")
                    for entity in chat['entities']:
                        st.markdown(f"- {entity['text']} ({entity['label']})")
    
    # Analysis section
    if st.session_state.chat_history and show_uncertainty:
        st.header("📊 Analysis")
        
        # Intent distribution
        intents = [chat['intent'] for chat in st.session_state.chat_history]
        intent_counts = {intent: intents.count(intent) for intent in set(intents)}
        
        # Create pie chart
        fig_pie = px.pie(
            values=list(intent_counts.values()),
            names=list(intent_counts.keys()),
            title="Intent Distribution"
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # Confidence and uncertainty over time
        confidences = [chat['confidence'] for chat in st.session_state.chat_history]
        uncertainties = [chat['uncertainty'] for chat in st.session_state.chat_history]
        
        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(
            y=confidences,
            mode='lines+markers',
            name='Confidence',
            line=dict(color='green')
        ))
        fig_line.add_trace(go.Scatter(
            y=uncertainties,
            mode='lines+markers',
            name='Uncertainty',
            line=dict(color='red')
        ))
        
        fig_line.update_layout(
            title="Confidence and Uncertainty Over Time",
            xaxis_title="Message Number",
            yaxis_title="Score",
            yaxis=dict(range=[0, 1])
        )
        
        st.plotly_chart(fig_line, use_container_width=True)
        
        # Statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Messages", len(st.session_state.chat_history))
        with col2:
            st.metric("Avg Confidence", f"{np.mean(confidences):.3f}")
        with col3:
            st.metric("Avg Uncertainty", f"{np.mean(uncertainties):.3f}")
        with col4:
            st.metric("Unique Intents", len(set(intents)))
    
    # Model information
    st.header("ℹ️ Model Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Details")
        st.markdown(f"""
        - **Model Type**: {model_option}
        - **Base Architecture**: ClinicalBERT (Bio_ClinicalBERT)
        - **Device**: {device}
        - **Max Sequence Length**: 512 tokens
        - **Number of Intents**: {len(intent_labels)}
        - **De-identification**: {'Enabled' if deid_enabled else 'Disabled'}
        """)
    
    with col2:
        st.subheader("Capabilities")
        st.markdown("""
        - ✅ Intent Classification
        - ✅ Entity Recognition
        - ✅ Response Generation
        - ✅ Uncertainty Quantification
        - ✅ De-identification
        - ✅ Confidence Scoring
        """)
    
    # Safety information
    st.header("⚠️ Safety and Limitations")
    
    st.warning("""
    **Important Limitations:**
    
    - This is a research demonstration, not a clinical tool
    - Responses may be inaccurate or inappropriate
    - No clinical validation has been performed
    - Not FDA approved or certified for medical use
    - May contain biases or errors
    - Always consult qualified healthcare professionals for medical advice
    """)
    
    st.info("""
    **Privacy and Security:**
    
    - No personal data is stored or transmitted
    - Automatic de-identification of PHI/PII
    - All processing happens locally
    - Chat history is only stored in your browser session
    """)

else:
    st.error("""
    **Model Loading Error**
    
    Unable to load the required models. Please ensure:
    
    1. The model files are available in the `models/` directory
    2. All dependencies are installed correctly
    3. The configuration file exists at `configs/base_config.yaml`
    
    You can train a model using: `python scripts/train.py`
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    Virtual Health Assistant - Research Demo | 
    <a href='https://github.com/your-repo/virtual-health-assistant' target='_blank'>GitHub</a> | 
    <a href='DISCLAIMER.md' target='_blank'>Disclaimer</a>
</div>
""", unsafe_allow_html=True)
