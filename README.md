# Intelligent Ticket Routing System

An end-to-end Natural Language Processing (NLP) pipeline for automated classification of enterprise support tickets.  
The system compares classical machine learning methods with transformer-based deep learning models to evaluate improvements in ticket routing accuracy and automation.

---

## Project Overview

Enterprise helpdesk systems receive a large volume of support tickets that must be manually reviewed and routed to the correct department.  
This project builds an automated ticket classification system that can assist in routing requests based on the textual content of the ticket.

Key objectives:

- Automatically classify support tickets into operational categories
- Compare classical NLP models with transformer-based models
- Simulate production-style routing decisions based on model confidence

Dataset characteristics:

- Total tickets: 47,837
- Number of classes: 8 operational categories
- Task: Multi-class text classification

---

## System Architecture

The project follows a modular machine learning pipeline:

1. Data Loading  
2. Text Preprocessing  
3. Train / Validation / Test Split  
4. Baseline Model (TF-IDF + Logistic Regression)  
5. Transformer Model (DistilBERT Fine-Tuning)  
6. Model Evaluation  
7. Confidence-Based Routing Simulation  
8. Inference Interface

---

## Model Approaches

### Baseline Model
TF-IDF vectorization combined with Logistic Regression was used to establish a classical machine learning benchmark.

Results:

- Accuracy: 85.2%
- Macro F1 Score: 0.852
- Auto-route rate: 42%
- Fallback rate: 22%

### Transformer Model
A DistilBERT transformer model was fine-tuned using HuggingFace Transformers to capture contextual language representations.

Results:

- Accuracy: 87.7%
- Macro F1 Score: 0.874
- Auto-route rate: 87%
- Fallback rate: 4%

The transformer model significantly improved routing confidence and reduced manual review requirements.

---

## Key Features

- End-to-end NLP pipeline for multi-class ticket classification
- Baseline classical ML model for performance benchmarking
- Transformer fine-tuning using DistilBERT
- Confidence-based routing simulation (auto-route / manual review / fallback)
- Modular project structure for training, evaluation, and inference
- Command-line inference tool for testing predictions on custom tickets

---

## Technology Stack

Languages and frameworks:

- Python

Machine learning libraries:

- Scikit-learn
- HuggingFace Transformers
- PyTorch

Data processing:

- Pandas
- NumPy

Interface:

- Streamlit (for demonstration UI)

---

## Project Structure
