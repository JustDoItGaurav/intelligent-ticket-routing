# 🧠 Intelligent Ticket Routing System

An end-to-end NLP pipeline for automated classification of enterprise support tickets.

## 🚀 Project Overview

This project compares classical machine learning with Transformer-based deep learning for multi-class ticket routing.

- Dataset: 47,837 real-world helpdesk tickets
- Classes: 8 operational categories
- Goal: Reduce manual routing workload

---

## 🏗 Architecture

1. Data Loading
2. Text Preprocessing
3. TF-IDF + Logistic Regression Baseline
4. DistilBERT Fine-Tuning (GPU)
5. Confidence-Based Routing Simulation

---

## 📊 Results

### Baseline (TF-IDF + Logistic Regression)
- Accuracy: 85.2%
- Macro F1: 0.852
- Auto-route: 42%
- Fallback: 22%

### DistilBERT
- Accuracy: 87.7%
- Macro F1: 0.874
- Auto-route: 87%
- Fallback: 4%

---

## 🧠 Key Features

- Probability calibration (baseline)
- Confidence threshold routing
- Modular ML architecture
- Train/Validation/Test split
- Transformer fine-tuning with HuggingFace

---

## 🛠 Tech Stack

- Python
- Scikit-learn
- HuggingFace Transformers
- PyTorch
- NumPy / Pandas

---

## ▶️ How to Run

```bash
python -m src.training.run_training