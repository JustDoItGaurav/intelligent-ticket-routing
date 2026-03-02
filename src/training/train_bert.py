import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from src.utils.logger import get_logger
from src.config import MODELS_DIR

logger = get_logger(__name__)


# =========================
# Custom Dataset
# =========================
class TicketDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.encodings = tokenizer(
            texts.tolist(),
            truncation=True,
            padding=True,
            max_length=max_len,
        )
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


# =========================
# Metrics Function
# =========================
def compute_metrics(pred):
    logits, labels = pred
    predictions = np.argmax(logits, axis=1)

    acc = accuracy_score(labels, predictions)
    macro_f1 = f1_score(labels, predictions, average="macro")

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
    }


# =========================
# Train BERT
# =========================
def train_bert(df):

    logger.info("Starting DistilBERT training...")

    X = df["Document"]
    y = df["Topic_group"]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_encoded, test_size=0.30, stratify=y_encoded, random_state=42
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )

    logger.info(f"Train size: {len(X_train)}")
    logger.info(f"Validation size: {len(X_val)}")
    logger.info(f"Test size: {len(X_test)}")

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    train_dataset = TicketDataset(X_train, y_train, tokenizer)
    val_dataset = TicketDataset(X_val, y_val, tokenizer)
    test_dataset = TicketDataset(X_test, y_test, tokenizer)

    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=len(label_encoder.classes_),
    )

    training_args = TrainingArguments(
        output_dir=str(MODELS_DIR / "bert"),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir=str(MODELS_DIR / "bert_logs"),
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # ======================
    # TRAIN
    # ======================
    trainer.train()

    # ======================
    # TEST EVALUATION
    # ======================
    predictions = trainer.predict(test_dataset)
    logits = predictions.predictions
    preds = np.argmax(logits, axis=1)

    test_acc = accuracy_score(y_test, preds)
    test_f1 = f1_score(y_test, preds, average="macro")

    logger.info(f"\nBERT Test Accuracy: {test_acc:.4f}")
    logger.info(f"BERT Test Macro F1: {test_f1:.4f}")
    logger.info(
        "\nBERT Test Classification Report:\n" + classification_report(y_test, preds)
    )

    # ======================
    # TEST CONFIDENCE ROUTING
    # ======================
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=1).numpy()

    confidence = np.max(probs, axis=1)

    logger.info(f"\nAverage BERT test confidence: {np.mean(confidence):.4f}")
    logger.info(f"BERT auto-route (>0.85): {(confidence > 0.85).mean() * 100:.2f}%")
    logger.info(f"BERT fallback (<0.6): {(confidence <= 0.6).mean() * 100:.2f}%")
    logger.info(f"High confidence (0.9+): {(confidence > 0.9).mean() * 100:.2f}%")
    logger.info(
        f"Medium confidence (0.6–0.9): "
        f"{((confidence > 0.6) & (confidence <= 0.9)).mean() * 100:.2f}%"
    )

    # ======================
    # SAVE MODEL
    # ======================
    bert_dir = MODELS_DIR / "bert"
    bert_dir.mkdir(parents=True, exist_ok=True)

    trainer.save_model(str(bert_dir))
    tokenizer.save_pretrained(str(bert_dir))

    logger.info("DistilBERT model saved successfully.")

    return trainer, label_encoder
