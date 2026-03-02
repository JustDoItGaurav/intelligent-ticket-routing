import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

MODEL_PATH = "models/bert"  # adjust if different
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model + tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(DEVICE)
model.eval()

# Label mapping (must match training)
label_map = {
    0: "Access",
    1: "Administrative rights",
    2: "HR Support",
    3: "Hardware",
    4: "Internal Project",
    5: "Miscellaneous",
    6: "Purchase",
    7: "Storage",
}


def predict_ticket(text):
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, padding=True, max_length=128
    )

    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)

    confidence, predicted_class = torch.max(probs, dim=1)

    confidence = confidence.item()
    predicted_class = predicted_class.item()

    print("\n==============================")
    print(f"Ticket: {text}")
    print(f"Prediction: {label_map[predicted_class]}")
    print(f"Confidence: {confidence:.4f}")

    if confidence > 0.85:
        print("Routing: Auto-route ✅")
    elif confidence > 0.6:
        print("Routing: Manual Review ⚠️")
    else:
        print("Routing: Fallback Queue ❌")
    print("==============================\n")


if __name__ == "__main__":
    while True:
        user_input = input("Enter ticket text (or 'exit'): ")
        if user_input.lower() == "exit":
            break
        predict_ticket(user_input)
