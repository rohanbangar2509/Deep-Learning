"""
Inference script for the fine-tuned IMDb BERT sentiment classifier.

Example:
    python predict.py --text "The movie was excellent and emotionally engaging."
"""

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = "outputs/bert-imdb"


def predict(text):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
    model.eval()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        probabilities = torch.softmax(logits, dim=-1)[0]
        predicted_id = probabilities.argmax().item()

    label = model.config.id2label[predicted_id]
    confidence = probabilities[predicted_id].item()

    return label, confidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True, help="Text/review to classify.")
    args = parser.parse_args()

    label, confidence = predict(args.text)
    print(f"Sentiment : {label}")
    print(f"Confidence: {confidence:.4f}")
