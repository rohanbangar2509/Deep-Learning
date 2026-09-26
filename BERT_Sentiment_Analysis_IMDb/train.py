"""
BERT Sentiment Analysis on IMDb
--------------------------------
Fine-tunes bert-base-uncased for binary sentiment classification.

Default configuration is intentionally moderate for student/lab hardware.
Increase TRAIN_SAMPLES / EPOCHS for a larger experiment.
"""

import os
import random
import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)
from tqdm.auto import tqdm

MODEL_NAME = "bert-base-uncased"
OUTPUT_DIR = "outputs/bert-imdb"
SEED = 42

TRAIN_SAMPLES = 8000
VAL_SAMPLES = 2000
TEST_SAMPLES = 2000

MAX_LENGTH = 256
BATCH_SIZE = 8
EPOCHS = 2
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_dataset():
    dataset = load_dataset("stanfordnlp/imdb")

    train_full = dataset["train"].shuffle(seed=SEED)
    test_full = dataset["test"].shuffle(seed=SEED)

    train = train_full.select(range(min(TRAIN_SAMPLES, len(train_full))))
    val = train_full.select(
        range(TRAIN_SAMPLES, min(TRAIN_SAMPLES + VAL_SAMPLES, len(train_full)))
    )
    test = test_full.select(range(min(TEST_SAMPLES, len(test_full))))

    return train, val, test


def tokenize_dataset(train, val, test, tokenizer):
    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_LENGTH,
        )

    train = train.map(tokenize, batched=True)
    val = val.map(tokenize, batched=True)
    test = test.map(tokenize, batched=True)

    keep_cols = ["input_ids", "attention_mask", "label"]
    if "token_type_ids" in train.column_names:
        keep_cols.insert(2, "token_type_ids")

    train.set_format("torch", columns=keep_cols)
    val.set_format("torch", columns=keep_cols)
    test.set_format("torch", columns=keep_cols)
    return train, val, test


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            total_loss += outputs.loss.item()
            preds = outputs.logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary", zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds)

    return {
        "loss": total_loss / max(len(loader), 1),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
    }


def main():
    set_seed()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train, val, test = prepare_dataset()
    print(f"Train: {len(train)} | Validation: {len(val)} | Test: {len(test)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train, val, test = tokenize_dataset(train, val, test, tokenizer)

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator)
    test_loader = DataLoader(test, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label={0: "NEGATIVE", 1: "POSITIVE"},
        label2id={"NEGATIVE": 0, "POSITIVE": 1},
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    best_f1 = -1.0
    history = {"train_loss": [], "val_loss": [], "val_accuracy": [], "val_f1": []}

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for batch in progress:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running_loss / len(train_loader)
        val_metrics = evaluate(model, val_loader, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["val_f1"].append(val_metrics["f1"])

        print(
            f"Epoch {epoch+1}: train_loss={train_loss:.4f}, "
            f"val_loss={val_metrics['loss']:.4f}, "
            f"val_accuracy={val_metrics['accuracy']:.4f}, "
            f"val_f1={val_metrics['f1']:.4f}"
        )

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            model.save_pretrained(OUTPUT_DIR)
            tokenizer.save_pretrained(OUTPUT_DIR)
            print("Saved best model.")

    # Reload best checkpoint before final test evaluation.
    model = AutoModelForSequenceClassification.from_pretrained(OUTPUT_DIR).to(device)
    test_metrics = evaluate(model, test_loader, device)

    print("\nFinal Test Results")
    print(f"Accuracy : {test_metrics['accuracy']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall   : {test_metrics['recall']:.4f}")
    print(f"F1-score : {test_metrics['f1']:.4f}")
    print("Confusion Matrix:")
    print(test_metrics["confusion_matrix"])

    np.save(os.path.join(OUTPUT_DIR, "confusion_matrix.npy"), test_metrics["confusion_matrix"])
    np.savez(
        os.path.join(OUTPUT_DIR, "training_history.npz"),
        train_loss=history["train_loss"],
        val_loss=history["val_loss"],
        val_accuracy=history["val_accuracy"],
        val_f1=history["val_f1"],
    )

    print(f"\nModel saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
