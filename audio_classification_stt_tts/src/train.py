"""Download Mini Speech Commands, train a CNN, and evaluate it."""
from pathlib import Path
import json
import random
import urllib.request
import zipfile
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from .audio_utils import load_audio, waveform_to_logmel

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
EXTRACT_DIR = DATA_DIR / "mini_speech_commands"
ZIP_PATH = DATA_DIR / "mini_speech_commands.zip"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
URL = "https://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip"
CLASSES = ["down", "go", "left", "no", "right", "stop", "up", "yes"]
SEED = 42
MAX_PER_CLASS = 900  # set to None to use all available examples

def download_dataset():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not EXTRACT_DIR.exists():
        if not ZIP_PATH.exists():
            print("Downloading Mini Speech Commands dataset (~180 MB)...")
            urllib.request.urlretrieve(URL, ZIP_PATH)
        print("Extracting dataset...")
        with zipfile.ZipFile(ZIP_PATH, "r") as zf:
            zf.extractall(DATA_DIR)
    if not EXTRACT_DIR.exists():
        raise FileNotFoundError(f"Expected dataset folder at {EXTRACT_DIR}")

def collect_examples():
    paths, labels = [], []
    for label_idx, label in enumerate(CLASSES):
        files = sorted((EXTRACT_DIR / label).glob("*.wav"))
        random.Random(SEED).shuffle(files)
        if MAX_PER_CLASS is not None:
            files = files[:MAX_PER_CLASS]
        paths.extend(files)
        labels.extend([label_idx] * len(files))
    if not paths:
        raise RuntimeError("No WAV files found. Check the dataset download and extraction.")
    return np.array(paths, dtype=object), np.array(labels, dtype=np.int64)

def make_features(paths):
    X = []
    for i, path in enumerate(paths, 1):
        X.append(waveform_to_logmel(load_audio(path)))
        if i % 500 == 0:
            print(f"Processed {i}/{len(paths)} audio files")
    return np.stack(X).astype(np.float32)

def build_model(input_shape, num_classes):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs, name="audio_command_cnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

def main():
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    MODEL_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    download_dataset()
    paths, labels = collect_examples()
    print("Examples per class:")
    for idx, label in enumerate(CLASSES):
        print(f"  {label:>5}: {(labels == idx).sum()}")
    X = make_features(paths)
    y = labels
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=SEED, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp
    )
    print(f"Train: {len(y_train)} | Validation: {len(y_val)} | Test: {len(y_test)}")
    model = build_model(X.shape[1:], len(CLASSES))
    model.summary()
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=4, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_DIR / "best_audio_command_cnn.keras",
            monitor="val_loss", save_best_only=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5
        ),
    ]
    history = model.fit(
        X_train, y_train, validation_data=(X_val, y_val),
        epochs=20, batch_size=32, callbacks=callbacks, verbose=1
    )
    model = tf.keras.models.load_model(MODEL_DIR / "best_audio_command_cnn.keras")
    probs = model.predict(X_test, verbose=0)
    preds = probs.argmax(axis=1)
    accuracy = accuracy_score(y_test, preds)
    report = classification_report(
        y_test, preds, labels=np.arange(len(CLASSES)),
        target_names=CLASSES, output_dict=True, zero_division=0
    )
    print(f"\nTest accuracy: {accuracy:.4f}")
    print(classification_report(
        y_test, preds, labels=np.arange(len(CLASSES)),
        target_names=CLASSES, zero_division=0
    ))
    with open(RESULTS_DIR / "classification_report.json", "w", encoding="utf-8") as f:
        json.dump({"test_accuracy": accuracy, "report": report}, f, indent=2)
    with open(MODEL_DIR / "class_names.json", "w", encoding="utf-8") as f:
        json.dump(CLASSES, f, indent=2)
    model.save(MODEL_DIR / "audio_command_cnn.keras")
    plt.figure(figsize=(9, 7))
    sns.heatmap(confusion_matrix(y_test, preds), annot=True, fmt="d",
                xticklabels=CLASSES, yticklabels=CLASSES, cmap="Blues")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("CNN Audio Classification — Test Confusion Matrix")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=160)
    plt.close()
    plt.figure(figsize=(8, 4))
    plt.plot(history.history["accuracy"], label="Training accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "training_curves.png", dpi=160)
    plt.close()
    print(f"\nSaved model to: {MODEL_DIR / 'audio_command_cnn.keras'}")
    print(f"Saved evaluation outputs to: {RESULTS_DIR}")

if __name__ == "__main__":
    main()
