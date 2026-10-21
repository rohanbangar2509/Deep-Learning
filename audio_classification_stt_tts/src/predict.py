"""Predict a spoken command from a WAV file."""
from pathlib import Path
import json
import sys
import time
import tensorflow as tf
from .audio_utils import file_to_features

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "audio_command_cnn.keras"
CLASSES_PATH = ROOT / "models" / "class_names.json"

def predict_file(path):
    if not MODEL_PATH.exists() or not CLASSES_PATH.exists():
        raise FileNotFoundError("Model not found. Train it first: python -m src.train")
    model = tf.keras.models.load_model(MODEL_PATH)
    classes = json.loads(CLASSES_PATH.read_text(encoding="utf-8"))
    features = file_to_features(path)[None, ...]
    start = time.perf_counter()
    probabilities = model.predict(features, verbose=0)[0]
    latency_ms = (time.perf_counter() - start) * 1000
    index = int(probabilities.argmax())
    return classes[index], float(probabilities[index]), latency_ms, dict(
        sorted(zip(classes, probabilities.tolist()), key=lambda x: x[1], reverse=True)
    )

def main():
    if len(sys.argv) < 2:
        print("Usage: python -m src.predict path/to/audio.wav")
        raise SystemExit(2)
    label, confidence, latency, ranked = predict_file(sys.argv[1])
    print(f"Predicted command: {label}")
    print(f"Confidence: {confidence:.2%}")
    print(f"Model inference latency: {latency:.1f} ms")
    print("Class probabilities:")
    for name, prob in ranked.items():
        print(f"  {name:>5}: {prob:.2%}")

if __name__ == "__main__":
    main()
