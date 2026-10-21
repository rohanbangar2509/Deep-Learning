"""Interactive audio classification + STT + TTS terminal demo."""
from pathlib import Path
from .predict import predict_file
from .stt_tts import transcribe_audio_file, speak_text

def main():
    print("=" * 58)
    print("AUDIO COMMAND RECOGNITION — CNN + STT + TTS")
    print("=" * 58)
    path = input("Enter path to a short WAV file: ").strip().strip('"')
    if not Path(path).is_file():
        print("File not found.")
        return
    try:
        label, confidence, latency, _ = predict_file(path)
    except Exception as exc:
        print(f"Classification failed: {exc}")
        return
    print(f"\nCNN prediction: {label} ({confidence:.1%} confidence)")
    print(f"Inference latency: {latency:.1f} ms")
    transcript = transcribe_audio_file(path)
    print(f"STT transcript: {transcript}")
    response = f"The predicted command is {label}. Confidence {confidence:.0%}."
    print(f"Spoken response: {response}")
    try:
        speak_text(response)
    except Exception as exc:
        print(f"TTS could not start on this system: {exc}")

if __name__ == "__main__":
    main()
