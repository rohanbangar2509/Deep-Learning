# Audio Classification with Speech-to-Text (STT) and Text-to-Speech (TTS)

**Author:** Rohan Bangar  
**Program:** B.Tech — CSE (Artificial Intelligence)

## 1. Project Overview

This project implements a spoken-command recognition application using a Convolutional Neural Network (CNN). It combines three tasks:

1. **Audio classification:** predicts which spoken command is present in an audio clip.
2. **Speech-to-Text (STT):** transcribes a spoken input into text using the SpeechRecognition Google Web Speech interface (internet required).
3. **Text-to-Speech (TTS):** speaks the application's response using `pyttsx3` (usually works offline with a system voice installed).

The CNN is trained on the **Mini Speech Commands** dataset, which contains short recordings of eight commands: `down`, `go`, `left`, `no`, `right`, `stop`, `up`, and `yes`.

> STT transcription and CNN audio classification are separate outputs. STT may return text outside the CNN's eight supported classes; in that case, the classifier still reports its own prediction and confidence.

## 2. Architecture

```text
Audio file / microphone
        |
        +--------------------------+
        |                          |
        v                          v
  Log-Mel spectrogram        Speech-to-Text (STT)
        |                          |
        v                          v
  CNN audio classifier        Transcribed text
        |                          |
        +------------+-------------+
                     |
                     v
          Response / prediction
                     |
                     v
           Text-to-Speech (TTS)
```

## 3. Dataset

The notebook downloads the public TensorFlow Mini Speech Commands dataset (approximately 180 MB; download size may vary). It is split into training, validation, and test sets. The code uses a reproducible random seed and limits the number of files per class to make experiments manageable on a laptop.

Dataset source: https://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip

## 4. Evaluation Metrics

- **Accuracy:** fraction of test examples classified correctly.
- **Precision:** proportion of predicted examples for a class that are correct.
- **Recall:** proportion of actual examples of a class correctly identified.
- **F1-score:** harmonic mean of precision and recall.
- **Confusion matrix:** summarizes correct and incorrect predictions by class.
- **Inference latency:** optional time required for a single model prediction.

Do not write fixed performance numbers into the report before running the notebook. Record the actual metrics printed by your execution.

## 5. Repository Structure

```text
audio_classification_stt_tts/
├── data/                         # Downloaded dataset is not committed
├── models/                       # Trained model output
├── notebooks/
│   └── Audio_Classification_STT_TTS.ipynb
├── results/                      # Metrics and plots generated at runtime
├── src/
│   ├── audio_utils.py             # Audio loading and log-Mel features
│   ├── train.py                   # Dataset preparation and CNN training
│   ├── predict.py                 # Predict a command from a WAV file
│   ├── stt_tts.py                 # STT and TTS helpers
│   └── app.py                     # Interactive terminal application
├── .gitignore
├── requirements.txt
└── README.md
```

## 6. Setup

Recommended: Python 3.10 or 3.11, a virtual environment, and a recent pip.

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If TensorFlow installation fails on your system, check the current TensorFlow installation guidance for your Python version. CPU training is supported; a GPU is optional.

## 7. Run the Project

### Option A — Jupyter Notebook (recommended for the assignment)

```bash
jupyter notebook
```

Open `notebooks/Audio_Classification_STT_TTS.ipynb` and run the cells in order. The notebook explains each stage, trains and evaluates the CNN, saves the model, and demonstrates STT/TTS.

### Option B — Train from the command line

Run from the project root:

```bash
python -m src.train
```

The first run downloads and prepares the dataset. It saves the trained model to `models/audio_command_cnn.keras`, class names to `models/class_names.json`, and evaluation outputs to `results/`.

### Predict from a WAV file

```bash
python -m src.predict path/to/audio.wav
```

The WAV file should contain one short command, preferably about one second long. Audio is resampled to 16 kHz and converted to a log-Mel spectrogram.

### Run the interactive STT/TTS application

```bash
python -m src.app
```

Choose a WAV file when prompted. The application reports the CNN class and confidence, attempts online STT transcription, and reads the response aloud. STT needs an internet connection and a working microphone is not required when using a WAV file. TTS depends on an installed system voice.

## 8. Methodology

1. Download and inspect the labeled audio dataset.
2. Split examples into train, validation, and test partitions.
3. Resample audio to 16 kHz and pad/trim it to a fixed duration.
4. Compute a log-Mel spectrogram for each waveform.
5. Train a CNN using the training set and monitor validation performance.
6. Evaluate once on the held-out test set using accuracy, precision, recall, F1-score, and a confusion matrix.
7. Save the model and class labels.
8. Run inference on a new WAV file.
9. Demonstrate STT transcription and TTS spoken output.

## 9. Important Notes

- The first dataset download requires internet access.
- Google Web Speech recognition requires internet and may have usage limits. It is used here as a practical STT component, not as the trained CNN.
- `pyttsx3` uses a system-installed speech engine; voice availability varies by operating system.
- The model recognizes only the eight dataset classes. It is not a general-purpose speech recognizer.
- For a fair evaluation, do not tune the model using test-set results.
- The dataset is downloaded at runtime and should not be committed to Git.

## 10. Learning Outcomes

After completing the assignment, you should be able to explain audio preprocessing, log-Mel spectrograms, CNN-based classification, train/validation/test splits, classification metrics, STT, TTS, model serialization, and inference.

## 11. Future Improvements

- Compare CNN with an LSTM/GRU or a pretrained audio model.
- Add noise augmentation and more audio classes.
- Use an offline STT model such as Whisper or Vosk.
- Build a web interface and evaluate performance across different microphones and background noise.
