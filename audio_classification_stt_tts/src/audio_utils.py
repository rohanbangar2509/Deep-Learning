"""Audio loading and log-Mel spectrogram utilities."""
from pathlib import Path
import numpy as np
import librosa

SAMPLE_RATE = 16_000
DURATION_SECONDS = 1.0
N_SAMPLES = int(SAMPLE_RATE * DURATION_SECONDS)
N_MELS = 64
N_FFT = 512
HOP_LENGTH = 160

def load_audio(path, sample_rate=SAMPLE_RATE):
    """Load mono audio, resample it, and pad/trim it to one second."""
    waveform, _ = librosa.load(str(path), sr=sample_rate, mono=True)
    if len(waveform) < N_SAMPLES:
        waveform = np.pad(waveform, (0, N_SAMPLES - len(waveform)))
    else:
        waveform = waveform[:N_SAMPLES]
    return waveform.astype(np.float32)

def waveform_to_logmel(waveform):
    """Convert a waveform to a normalized log-Mel spectrogram."""
    mel = librosa.feature.melspectrogram(
        y=np.asarray(waveform, dtype=np.float32),
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=20,
        fmax=SAMPLE_RATE // 2,
        power=2.0,
    )
    logmel = librosa.power_to_db(mel, ref=np.max)
    # Per-example standardization helps reduce absolute recording-level differences.
    logmel = (logmel - logmel.mean()) / (logmel.std() + 1e-6)
    return logmel.astype(np.float32)[..., np.newaxis]

def file_to_features(path):
    return waveform_to_logmel(load_audio(path))
