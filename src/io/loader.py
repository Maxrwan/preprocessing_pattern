"""
     LOAD DATA
"""

import librosa
import numpy as np

def load_audio(file_path: str):
    try:
        signal, sr = librosa.load(file_path, sr=16000, mono = True, res_type = "kaiser_fast")

        if signal is None or len(signal) == 0:
            raise ValueError("Empty signal")

        # convert stereo → mono if needed
        if signal.ndim > 1:
            signal = signal.mean(axis=1)

        return np.asarray(signal), sr

    except Exception as e:
        print(f"[ERROR] Failed to load {file_path}: {e}")
        return None, None