""" 
    TIME FEATURES
"""

import numpy as np
import librosa

def compute_time_features(signal):
    return {
        "mean": float(np.mean(signal)),
        "std": float(np.std(signal)),
        "rms": float(np.sqrt(np.mean(signal**2))),
        "zcr": float(np.mean(librosa.feature.zero_crossing_rate(signal)))
    }