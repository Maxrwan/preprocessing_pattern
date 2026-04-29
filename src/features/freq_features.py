""" 
    FREQUENCY FEATURES 
"""

import numpy as np
import librosa

def compute_freq_features(signal, sr):
    centroid = librosa.feature.spectral_centroid(y=signal, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=signal, sr=sr)

    return {
        "spectral_centroid": float(np.mean(centroid)),
        "spectral_bandwidth": float(np.mean(bandwidth)),
    }