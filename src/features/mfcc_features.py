""" 
    MFCC FEATURES
"""

import numpy as np
import librosa

def compute_mfcc_features(signal, sr, n_mfcc=13):
    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)

    features = {}
    for i in range(n_mfcc):
        features[f"mfcc_{i}_mean"] = float(np.mean(mfccs[i]))
        features[f"mfcc_{i}_std"] = float(np.std(mfccs[i]))

    return features