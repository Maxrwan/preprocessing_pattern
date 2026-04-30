""" 
    MFCC FEATURES
"""

import librosa
import numpy as np

def pad_mfcc(mfcc, target_length=44):
    if mfcc.shape[1] < target_length:
        pad_width = target_length - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)))
    else:
        mfcc = mfcc[:, :target_length]
    return mfcc

def compute_mfcc(signal, sr, n_mfcc=13):
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)

    mfcc = pad_mfcc(mfcc)

    # normalize (important for CNN)
    mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-6)

    return mfcc.astype(np.float32)