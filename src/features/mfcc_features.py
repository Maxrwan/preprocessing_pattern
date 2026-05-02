""" 
    MFCC FEATURES
"""

import librosa
import numpy as np

def pad_spectrogram(spec, target_length=128):
    if spec.shape[1] < target_length:
        pad_width = target_length - spec.shape[1]
        spec = np.pad(spec, ((0, 0), (0, pad_width)))
    else:
        spec = spec[:, :target_length]
    return spec


def compute_logmel(signal, sr, n_mels=128, hop_length=512):
    
    # 🔹 Compute mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=signal,
        sr=sr,
        n_mels=n_mels,
        hop_length=hop_length
    )

    # 🔹 Convert to log scale (THIS IS KEY)
    log_mel = librosa.power_to_db(mel, ref=np.max)

    # 🔹 Pad / trim (same idea as before but better resolution)
    log_mel = pad_spectrogram(log_mel, target_length=128)
    delta = librosa.feature.delta(log_mel)
    delta2 = librosa.feature.delta(log_mel, order=2)
        
    return log_mel.astype(np.float32), delta.astype(np.float32), delta2.astype(np.float32)