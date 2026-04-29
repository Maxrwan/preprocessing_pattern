""" 
    FEATURE PIPELINE
"""

from .time_features import compute_time_features
from .freq_features import compute_freq_features

def extract_features(signal):
    feats = compute_time_features(signal)
    feats.update(compute_freq_features(signal))
    return feats

