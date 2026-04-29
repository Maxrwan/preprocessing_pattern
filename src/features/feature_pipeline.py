""" 
    FEATURE PIPELINE
"""

from .time_features import compute_time_features
from .freq_features import compute_freq_features
from .mfcc_features import compute_mfcc_features

def extract_features(signal, sr):
    feats = {}

    feats.update(compute_time_features(signal))
    feats.update(compute_freq_features(signal, sr))
    feats.update(compute_mfcc_features(signal, sr))

    return feats

