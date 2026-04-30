""" 
    FEATURE PIPELINE
"""

from .time_features import compute_time_features
from .freq_features import compute_freq_features
from .mfcc_features import compute_mfcc

def extract_features(signal, sr):
    return compute_mfcc(signal,sr)

