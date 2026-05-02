""" 
    FEATURE PIPELINE
"""

from .time_features import compute_time_features
from .freq_features import compute_freq_features
from .logmel_features import compute_logmel

def extract_features(signal, sr):
    return compute_logmel(signal,sr)

