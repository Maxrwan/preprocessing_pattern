""" 
    FREQUENCY FEATURES 
"""

import numpy as np
from scipy.fft import fft 

def compute_freq_features(signal):
    spectrum = np.abs(fft(signal)) # type: ignore
    return {
        "spec_max" : np.max(spectrum),
        "spec_mean" : np.mean(spectrum)
    }