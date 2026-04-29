""" 
    TIME FEATURES
"""

import numpy as np 

def compute_time_features(signal):
    return {
        "mean" : np.mean(signal),
        "std" : np.std(signal), 
        "rms" : np.sqrt(np.mean(signal**2))
    }