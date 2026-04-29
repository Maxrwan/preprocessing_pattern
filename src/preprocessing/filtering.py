""" 
    FILTER DATA
"""

from scipy.signal import butter, filtfilt
import numpy as np

def lowpass(signal: np.ndarray, cutoff: float, fs: float) -> np.ndarray:
    b, a = butter(4, cutoff / (fs / 2), btype='low', output='ba') # type: ignore
    filtered = filtfilt(b, a, signal)
    return filtered
