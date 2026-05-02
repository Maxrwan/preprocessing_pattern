""" 
    WINDOW DATA
"""

def window_signal(signal, size, step):
    for i in range(0, len(signal) - size, step):
        yield signal[i:i+size]
        
        
import numpy as np

def center_window_energy(window):
    """
    Shifts the window so that the highest-energy region is centered.
    Keeps same length.
    """

    energy = np.abs(window)

    peak_idx = np.argmax(energy)
    center = len(window) // 2

    shift = center - peak_idx

    # shift signal
    aligned = np.roll(window, shift)

    return aligned