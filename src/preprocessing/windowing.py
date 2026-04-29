""" 
    WINDOW DATA
"""

def window_signal(signal, size, step):
    for i in range(0, len(signal) - size, step):
        yield signal[i:i+size]