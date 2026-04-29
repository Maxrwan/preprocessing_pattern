"""
    Config file
"""

# ================================
# PATHS
# ================================

# Google Drive mounted path
INPUT_PATH = "/content/drive/MyDrive/Students/"
OUTPUT_PATH = "/content/drive/MyDrive/processed/"

# ================================
# FILE SETTINGS
# ================================

FILE_EXTENSION = ".wav"

# ================================
# WINDOWING PARAMETERS
# ================================

WINDOW_SIZE = 2048
STEP_SIZE = 1024  # 50% overlap

# ================================
# FILTER SETTINGS
# ================================

CUTOFF_FREQ = 200  # Hz (low-pass filter)

# ================================
# LABEL MAPPING (Machine + Condition)
# ================================

LABEL_MAP = {
    ("Machine 1", "Normal"): 0,
    ("Machine 1", "Abnormal"): 1,
    ("Machine 2", "Normal"): 2,
    ("Machine 2", "Abnormal"): 3,
    ("Machine 3", "Normal"): 4,
    ("Machine 3", "Abnormal"): 5,
}

# ================================
# OPTIONAL (DEBUG / SAFETY)
# ================================

# Limit number of files for testing (set to None for full run)
MAX_FILES = 5

# Enable debug prints
DEBUG = True