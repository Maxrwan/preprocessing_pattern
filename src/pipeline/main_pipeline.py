""" 
    MAIN PIPELINE 
"""

import pandas as pd

from src.io.loader import load_audio
from src.preprocessing.filtering import lowpass
from src.preprocessing.windowing import window_signal
from src.features.feature_pipeline import extract_features

from config.config import (
    WINDOW_SIZE,
    STEP_SIZE,
    CUTOFF_FREQ,
    LABEL_MAP,
    DEBUG
)

print("Pipeline initialized 🖕🫦")

def process_file(file_info):
    """
    Processes a single WAV file:
    - Loads audio
    - Applies filtering
    - Splits into windows
    - Extracts features
    - Attaches label + metadata
    """

    file_path = file_info["file_path"]
    machine = file_info["machine"]
    condition = file_info["label_name"]

    if DEBUG:
        print(f"[INFO] Processing: {file_path}")

    # =========================
    # Load audio
    # =========================
    signal, sr = load_audio(file_path)

    # Safety check
    if signal is None or len(signal) == 0:
        if DEBUG:
            print(f"[WARNING] Empty signal: {file_path}")
        return []

    # =========================
    # Filtering
    # =========================
    try:
        filtered = lowpass(signal, CUTOFF_FREQ, sr) # type: ignore
    except Exception as e:
        if DEBUG:
            print(f"[ERROR] Filtering failed: {file_path} | {e}")
        return []

    # =========================
    # Label mapping
    # =========================
    key = (machine, condition)

    if key not in LABEL_MAP:
        raise ValueError(f"Unknown label combination: {key}")

    label = LABEL_MAP[key]

    # =========================
    # Windowing + feature extraction
    # =========================
    all_features = []

    for window in window_signal(filtered, WINDOW_SIZE, STEP_SIZE):

        # Skip too-small windows
        if len(window) < WINDOW_SIZE:
            continue

        try:
            feats = extract_features(window)

            # Attach metadata
            feats["label"] = label
            print(machine,condition, label)
            feats["machine"] = machine
            feats["condition"] = condition

            all_features.append(feats)

        except Exception as e:
            if DEBUG:
                print(f"[ERROR] Feature extraction failed: {file_path} | {e}")
            continue

    return all_features


def process_dataset(dataset):
    """
    Processes entire dataset list (from dataset.py)
    """

    all_data = []

    for i, file_info in enumerate(dataset):

        if DEBUG:
            print(f"[INFO] File {i+1}/{len(dataset)}")

        feats = process_file(file_info)
        all_data.extend(feats)

    # Convert to DataFrame
    df = pd.DataFrame(all_data)

    return df