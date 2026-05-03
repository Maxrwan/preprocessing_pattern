""" 
    MAIN PIPELINE 
"""

import pandas as pd
import numpy as np

from src.io.loader import load_audio
from src.preprocessing.filtering import lowpass
from src.preprocessing.windowing import window_signal, center_window_energy, trim_signal
from src.features.feature_pipeline import extract_features

from collections import defaultdict

failure_counter = defaultdict(int)

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

        window = center_window_energy(window)
        
        # TEMP DISABLED FOR DEBUGGING 
        # if np.mean(np.abs(window)) < 0.005:
        #     continue 
        
        try:
            feats = extract_features(window, sr)

            all_features.append({
                "features" : feats,
                "label": label,
                "machine" : machine,
                "condition" : condition
            })

        except Exception as e:
            if DEBUG:
                print(f"[ERROR] Feature extraction failed: {file_path} | {e}")
            continue

    if len(all_features) == 0:
        print(f"[WARNING] No features extracted: {file_path}")
        failure_counter[(machine, condition)] += 1
    return all_features


def process_dataset(dataset):
    """
    Processes entire dataset list (from dataset.py)
    """

    from multiprocessing import Pool, cpu_count
    from tqdm import tqdm 
    
    with Pool(4) as p:
        results = list(tqdm(p.imap(process_file, dataset), total = len(dataset)))
            
    all_data = []
    
    for r in results:
        all_data.extend(r)
        
    print("\n ==== Failure Summary ====")
    for k, v in failure_counter.items():
        print(f"{k}: {v}")
    return all_data