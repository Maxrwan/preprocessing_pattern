""" 
    ENTRY POINT 
"""

"""
RUN SCRIPT
- Creates balanced batches
- Processes WAV files into MFCC matrices
- Saves batches as .npz for CNN training
"""

import os
import numpy as np

from src.io.dataset import get_classwise_files
from src.io.batching import create_balanced_batches
from src.pipeline.main_pipeline import process_dataset
from config.config import INPUT_PATH, OUTPUT_PATH


def main():
    print("🚀 Starting preprocessing pipeline...")

    # =========================
    # 1. Load dataset structure
    # =========================
    class_dict = get_classwise_files(INPUT_PATH)

    print(f"[INFO] Classes found: {len(class_dict)}")

    # =========================
    # 2. Create balanced batches
    # =========================
    batches = create_balanced_batches(class_dict, files_per_class=5)

    print(f"[INFO] Total batches: {len(batches)}")

    # =========================
    # 3. Ensure output directory exists
    # =========================
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    # =========================
    # 4. Process each batch
    # =========================
    for i, batch in enumerate(batches):
        output_file = os.path.join(OUTPUT_PATH, f"batch_{i}.npz")

        # 🔥 Resume capability
        if os.path.exists(output_file):
            print(f"[SKIP] Batch {i} already exists")
            continue

        print(f"\n📦 Processing batch {i+1}/{len(batches)}")
        print(f"[INFO] Files in batch: {len(batch)}")

        # =========================
        # Process batch
        # =========================
        data = process_dataset(batch)

        if len(data) == 0:
            print(f"[WARNING] Batch {i} produced no data, skipping")
            continue

        # =========================
        # Convert to arrays
        # =========================
        X = []
        y = []

        for sample in data:
            X.append(sample["mfcc"]) #type: ignore 
            y.append(sample["label"]) #type: ignore 

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int64)

        print(f"[INFO] Batch shape: X={X.shape}, y={y.shape}")

        # =========================
        # Save batch
        # =========================
        np.savez(output_file, X=X, y=y)

        print(f"[SAVED] {output_file}")

        # =========================
        # Free memory
        # =========================
        del X, y, data

    print("\n✅ All batches processed!")


if __name__ == "__main__":
    main()