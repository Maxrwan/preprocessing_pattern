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

from collections import Counter


from src.io.dataset import get_classwise_files
from src.io.batching import create_balanced_batches, create_batches
from src.pipeline.main_pipeline import process_dataset
from config.config import INPUT_PATH, OUTPUT_PATH


def main():
    print("🚀 Starting preprocessing pipeline...")

    # =========================
    # 1. Load dataset structure
    # =========================
    class_dict = get_classwise_files(INPUT_PATH)
    for k, v in class_dict.items():
        print(k, len(v))

    print(f"[INFO] Classes found: {len(class_dict)}")

    # =========================
    # 2. Create balanced batches
    # =========================
    batches = create_batches(class_dict)

    print(f"[INFO] Total batches: {len(batches)}")

    # =========================
    # 3. Ensure output directory exists
    # =========================
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    # =========================
    # 4. Process each batch
    # =========================
    
    existing_batches = len([f for f in os.listdir(OUTPUT_PATH) if f.endswith(".npz")])
    
    for i, batch in enumerate(batches):
        # 🔥 Resume capability NEW
        output_file = os.path.join(OUTPUT_PATH, f"batch_{existing_batches + i}.npz")

        if os.path.exists(output_file):
            print(f"[SKIP] {output_file} already exists")
            continue
        
        print(f"\n📦 Processing batch {i+1}/{len(batches)}")
        print(f"[INFO] Files in batch: {len(batch)}")

        # =========================
        # Process batch
        # =========================
        data = process_dataset(batch)
        
        
        labels = [sample["label"] for sample in data]
        print(f"[DEBUG] Batch {i} label distribution:", Counter(labels))

        if len(data) == 0:
            print(f"[WARNING] Batch {i} produced no data, skipping")
            continue

        # =========================
        # Convert to arrays
        # =========================
        X = []
        y = []
            
        for sample in data:
            X.append(sample["features"]) #type: ignore 
            y.append(sample["label"]) #type: ignore 

        X = np.array(X, dtype=np.float16)
        y = np.array(y, dtype=np.int64)

        print(f"[INFO] Batch shape: X={X.shape}, y={y.shape}")

        # =========================
        # Save batch
        # =========================
        
        file_paths = [item["file_path"] for item in batch]
        
        np.savez_compressed(output_file, X=X, y=y, file_paths = file_paths)

        print(f"[SAVED] {output_file}")

        # =========================
        # Free memory
        # =========================
        del X, y, data

    print("\n✅ All batches processed!")


if __name__ == "__main__":
    main()
    