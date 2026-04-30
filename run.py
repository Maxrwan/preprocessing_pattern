""" 
    ENTRY POINT 
"""

from src.pipeline.main_pipeline import process_dataset
from src.io.dataset import get_classwise_files
from src.io.batching import create_balanced_batches
from config.config import INPUT_PATH, OUTPUT_PATH

import pandas as pd
import os

def main():
    class_dict = get_classwise_files(INPUT_PATH)

    batches = create_balanced_batches(class_dict, files_per_class=5)

    print(f"Total batches: {len(batches)}")

    for i, batch in enumerate(batches):
        print(f"Processing batch {i+1}/{len(batches)}")

        df = process_dataset(batch)

        output_file = os.path.join(
            OUTPUT_PATH, f"batch_{i}.parquet"
        )

        df.to_parquet(output_file)

if __name__ == "__main__":
    main()