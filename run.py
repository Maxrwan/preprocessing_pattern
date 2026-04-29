""" 
    ENTRY POINT 
"""

from src.pipeline.main_pipeline import process_dataset
from src.io.dataset import get_all_files
from config.config import INPUT_PATH, OUTPUT_PATH, MAX_FILES

def main():
    dataset = get_all_files(INPUT_PATH)

    if MAX_FILES is not None:
        dataset = dataset[:5]

    df = process_dataset(dataset)

    df.to_parquet(OUTPUT_PATH + "final_dataset.parquet")

if __name__ == "__main__":
    main()