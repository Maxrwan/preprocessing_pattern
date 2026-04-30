""" 
    BATCHING FILE 
"""
import random


def create_balanced_batches(class_dict, files_per_class=5, shuffle=True):
    """
    Creates balanced batches where each batch contains the same number
    of files from each class.

    Args:
        class_dict (dict):
            {
                (machine, condition): [file_info, file_info, ...]
            }

        files_per_class (int):
            Number of files per class per batch

        shuffle (bool):
            Whether to shuffle files before batching

    Returns:
        batches (list of lists):
            [
                [file_info, file_info, ...],   # batch 0
                [file_info, file_info, ...],   # batch 1
                ...
            ]
    """

    # =========================
    # 1. Shuffle each class (optional)
    # =========================
    if shuffle:
        for key in class_dict:
            random.shuffle(class_dict[key])

    # =========================
    # 2. Find smallest class size
    # =========================
    min_files = min(len(files) for files in class_dict.values())

    if min_files < files_per_class:
        raise ValueError(
            f"Not enough files per class. "
            f"Smallest class has {min_files} files, "
            f"but files_per_class={files_per_class}"
        )

    # =========================
    # 3. Compute number of batches
    # =========================
    num_batches = min_files // files_per_class

    print(f"[INFO] Minimum files per class: {min_files}")
    print(f"[INFO] Files per class per batch: {files_per_class}")
    print(f"[INFO] Total batches: {num_batches}")

    # =========================
    # 4. Create batches
    # =========================
    batches = []

    for b in range(num_batches):
        batch = []

        for key, file_list in class_dict.items():
            start = b * files_per_class
            end = start + files_per_class

            selected_files = file_list[start:end]

            batch.extend(selected_files)

        batches.append(batch)

    return batches


def inspect_class_distribution(class_dict):
    """
    Utility function to print number of files per class
    """

    print("\n[INFO] Class distribution:")
    for key, files in class_dict.items():
        machine, condition = key
        print(f"{machine} - {condition}: {len(files)} files")


def inspect_batch(batch):
    """
    Debug utility to check class balance inside a batch
    """

    from collections import Counter

    counter = Counter()

    for item in batch:
        key = (item["machine"], item["label_name"])
        counter[key] += 1

    print("\n[INFO] Batch distribution:")
    for key, count in counter.items():
        print(f"{key}: {count}")