""" 
    BATCHING FILE 
"""

def create_balanced_batches(class_dict, files_per_class=10):
    """
    Creates batches with equal number of files from each class
    """

    import math

    # Find minimum class size
    min_files = min(len(v) for v in class_dict.values())

    num_batches = min_files // files_per_class

    batches = []

    for b in range(num_batches):
        batch = []

        for key in class_dict:
            start = b * files_per_class
            end = start + files_per_class

            batch.extend(class_dict[key][start:end])

        batches.append(batch)

    return batches