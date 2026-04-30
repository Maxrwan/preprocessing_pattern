""" 
    DATASET 
"""

import os

def get_classwise_files(base_path):
    class_dict = {}

    for machine in os.listdir(base_path):
        machine_path = os.path.join(base_path, machine, "machine_data")

        if not os.path.isdir(machine_path):
            continue

        for condition in os.listdir(machine_path):
            condition_path = os.path.join(machine_path, condition)

            if not os.path.isdir(condition_path):
                continue

            key = (machine, condition)
            class_dict[key] = []

            for file in os.listdir(condition_path):
                if file.endswith(".wav"):
                    full_path = os.path.join(condition_path, file)

                    class_dict[key].append({
                        "file_path": full_path,
                        "machine": machine,
                        "label_name": condition
                    })

    return class_dict