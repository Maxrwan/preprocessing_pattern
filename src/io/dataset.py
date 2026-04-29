""" 
    DATASET 
"""

import os

def get_all_files(base_path):
    data = []

    for machine in os.listdir(base_path):
        machine_path = os.path.join(base_path, machine, "machine_data")

        if not os.path.isdir(machine_path):
            continue

        for label_name in os.listdir(machine_path):
            label_path = os.path.join(machine_path, label_name)

            if not os.path.isdir(label_path):
                continue

            for file in os.listdir(label_path):
                if file.endswith(".wav"):
                    full_path = os.path.join(label_path, file)

                    data.append({
                        "file_path": full_path,
                        "label_name": label_name,
                        "machine": machine
                    })

    return data