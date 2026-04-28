import random
import json

def train_test_split(array, test_size=0.2):
    random.shuffle(array)
    split_index = int(len(array) * (1 - test_size))
    return array[:split_index], array[split_index:]

def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def check_data_integrity(data):
    count_missed_data = 0
    clean_data = list()
    for item in data:
        if len(item["reasoning_steps"]) != len(item["step_labels"]):
            count_missed_data += 1
            continue
        clean_data.append(item)
    print(f"{count_missed_data} faaulty data removed")
    return clean_data
