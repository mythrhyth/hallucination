from models.common import load_data, check_data_integrity, train_test_split
from models.injection.heuristics import generate_pure_dataset
from models.detection.NLI_aggrigate.training import main_training_loop
#universl librries import
import json
import os

#config import
from config import DATA_PAATH,  MODEL_SAVE_PATH, VOCAB_PATH

if __name__ == "__main__":
    if __name__ == "__main__":
        injection_save_path = MODEL_SAVE_PATH / "injected_data/heuristics.json"
        detection_save_path = MODEL_SAVE_PATH / "nli/model.pt"
        vocab_path = VOCAB_PATH / "nli.json"

        if os.path.exists(injection_save_path):
            with open(injection_save_path, "r") as f:
                data = json.load(f)
            print("Loaded existing array of length :", len(data))
        else:
            print("couldn't find the pre injected data\ncreating injected data from annotated data")
            data = load_data(DATA_PAATH)
            data = check_data_integrity(data)
            data += generate_pure_dataset(data,output_file=injection_save_path, save_output=True)
            with open(injection_save_path, 'w') as f:
                json.dump(data, f, indent=2)
                print(f"saved to {injection_save_path}")
        train_data, test_data = train_test_split(data)


        main_training_loop(train_data,test_data,vocab_path,detection_save_path, epochs=20)
