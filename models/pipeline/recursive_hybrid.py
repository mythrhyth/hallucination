# ================================================================
# Recursive Hybrid Hallucination Detection Pipeline
# ================================================================

# ---- Custom model function imports ----
from models.common import load_data, check_data_integrity
from injectionModel.heuristics import generate_pure_dataset
from models.detection.recursive_hybrid.train import train_recursive_hybrid_model  # <-- new train function

# ---- Universal library imports ----
import json
import os


from config import DATA_PATH, MODEL_SAVE_PATH, VOCAB_PATH, RECURSIVE_MODEL_PATH



if __name__ == "__main__":
    # Define all relevant paths
    injection_save_path = MODEL_SAVE_PATH / "injected_data" / "heuristics.json"
    detection_save_path = RECURSIVE_MODEL_PATH  # 
    vocab_path = VOCAB_PATH / "recursive_hybrid.json"

   
    if os.path.exists(injection_save_path):
        with open(injection_save_path, "r") as f:
            data = json.load(f)
        print(f"✅ Loaded existing injected dataset of length: {len(data)}")
    else:
        print("⚙️ Injected dataset not found — generating from annotated data...")
        data = load_data(DATA_PATH)
        data = check_data_integrity(data)
        data += generate_pure_dataset(
            data,
            output_file=injection_save_path,
            save_output=True
        )
        with open(injection_save_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved injected dataset to: {injection_save_path}")

   
    print("\n🚀 Starting Recursive Hybrid Model training...")
    train_recursive_hybrid_model(
        data=data,
        save_path=detection_save_path,
        vocab_path=vocab_path,
        batch_size=4,
        epochs=10,
        lr=2e-5,
        max_seq_len=256
    )
    print(f"\nModel training complete! Saved to: {detection_save_path}")
