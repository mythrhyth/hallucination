from pathlib import Path
import torch

ROOT = Path(__file__).parent

DATA_PAATH = ROOT / "data" / "merged_data.json"
GLOVE_PATH = ROOT / "mlruns" / "GloVe" / "glove.6B.100d.txt"
VOCAB_PATH = ROOT / "mlruns" / "vocab"

RECURSIVE_MODEL_DIR = ROOT / "mlruns" / "recursive_hybrid"
RECURSIVE_MODEL_PATH = RECURSIVE_MODEL_DIR / "best_recursive_model.pt"

MODEL_SAVE_PATH = ROOT / "mlruns"
MODEL_NAME = "microsoft/deberta-v3-small"
MAX_SEQ_LEN = 256
NUM_CLASSES = 2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
