from fastapi import APIRouter

import json
from pydantic import BaseModel
import torch
import torch.nn.functional as F

from models.detection.lstm_with_embedding.model import LSTM_Based_Detector
from config import VOCAB_PATH, MODEL_SAVE_PATH

lstm_emb_router = APIRouter(prefix="/lstm_emb", tags=["lstm_emb"])

class InferenceInput(BaseModel):
    context: str
    question: str
    reasoning_steps: list[str]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(VOCAB_PATH / "lstm_emb.json") as f:
    vocab = json.load(f)
word2idx = {w: i for i, w in enumerate(vocab)}

def tokenize(sentence, max_len=50):
    tokens = sentence.lower().split()
    ids = [word2idx.get(tok, 1) for tok in tokens[:max_len]]  # 1 → OOV
    ids += [0] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)

embedding_dim, hidden_dim = 100, 128
vocab_size = len(vocab)

model = LSTM_Based_Detector(embedding_dim, hidden_dim, vocab_size)
model.load_state_dict(torch.load(MODEL_SAVE_PATH / "lstm_emb/model.pt", map_location=device))
model.to(device)
model.eval()

@lstm_emb_router.post("/predict")
def predict(input_data:InferenceInput):
    context = input_data.context
    question = input_data.question
    steps = input_data.reasoning_steps

    cq_text = f"{context} {question}"
    cq_tensor = tokenize(cq_text).unsqueeze(0).to(device)  # [1, seq_len]

    step_tensors = [tokenize(s) for s in steps]
    steps_tensor = torch.stack(step_tensors).unsqueeze(0).to(device)  # [1, N, seq_len]

    with torch.no_grad():
        logits = model(cq_tensor, steps_tensor)  # [1, N, 2]
        probs = F.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1).squeeze(0).tolist()

    results = []
    for step, label, prob in zip(steps, preds, probs.squeeze(0)):
        results.append({
            "step": step,
            "prediction": "consistent" if label == 0 else "inconsistent",
            "confidence": round(prob[label].item(), 4)
        })

    return {
        "context": context,
        "question": question,
        "results": results
    }
