from fastapi import APIRouter
from pydantic import BaseModel
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from models.detection.recursive_hybrid.model import RecursiveHybridModel
from config import MODEL_NAME, RECURSIVE_MODEL_PATH, DEVICE


# ---------------- Router ----------------
recursive_router = APIRouter(prefix="/recursive_hybrid", tags=["recursive_hybrid"])


# ---------------- Request Schema ----------------
class InferenceInput(BaseModel):
    context: str
    question: str
    reasoning_steps: list[str]


# ---------------- Load Model ----------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = RecursiveHybridModel(model_name=MODEL_NAME)
model.load_state_dict(torch.load(RECURSIVE_MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

print(f"✅ Recursive Hybrid Model loaded from: {RECURSIVE_MODEL_PATH}")


# ---------------- Prediction Endpoint ----------------
@recursive_router.post("/predict")
def predict(input_data: InferenceInput):
    """
    Perform hallucination consistency detection using Recursive Hybrid Model.
    """
    context = input_data.context
    question = input_data.question
    steps = input_data.reasoning_steps

    combined_text = f"{context} {question}"

    # Tokenize context + question
    encodings = tokenizer(
        combined_text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=256
    )

    # Tokenize reasoning steps
    step_encodings = [
        tokenizer(
            step,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=256
        ) for step in steps
    ]

    # Stack the encoded steps
    step_input_ids = torch.stack([s["input_ids"].squeeze(0) for s in step_encodings])  # [N, seq_len]
    step_attention_masks = torch.stack([s["attention_mask"].squeeze(0) for s in step_encodings])  # [N, seq_len]

    # Send to device
    encodings = {k: v.to(DEVICE) for k, v in encodings.items()}
    step_input_ids = step_input_ids.unsqueeze(0).to(DEVICE)           # [1, N, seq_len]
    step_attention_masks = step_attention_masks.unsqueeze(0).to(DEVICE)  # [1, N, seq_len]

    # Forward pass
    with torch.no_grad():
        logits = model(encodings, (step_input_ids, step_attention_masks))  # depends on your model forward()
        probs = F.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1).squeeze(0).tolist()

    # Format output
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
