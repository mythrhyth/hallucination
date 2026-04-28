import os
import json
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


# ---------------- Config ----------------
DATA_PATH = "/kaggle/input/dataset/heuristics.json"
MODEL_NAME = "microsoft/deberta-v3-small"
MAX_SEQ_LEN = 256
BATCH_SIZE = 4
EPOCHS = 10
LR = 2e-5
SAVE_PATH = "/kaggle/working/best_recursive_model.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 2


# ---------------- Dataset ----------------
class ReasoningDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        context = sample.get("context", "")
        question = sample.get("question", "")
        steps = sample.get("reasoning_steps", [])
        labels = sample.get("step_labels", [])

        if isinstance(steps, str):
            steps = [steps]
        if not isinstance(labels, list):
            labels = list(labels)

        joined_steps = [f"Context: {context} Question: {question} Step: {step}" for step in steps]

        encoded = self.tokenizer(
            joined_steps,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        labels = torch.tensor(labels, dtype=torch.long)

        return {"input_ids": input_ids, "attention_mask": attention_mask}, labels


# ---------------- Collate fn ----------------
def collate_fn(batch):
    encs, labs = zip(*batch)
    max_steps = max(e["input_ids"].shape[0] for e in encs)
    seq_len = encs[0]["input_ids"].shape[1]

    padded_input_ids = []
    padded_attention = []
    padded_labels = []

    for e, l in zip(encs, labs):
        n_steps = e["input_ids"].shape[0]
        pad_steps = max_steps - n_steps

        if pad_steps > 0:
            pad_ids = torch.zeros(pad_steps, seq_len, dtype=torch.long)
            pad_mask = torch.zeros(pad_steps, seq_len, dtype=torch.long)
            input_ids = torch.cat([e["input_ids"], pad_ids], dim=0)
            attention_mask = torch.cat([e["attention_mask"], pad_mask], dim=0)
            labels = torch.cat([l, torch.full((pad_steps,), -100, dtype=torch.long)], dim=0)
        else:
            input_ids = e["input_ids"]
            attention_mask = e["attention_mask"]
            labels = l

        padded_input_ids.append(input_ids)
        padded_attention.append(attention_mask)
        padded_labels.append(labels)

    batch_input_ids = torch.stack(padded_input_ids)
    batch_attention = torch.stack(padded_attention)
    batch_labels = torch.stack(padded_labels)

    encodings = {"input_ids": batch_input_ids, "attention_mask": batch_attention}
    return encodings, batch_labels


# ---------------- Model ----------------
class RecursiveHybridModel(nn.Module):
    def __init__(self, model_name=MODEL_NAME, hidden_size=768, lstm_hidden=512, num_classes=NUM_CLASSES):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(hidden_size, lstm_hidden, num_layers=1, bidirectional=True, batch_first=True)
        self.attn = nn.Linear(lstm_hidden * 2, 1)
        self.classifier = nn.Linear(lstm_hidden * 2, num_classes)
        self.dropout = nn.Dropout(0.1)

    def forward(self, encodings):
        input_ids = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]
        B, S, L = input_ids.shape

        flat_input_ids = input_ids.view(B * S, L)
        flat_attention = attention_mask.view(B * S, L)

        outputs = self.encoder(input_ids=flat_input_ids, attention_mask=flat_attention)
        cls = outputs.last_hidden_state[:, 0, :]
        step_emb = cls.view(B, S, -1)
        step_emb = self.dropout(step_emb)

        lstm_out, _ = self.lstm(step_emb)
        attn_weights = torch.softmax(self.attn(lstm_out), dim=1)
        context_vec = (attn_weights * lstm_out).sum(dim=1, keepdim=True)
        fused = lstm_out + context_vec

        logits = self.classifier(fused)
        return logits


# ---------------- Utility ----------------
def compute_accuracy_stepwise(preds, labels, ignore_index=-100):
    preds = preds.detach().cpu()
    labels = labels.detach().cpu()
    mask = labels != ignore_index
    if mask.sum().item() == 0:
        return 0.0
    correct = (preds[mask] == labels[mask]).sum().item()
    total = mask.sum().item()
    return correct / total


# ---------------- Main ----------------
def main():
    with open(DATA_PATH, "r") as f:
        samples = json.load(f)

    train_data, temp_data = train_test_split(samples, test_size=0.3, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = ReasoningDataset(train_data, tokenizer, max_len=MAX_SEQ_LEN)
    val_dataset = ReasoningDataset(val_data, tokenizer, max_len=MAX_SEQ_LEN)
    test_dataset = ReasoningDataset(test_data, tokenizer, max_len=MAX_SEQ_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model = RecursiveHybridModel(model_name=MODEL_NAME).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        model.train()
        train_loss_total = 0.0
        train_acc_total = 0.0
        train_steps = 0

        print(f"Epoch {epoch+1}/{EPOCHS} - Training")
        for encodings, labels in tqdm(train_loader, desc="train"):
            encodings = {k: v.to(DEVICE) for k, v in encodings.items()}
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            logits = model(encodings)
            loss = loss_fn(logits.view(-1, NUM_CLASSES), labels.view(-1))
            preds = torch.argmax(logits, dim=-1)
            acc = compute_accuracy_stepwise(preds, labels)

            loss.backward()
            optimizer.step()

            train_loss_total += loss.item()
            train_acc_total += acc
            train_steps += 1

        avg_train_loss = train_loss_total / train_steps
        avg_train_acc = train_acc_total / train_steps

        model.eval()
        val_loss_total = 0.0
        val_acc_total = 0.0
        val_steps = 0

        with torch.no_grad():
            for encodings, labels in tqdm(val_loader, desc="val"):
                encodings = {k: v.to(DEVICE) for k, v in encodings.items()}
                labels = labels.to(DEVICE)

                logits = model(encodings)
                loss = loss_fn(logits.view(-1, NUM_CLASSES), labels.view(-1))
                preds = torch.argmax(logits, dim=-1)
                acc = compute_accuracy_stepwise(preds, labels)

                val_loss_total += loss.item()
                val_acc_total += acc
                val_steps += 1

        avg_val_loss = val_loss_total / val_steps
        avg_val_acc = val_acc_total / val_steps

        print(f"Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.4f}")
        print(f"Val   Loss: {avg_val_loss:.4f} | Val   Acc: {avg_val_acc:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"New best model saved (Val Loss: {avg_val_loss:.4f})")

    print("Loading best model for final evaluation...")
    model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE))
    model.eval()

    test_loss_total = 0.0
    test_acc_total = 0.0
    test_steps = 0

    with torch.no_grad():
        for encodings, labels in tqdm(test_loader, desc="test"):
            encodings = {k: v.to(DEVICE) for k, v in encodings.items()}
            labels = labels.to(DEVICE)

            logits = model(encodings)
            loss = loss_fn(logits.view(-1, NUM_CLASSES), labels.view(-1))
            preds = torch.argmax(logits, dim=-1)
            acc = compute_accuracy_stepwise(preds, labels)

            test_loss_total += loss.item()
            test_acc_total += acc
            test_steps += 1

    avg_test_loss = test_loss_total / test_steps
    avg_test_acc = test_acc_total / test_steps

    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Test Loss: {avg_test_loss:.4f} | Test Accuracy: {avg_test_acc:.4f}")


if __name__ == "__main__":
    main()
