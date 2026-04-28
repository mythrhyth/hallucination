# training.py
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.detection.lstm_attention.tokenization import build_vocab, ReasoningDataset, collate_fn, PAD_IDX, UNK_IDX
from models.detection.lstm_attention.model import LSTM_Attention_HallucinationDetector
import random

random.seed(42)
torch.manual_seed(42)

# -----------------------------
# GloVe loader
# -----------------------------
def load_glove_embeddings(glove_path, vocab, embedding_dim=100):
    embeddings = torch.randn(len(vocab)+2, embedding_dim) * 0.01
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            vector = torch.tensor([float(x) for x in parts[1:]], dtype=torch.float)
            if word in vocab:
                embeddings[vocab[word]] = vector
    return embeddings

# -----------------------------
# Train/Test split
# -----------------------------
def train_test_split(data, test_size=0.2):
    random.shuffle(data)
    split_idx = int(len(data) * (1-test_size))
    return data[:split_idx], data[split_idx:]

# -----------------------------
# Training function
# -----------------------------
def train_model(model, train_loader, criterion, optimizer, device, epochs=10):
    model.to(device)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for cq, steps, step_labels in train_loader:
            if cq is None:
                continue
            cq, steps, step_labels = cq.to(device), steps.to(device), step_labels.to(device)
            optimizer.zero_grad()
            step_logits = model(cq, steps)                      # [B, N, 2]

            # flatten for loss
            loss = criterion(step_logits.view(-1, 2), step_labels.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}] | Step Loss: {total_loss/len(train_loader):.4f}")

# -----------------------------
# Evaluation
# -----------------------------
def evaluate_model(model, test_loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for cq, steps, step_labels in test_loader:
            if cq is None:
                continue
            cq, steps, step_labels = cq.to(device), steps.to(device), step_labels.to(device)
            step_logits = model(cq, steps)
            preds = step_logits.argmax(dim=-1)
            correct += (preds == step_labels).sum().item()
            total += step_labels.numel()
    acc = correct / total if total > 0 else 0
    print(f"Test Step-Level Accuracy: {acc:.4f}")

# -----------------------------
# Main training loop
# -----------------------------
def main_training_loop(data, glove_path,vocab_path, save_path, embedding_dim=100, hidden_dim=64, batch_size=16, epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data, test_data = train_test_split(data)

    # Build vocab from training set
    vocab = build_vocab(train_data)
    print(f"Vocab size: {len(vocab)}")
    vocab["<IDX>"] = PAD_IDX
    vocab["<UNK>"] = UNK_IDX
    with open(vocab_path, "w") as f:
        json.dump(vocab, f)

    # Load GloVe embeddings
    pretrained_embeddings = load_glove_embeddings(glove_path, vocab, embedding_dim)
    print("Loaded pretrained embeddings.")

    # Datasets & loaders
    train_dataset = ReasoningDataset(train_data, vocab)
    test_dataset = ReasoningDataset(test_data, vocab)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Model
    model = LSTM_Attention_HallucinationDetector(
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        vocab_size=len(vocab)+2,
        pretrained_embeddings=pretrained_embeddings,
        freeze_embeddings=False
    )

    # Loss & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train
    train_model(model, train_loader, criterion, optimizer, device, epochs=epochs)

    # Evaluate
    evaluate_model(model, test_loader, device)

    # Save model
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    data_path = "./data/reasoning_dataset.json"
    glove_path = "./GloVe/glove.6B.100d.txt"
    save_path = "./model.pt"

    main_training_loop(data_path, glove_path, save_path)
