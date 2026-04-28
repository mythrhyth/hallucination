# train_glove.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import json
import random
from models.detection.lstm_with_embedding.model import LSTM_Based_Detector
from models.detection.lstm_with_embedding.tokenization import build_vocab, ReasoningDataset, collate_fn, PAD_IDX, UNK_IDX
from config import VOCAB_PATH

random.seed(42)

# GloVe Loading Function

def load_glove_embeddings(glove_path, vocab, embedding_dim=100):
    print(f"Loading GloVe embeddings from {glove_path}...")
    embeddings_index = {}
    with open(glove_path, "r", encoding="utf8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype="float32")
            embeddings_index[word] = vector

    embedding_matrix = np.zeros((len(vocab) + 2, embedding_dim))
    oov_count = 0

    for word, idx in vocab.items():
        if word in embeddings_index:
            embedding_matrix[idx] = embeddings_index[word]
        else:
            embedding_matrix[idx] = np.random.normal(scale=0.6, size=(embedding_dim,))
            oov_count += 1

    print(f"Loaded embeddings for {len(vocab) - oov_count} words, {oov_count} OOV.")
    return torch.tensor(embedding_matrix, dtype=torch.float32)

# Training Helpers

def train_model(model, train_loader, criterion_step, optimizer, device, epochs=10):
    model.to(device)
    for epoch in range(epochs):
        model.train()
        step_loss = 0
        for cq, steps, step_labels in train_loader:
            cq, steps, step_labels = cq.to(device), steps.to(device), step_labels.to(device)
            optimizer.zero_grad()
            step_logits = model(cq, steps)
            loss_step = criterion_step(step_logits.view(-1, step_logits.shape[-1]), step_labels.view(-1))      # ensures shape [B]
            loss_step.backward()
            optimizer.step()
            step_loss += loss_step.item()
            # print("step_logits:", step_logits.shape)
            # print("step_labels:", step_labels.shape)
            # print("After view:", step_logits.view(-1, step_logits.shape[-1]).shape, step_labels.view(-1).shape)

        avg_loss = step_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}] | Step Loss: {avg_loss:.4f}")

def train_test_split(array, test_size=0.2):
    random.shuffle(array)
    split_index = int(len(array) * (1 - test_size))
    return array[:split_index], array[split_index:]

def evaluate_model(model, test_loader, device):
    model.eval()
    total, correct = 0, 0

    with torch.no_grad():
        for batch in test_loader:
            if batch is None:
                continue
            cq, steps, step_labels = batch
            cq, steps, step_labels = cq.to(device), steps.to(device), step_labels.to(device)

            step_logits = model(cq, steps)  # [B, N, 2]
            step_preds = step_logits.argmax(dim=-1)  # [B, N]

            # Mask out padded steps
            mask = step_labels != 0  # assuming 0 is padding
            total += mask.sum().item()
            correct += (step_preds == step_labels).masked_select(mask).sum().item()

    accuracy = correct / total if total > 0 else 0
    print(f"Test Step-Level Accuracy: {accuracy:.4f}")
    return accuracy

# Main Function

def main_training_loop(data, glove_path, save_path,vocab_path, embedding_dim = 100, hidden_dim = 64,freeze_emb = False, epoch = 10 ):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data, test_data = train_test_split(data)
    vocab = build_vocab(train_data)
    vocab_cp = vocab.copy()
    print(f"Vocab size: {len(vocab)}")
    vocab_cp["<PAD>"] = PAD_IDX
    vocab_cp["<UNK>"] = UNK_IDX
    with open(vocab_path, "w") as f:
        json.dump(vocab_cp, f)
    pretrained_embeddings = load_glove_embeddings(glove_path, vocab, embedding_dim)


    train_dataset = ReasoningDataset(train_data, vocab)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

    test_dataset = ReasoningDataset(test_data, vocab)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

    model = LSTM_Based_Detector(
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        vocab_size=len(vocab) + 2,
        pretrained_embeddings=pretrained_embeddings,
        freeze_embeddings=freeze_emb  # You can freeze if you want static GloVe,
    )

    criterion_step = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_model(model, train_loader, criterion_step, optimizer, device, epochs=epoch)
    evaluate_model(model, test_loader, device)

    torch.save(model.state_dict(), save_path)
    print(f"Training complete. Model saved to {save_path}")

if __name__ == "__main__":
    data_path = "../../../data/merged_data.json"
    glove_path = "./GloVe/glove.6B.100d.txt"
    save_path = "./mlruns"
    main_training_loop(data_path, glove_path, save_path)
