import os
import json
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
import numpy as np

from model import RecursiveHybridModel
from tokenization import ReasoningDataset, collate_fn

# Configuration
class TrainingConfig:
    DATA_PATH = "/kaggle/input/dataset/heuristics.json"
    MODEL_NAME = "microsoft/deberta-v3-small"
    MAX_SEQ_LEN = 256
    BATCH_SIZE = 4
    EPOCHS = 10
    LR = 2e-5
    SAVE_PATH = "/kaggle/working/best_recursive_model.pt"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_CLASSES = 2

def compute_accuracy_stepwise(preds, labels, ignore_index=-100):
    """Compute accuracy while ignoring padding tokens"""
    preds = preds.detach().cpu()
    labels = labels.detach().cpu()
    mask = labels != ignore_index
    if mask.sum().item() == 0:
        return 0.0
    correct = (preds[mask] == labels[mask]).sum().item()
    total = mask.sum().item()
    return correct / total

def train_epoch(model, train_loader, optimizer, loss_fn, device):
    """Train for one epoch"""
    model.train()
    train_loss_total = 0.0
    train_acc_total = 0.0
    train_steps = 0

    for encodings, labels in tqdm(train_loader, desc="Training"):
        encodings = {k: v.to(device) for k, v in encodings.items()}
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(encodings)
        loss = loss_fn(logits.view(-1, TrainingConfig.NUM_CLASSES), labels.view(-1))
        preds = torch.argmax(logits, dim=-1)
        acc = compute_accuracy_stepwise(preds, labels)

        loss.backward()
        optimizer.step()

        train_loss_total += loss.item()
        train_acc_total += acc
        train_steps += 1

    return train_loss_total / train_steps, train_acc_total / train_steps

def evaluate_model(model, data_loader, loss_fn, device, desc="Validation"):
    """Evaluate model on validation/test set"""
    model.eval()
    loss_total = 0.0
    acc_total = 0.0
    steps = 0

    with torch.no_grad():
        for encodings, labels in tqdm(data_loader, desc=desc):
            encodings = {k: v.to(device) for k, v in encodings.items()}
            labels = labels.to(device)

            logits = model(encodings)
            loss = loss_fn(logits.view(-1, TrainingConfig.NUM_CLASSES), labels.view(-1))
            preds = torch.argmax(logits, dim=-1)
            acc = compute_accuracy_stepwise(preds, labels)

            loss_total += loss.item()
            acc_total += acc
            steps += 1

    return loss_total / steps, acc_total / steps

def main():
    config = TrainingConfig()
    
    # Load and split data
    print("Loading dataset...")
    with open(config.DATA_PATH, "r") as f:
        samples = json.load(f)

    train_data, temp_data = train_test_split(samples, test_size=0.3, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

    # Initialize tokenizer and datasets
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)

    train_dataset = ReasoningDataset(train_data, tokenizer, max_len=config.MAX_SEQ_LEN)
    val_dataset = ReasoningDataset(val_data, tokenizer, max_len=config.MAX_SEQ_LEN)
    test_dataset = ReasoningDataset(test_data, tokenizer, max_len=config.MAX_SEQ_LEN)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, 
                             shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, 
                           shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, 
                            shuffle=False, collate_fn=collate_fn)

    # Initialize model, optimizer, and loss function
    print("Initializing model...")
    model = RecursiveHybridModel(model_name=config.MODEL_NAME).to(config.DEVICE)
    optimizer = AdamW(model.parameters(), lr=config.LR)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    best_val_loss = float("inf")
    training_history = []

    # Training loop
    for epoch in range(config.EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.EPOCHS}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, loss_fn, config.DEVICE)
        
        # Validate
        val_loss, val_acc = evaluate_model(model, val_loader, loss_fn, config.DEVICE, "Validation")
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), config.SAVE_PATH)
            print(f"New best model saved (Val Loss: {val_loss:.4f})")
        
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        })

    # Final evaluation on test set
    print("\nLoading best model for final evaluation...")
    model.load_state_dict(torch.load(config.SAVE_PATH, map_location=config.DEVICE))
    
    test_loss, test_acc = evaluate_model(model, test_loader, loss_fn, config.DEVICE, "Test")
    
    print(f"\n{'='*50}")
    print(f"Training Complete!")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")
    print(f"{'='*50}")

    return training_history, best_val_loss, test_acc

if __name__ == "__main__":
    main()