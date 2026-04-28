from sklearn.metrics import accuracy_score
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
import json

from tokenization import NLIStepDataset, collate_fn, build_vocab
from model import NLIModel, GlobalClassifier

def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def train_epoch_step(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for premise, hypothesis, labels, _ in dataloader:
        premise, hypothesis, labels = premise.to(device), hypothesis.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(premise, hypothesis)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds = output.argmax(dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().tolist())
    acc = accuracy_score(all_labels, all_preds)
    return total_loss / len(dataloader), acc


def train_epoch_global(step_model, global_model, dataloader, criterion, optimizer, device):
    step_model.eval()
    global_model.train()
    total_loss, all_preds, all_labels = 0, [], []
    for premise, hypothesis, _, global_labels in dataloader:
        premise, hypothesis, global_labels = premise.to(device), hypothesis.to(device), global_labels.to(device)

        with torch.no_grad():
            step_output = step_model(premise, hypothesis)  # log-probs
            step_probs = torch.exp(step_output)[:, 1]  # hallucination probs

            # aggregate step hallucination signals
            global_feature = step_probs.unsqueeze(1)  # (batch=1, feature=1)

        # train global classifier
        optimizer.zero_grad()
        global_out = global_model(global_feature)
        loss = criterion(global_out, global_labels.to(device))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = global_out.argmax(dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(global_labels.tolist())

    acc = accuracy_score(all_labels, all_preds)
    return total_loss / len(dataloader), acc


def evaluate_step(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for premise, hypothesis, labels, _ in dataloader:
            premise, hypothesis, labels = premise.to(device), hypothesis.to(device), labels.to(device)
            output = model(premise, hypothesis)
            loss = criterion(output, labels)
            total_loss += loss.item()
            preds = output.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().tolist())
    acc = accuracy_score(all_labels, all_preds)
    return total_loss / len(dataloader), acc


def main(train_data, test_data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    #loading data
    #building vocab
    vocab = build_vocab(train_data)
    vocab_size = len(vocab) + 2  # plus PAD and UNK

    #preparing dataset
    train_dataset = NLIStepDataset(train_data, vocab)
    test_dataset = NLIStepDataset(test_data, vocab)


    #preparing data loader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    #NLI model for granular detection
    step_model = NLIModel(vocab_size).to(device)
    step_criterion = nn.NLLLoss()
    step_optimizer = optim.Adam(step_model.parameters(), lr=0.001)

    epochs = 10
    print("Training step level NLI model")
    for epoch in range(1, epochs+1):
        train_loss, train_acc = train_epoch_step(step_model, train_loader, step_criterion, step_optimizer, device)
        print(f"[Step NLI] Epoch {epoch}: Train Loss {train_loss:.4f}, Train Acc {train_acc:.4f}")

    #Global classifier model
    global_model = GlobalClassifier(hidden_dim=32).to(device)
    global_criterion = nn.NLLLoss()
    global_optimizer = optim.Adam(global_model.parameters(), lr=0.001)
    print("\n### Training Global Classifier ###")
    global_epochs = 5
    for epoch in range(1, global_epochs+1):
        global_loss, global_acc = train_epoch_global(step_model, global_model, train_loader, global_criterion, global_optimizer, device)
        print(f"[Global Model] Epoch {epoch}: Train Loss {global_loss:.4f}, Train Accuracy {global_acc:.4f}")

    # Final evaluation
    # print("\n### Final Evaluation of Global Classifier ###")

    print("\n### Final Evaluation of Seep NLI Classifier ###")
    step_test_loss, step_test_acc =  evaluate_step(step_model, test_loader, step_criterion, device)
    print(f"[Step NLI Model] Test Loss {step_test_loss:.4f}, Test Acc {step_test_acc:.4f}")

    glob_test_loss, glob_test_acc = train_epoch_global(step_model, global_model, test_loader, global_criterion, global_optimizer, device)
    print(f"[Global Model] Test Loss {glob_test_loss:.4f}, Test Acc {glob_test_acc:.4f}")

if __name__ == "__main__":
        train_file = ["../../../data/basic_language/gpt.json","../../../data/basic_language/gpt1.json"]
        test_file = "../../../data/basic_language/test.json"
        train_data = []
        for file in train_file:
            train_data += load_data(file)
        test_data = load_data(test_file)

        main(train_data, test_data)
