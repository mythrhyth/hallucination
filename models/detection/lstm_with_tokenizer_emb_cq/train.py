# train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.detection.lstm_with_tokenizer_emb_cq.model import LSTM_Based_Detector_BERT
from models.detection.lstm_with_tokenizer_emb_cq.tokenization import ReasoningDataset, collate_fn

from config import MODEL_SAVE_PATH

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(data, save_path,epochs=10, batch_size=4, lr=1e-4):

    dataset = ReasoningDataset(data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    model = LSTM_Based_Detector_BERT().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        for cq_input_ids, cq_attention_mask, steps_input_ids, steps_attention_mask, step_labels in loader:
            cq_input_ids = cq_input_ids.to(device)
            cq_attention_mask = cq_attention_mask.to(device)
            steps_input_ids = steps_input_ids.to(device)
            steps_attention_mask = steps_attention_mask.to(device)
            step_labels = step_labels.to(device)

            optimizer.zero_grad()
            step_logits = model(cq_input_ids, cq_attention_mask, steps_input_ids, steps_attention_mask)
            loss = criterion(step_logits.view(-1,2), step_labels.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), save_path)
    print("Model saved as model.pt")

if __name__ == "__main__":
    train_model("data.json")
