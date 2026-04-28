import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

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

        # Ensure steps and labels are lists
        if isinstance(steps, str):
            steps = [steps]
        if not isinstance(labels, list):
            labels = list(labels)

        # Create input strings for each step
        joined_steps = [f"Context: {context} Question: {question} Step: {step}" for step in steps]

        # Tokenize all steps
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

def collate_fn(batch):
    """Custom collate function to handle variable number of reasoning steps"""
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
            # Create padding tensors
            pad_ids = torch.zeros(pad_steps, seq_len, dtype=torch.long)
            pad_mask = torch.zeros(pad_steps, seq_len, dtype=torch.long)
            
            # Concatenate original with padding
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

    # Stack all batches
    batch_input_ids = torch.stack(padded_input_ids)
    batch_attention = torch.stack(padded_attention)
    batch_labels = torch.stack(padded_labels)

    encodings = {"input_ids": batch_input_ids, "attention_mask": batch_attention}
    return encodings, batch_labels