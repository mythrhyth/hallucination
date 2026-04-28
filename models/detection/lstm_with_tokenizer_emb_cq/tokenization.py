# tokenize.py
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class ReasoningDataset(Dataset):
    def __init__(self, data, pretrained_model="bert-base-uncased", max_cq_len=64, max_step_len=32):
        self.data = data
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.max_cq_len = max_cq_len
        self.max_step_len = max_step_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        context_question = item["context"] + " " + item["question"]
        steps = item["reasoning_steps"]
        step_labels = item["step_labels"]

        # Tokenize context+question
        cq_tokens = self.tokenizer(
            context_question,
            truncation=True,
            padding="max_length",
            max_length=self.max_cq_len,
            return_tensors="pt"
        )

        # Tokenize steps
        steps_tokens = self.tokenizer(
            steps,
            truncation=True,
            padding="max_length",
            max_length=self.max_step_len,
            return_tensors="pt"
        )

        return {
            "cq_input_ids": cq_tokens["input_ids"].squeeze(0),
            "cq_attention_mask": cq_tokens["attention_mask"].squeeze(0),
            "steps_input_ids": steps_tokens["input_ids"],
            "steps_attention_mask": steps_tokens["attention_mask"],
            "step_labels": torch.tensor(step_labels, dtype=torch.long)
        }

def collate_fn(batch):
    # Batch context+question
    cq_input_ids = torch.stack([b["cq_input_ids"] for b in batch])
    cq_attention_mask = torch.stack([b["cq_attention_mask"] for b in batch])

    # Batch steps
    max_steps = max(b["steps_input_ids"].shape[0] for b in batch)
    max_step_len = max(b["steps_input_ids"].shape[1] for b in batch)
    B = len(batch)

    steps_input_ids = torch.zeros(B, max_steps, max_step_len, dtype=torch.long)
    steps_attention_mask = torch.zeros(B, max_steps, max_step_len, dtype=torch.long)
    step_labels = torch.zeros(B, max_steps, dtype=torch.long)

    for i, b in enumerate(batch):
        n_steps, s_len = b["steps_input_ids"].shape
        steps_input_ids[i, :n_steps, :s_len] = b["steps_input_ids"]
        steps_attention_mask[i, :n_steps, :s_len] = b["steps_attention_mask"]
        step_labels[i, :n_steps] = b["step_labels"]

    return cq_input_ids, cq_attention_mask, steps_input_ids, steps_attention_mask, step_labels
