from collections import Counter
import re
from torch.utils.data import Dataset
import torch
from torch.nn.utils.rnn import pad_sequence

from models.common import load_data
from config import DATA_PAATH, VOCAB_PATH
import json

PAD_IDX = 0
UNK_IDX = 1


def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

def build_vocab(data, min_freq=1):

    # print(data)
    count = 0
    counter = Counter()
    for item in data:
        # print(item)
        # count+= 1
        # print(count)
        counter.update(tokenize(item["context"]))
        counter.update(tokenize(item["question"]))
        for step in item["reasoning_steps"]:
            counter.update(tokenize(step))
    return {word: idx + 2 for idx, (word, freq) in enumerate(counter.items()) if freq >= min_freq}

class ReasoningDataset(Dataset):
    def __init__(self, data, vocab):
        self.data = data
        self.vocab = vocab

    def text_to_ids(self, text):
        return [self.vocab.get(w, UNK_IDX) for w in tokenize(text)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        reasoning_steps = item["reasoning_steps"]
        step_labels = item["step_labels"]
        if len(reasoning_steps) != len(step_labels):
            return None
        context_question = torch.tensor(self.text_to_ids(item["context"] + " " + item["question"]), dtype=torch.long)
        reasoning_steps = [torch.tensor(self.text_to_ids(step), dtype=torch.long) for step in item["reasoning_steps"]]
        step_labels = torch.tensor(item["step_labels"], dtype=torch.long)
        return context_question, reasoning_steps, step_labels



def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    context_qn, steps, step_labels = zip(*batch)

    # Pad context_question
    cq_padded = pad_sequence(context_qn, batch_first=True, padding_value=0)

    # Determine max number of reasoning steps in batch
    max_num_steps = max(len(s) for s in steps)

    # Pad steps and step labels per sample
    padded_steps = []
    for s in steps:
        # pad each step sequence in this sample
        s_padded = pad_sequence(s, batch_first=True, padding_value=0)
        # if fewer steps than max_num_steps, pad extra empty steps
        if len(s_padded) < max_num_steps:
            pad_shape = (max_num_steps - len(s_padded), s_padded.shape[1])
            pad_extra = torch.zeros(pad_shape, dtype=torch.long)
            s_padded = torch.cat([s_padded, pad_extra], dim=0)
        padded_steps.append(s_padded)

    # Now pad across samples to form [batch, max_steps, step_len]
    max_step_len = max(s.shape[1] for s in padded_steps)
    final_steps = torch.zeros(len(padded_steps), max_num_steps, max_step_len, dtype=torch.long)
    for i, s in enumerate(padded_steps):
        step_count, step_len = s.shape
        final_steps[i, :step_count, :step_len] = s

    # Pad step labels similarly
    max_label_len = max(len(l) for l in step_labels)
    step_labels_padded = torch.zeros(len(step_labels), max_label_len, dtype=torch.long)
    for i, l in enumerate(step_labels):
        step_labels_padded[i, :len(l)] = l

    return cq_padded, final_steps, step_labels_padded

if __name__ == "__main__":
    data =  load_data(DATA_PAATH)
    vocab = build_vocab(data)

    with open(VOCAB_PATH, "w") as f:
        json.dump(vocab, f)
