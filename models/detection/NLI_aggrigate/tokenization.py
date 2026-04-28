import re
from collections import Counter
import torch
from torch.utils.data import Dataset



def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

def build_vocab(data, min_freq = 1):
    counter = Counter()
    for item in data:
        for step in item["reasoning_steps"]:
            counter.update(tokenize(step))
    return {word : idx+2 for idx, (word, freq) in enumerate(counter.items()) if freq >= min_freq}

PAD_IDX = 0
UNK_IDX = 1



class NLIStepDataset(Dataset):

    def __init__(self, data, vocab):
        self.samples = []
        for item in data:
            steps = item["reasoning_steps"]
            labels = item["step_labels"]

            for i in range(len(steps) - 1):
                premise = steps[i]
                hypothesis = steps[i + 1]
                label = labels[i + 1]
                self.samples.append((premise, hypothesis, label, item["final_answer_correct"]))
        self.vocab = vocab

    def text_to_ids(self, text):
        return [self.vocab.get(w, UNK_IDX) for w in tokenize(text)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        premise, hypothesis, step_label, global_label = self.samples[idx]
        premise_ids = self.text_to_ids(premise)
        hypothesis_ids = self.text_to_ids(hypothesis)
        return torch.tensor(premise_ids), torch.tensor(hypothesis_ids), torch.tensor(step_label), torch.tensor(global_label)

def collate_fn(batch):
    premises, hypotheses, step_labels, global_labels = zip(*batch)
    prem_lens = [len(x) for x in premises]
    hyp_lens = [len(x) for x in hypotheses]

    max_prem = max(prem_lens)
    max_hyp = max(hyp_lens)

    prem_padded = torch.zeros(len(batch), max_prem, dtype=torch.long)
    hyp_padded = torch.zeros(len(batch), max_hyp, dtype=torch.long)

    for i, (p, h) in enumerate(zip(premises, hypotheses)):
        prem_padded[i, :len(p)] = p
        hyp_padded[i, :len(h)] = h

    return prem_padded, hyp_padded, torch.tensor(step_labels), torch.tensor(global_labels)
