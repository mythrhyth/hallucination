import torch
import torch.nn as nn

PAD_IDX = 0
class NLIModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim = 100, hidden_dim = 64):
        super(NLIModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size,embedding_dim,padding_idx=PAD_IDX)
        self.fc1 = nn.Linear(embedding_dim * 3, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 2)
        self.log_softmax = nn.LogSoftmax(dim = 1)

    def forward(self, premise, hypothesis):
        prem_emb = self.embedding(premise).mean(dim = 1)
        hyp_emb = self.embedding(hypothesis).mean(dim = 1)
        features = torch.cat([prem_emb, hyp_emb, torch.abs(prem_emb - hyp_emb)], dim=1)
        x = self.fc1(features)
        x = self.relu(x)
        x = self.fc2(x)
        return self.log_softmax(x)

class GlobalClassifier(nn.Module):
    def __init__(self,hidden_dim=32):
        super(GlobalClassifier, self).__init__()
        self.fc1 = nn.Linear(1, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 2)  # 0 = incorrect reasoning, 1 = correct

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return nn.LogSoftmax(dim=1)(x)
