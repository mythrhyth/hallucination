import torch
import torch.nn as nn

class LSTM_Based_Detector(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, pretrained_embeddings=None, freeze_embeddings=False):
        super().__init__()

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # Load pretrained embeddings (e.g., GloVe)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            if freeze_embeddings:
                self.embedding.weight.requires_grad = False

        # Shared bidirectional LSTM
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)

        # Classifiers
        self.step_classifier = nn.Linear(hidden_dim * 4, 2)  # Per-step classification


    def forward(self, cq, steps):
        """
        cq: [batch_size, cq_seq_len]
        steps: [batch_size, num_steps, step_seq_len]
        """

        cq_emb = self.embedding(cq)                     # [B, cq_len, emb_dim]
        _, (cq_hidden, _) = self.lstm(cq_emb)          # [2, B, hidden_dim]
        cq_repr = torch.cat((cq_hidden[-2], cq_hidden[-1]), dim=-1)  # [B, hidden*2]

        # Reasoning steps
        batch_size, num_steps, step_seq_len = steps.shape
        steps = steps.view(batch_size * num_steps, step_seq_len)      # flatten steps
        steps_emb = self.embedding(steps)                             # [B*N, step_seq_len, emb_dim]
        _, (steps_hidden, _) = self.lstm(steps_emb)                   # [2, B*N, hidden_dim]


        steps_repr = torch.cat((steps_hidden[-2], steps_hidden[-1]), dim=-1)  # [B*N, hidden*2]
        steps_repr = steps_repr.view(batch_size, num_steps, -1)       # [B, N, hidden*2]

        cq_exp = cq_repr.unsqueeze(1).expand(-1, num_steps, -1)  # [B, N, hidden*2]
        combined = torch.cat([steps_repr, cq_exp], dim=-1)       # [B, N, hidden*4]

        # Step-level logits
        step_logits = self.step_classifier(combined)             # [B, N, 2]

        return step_logits
