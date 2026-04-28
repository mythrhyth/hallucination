import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM_Attention_HallucinationDetector(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, pretrained_embeddings=None, freeze_embeddings=False):
        super().__init__()

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            if freeze_embeddings:
                self.embedding.weight.requires_grad = False

        # LSTM encoders
        self.cq_lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.step_lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)

        # Attention projection
        self.attn_proj = nn.Linear(hidden_dim*2, hidden_dim*2)

        # Step classifier
        self.step_classifier = nn.Linear(hidden_dim*4, 2)  # step_repr + attended_context

    def forward(self, cq, steps):
        """
        cq: [B, cq_seq_len]
        steps: [B, num_steps, step_seq_len]
        """
        B, N, L = steps.shape

        # --- Context + Question Encoding ---
        cq_emb = self.embedding(cq)                        # [B, cq_seq_len, emb_dim]
        _, (cq_hidden, _) = self.cq_lstm(cq_emb)          # [2, B, hidden]
        cq_repr = torch.cat((cq_hidden[-2], cq_hidden[-1]), dim=-1)  # [B, hidden*2]

        # --- Reasoning Steps Encoding ---
        steps_flat = steps.view(B*N, L)
        steps_emb = self.embedding(steps_flat)            # [B*N, L, emb_dim]
        _, (steps_hidden, _) = self.step_lstm(steps_emb)
        step_repr = torch.cat((steps_hidden[-2], steps_hidden[-1]), dim=-1)  # [B*N, hidden*2]
        step_repr = step_repr.view(B, N, -1)             # [B, N, hidden*2]

        # --- Attention over CQ for each step ---
        cq_attn = self.attn_proj(cq_repr).unsqueeze(1)   # [B, 1, hidden*2]
        attn_scores = torch.bmm(step_repr, cq_attn.transpose(1,2)).squeeze(-1)  # [B, N]
        attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(-1)               # [B, N, 1]
        attended_cq = attn_weights * step_repr                                       # [B, N, hidden*2]

        # Concatenate step_repr + attended context
        combined = torch.cat([step_repr, attended_cq], dim=-1)  # [B, N, hidden*4]

        # Step-level logits
        step_logits = self.step_classifier(combined)           # [B, N, 2]

        return step_logits
