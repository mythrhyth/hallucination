# models.py
import torch
import torch.nn as nn
from transformers import AutoModel

class LSTM_Based_Detector_BERT(nn.Module):
    def __init__(self, hidden_dim=64, pretrained_model="bert-base-uncased", freeze_bert=False):
        super().__init__()
        self.bert = AutoModel.from_pretrained(pretrained_model)
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        self.lstm = nn.LSTM(self.bert.config.hidden_size, hidden_dim,
                            batch_first=True, bidirectional=True)

        self.step_classifier = nn.Linear(hidden_dim*4, 2)  # hallucination or not

    def forward(self, cq_input_ids, cq_attention_mask, steps_input_ids, steps_attention_mask):
        B, N, L = steps_input_ids.shape

        # Encode context+question
        cq_emb = self.bert(input_ids=cq_input_ids, attention_mask=cq_attention_mask).last_hidden_state
        _, (cq_hidden, _) = self.lstm(cq_emb)
        cq_repr = torch.cat((cq_hidden[-2], cq_hidden[-1]), dim=-1)

        # Encode reasoning steps
        steps_input_ids_flat = steps_input_ids.view(B*N, L)
        steps_mask_flat = steps_attention_mask.view(B*N, L)
        steps_emb = self.bert(input_ids=steps_input_ids_flat, attention_mask=steps_mask_flat).last_hidden_state
        _, (steps_hidden, _) = self.lstm(steps_emb)
        steps_repr = torch.cat((steps_hidden[-2], steps_hidden[-1]), dim=-1).view(B, N, -1)

        cq_expanded = cq_repr.unsqueeze(1).expand(-1, N, -1)
        combined_repr = torch.cat([steps_repr, cq_expanded], dim=-1)

        step_logits = self.step_classifier(combined_repr)  # [B, N, 2]
        return step_logits
