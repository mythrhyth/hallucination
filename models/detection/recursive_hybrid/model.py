import torch
import torch.nn as nn
from transformers import AutoModel

class RecursiveHybridModel(nn.Module):
    def __init__(self, model_name="microsoft/deberta-v3-small", hidden_size=768,
                 lstm_hidden=512, num_classes=2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(hidden_size, lstm_hidden, num_layers=1,
                            bidirectional=True, batch_first=True)
        self.attn = nn.Linear(lstm_hidden * 2, 1)
        self.classifier = nn.Linear(lstm_hidden * 2, num_classes)
        self.dropout = nn.Dropout(0.1)

    def forward(self, context_encodings, step_inputs):
        """
        context_encodings: dict with 'input_ids', 'attention_mask' for context+question
        step_inputs: tuple of (step_input_ids, step_attention_masks)
                     [B, N, L] each (batch, num_steps, seq_len)
        """
        step_input_ids, step_attention_masks = step_inputs
        B, N, L = step_input_ids.shape

        # ---------- Encode reasoning steps ----------
        flat_input_ids = step_input_ids.view(B * N, L)
        flat_attention = step_attention_masks.view(B * N, L)

        step_outputs = self.encoder(input_ids=flat_input_ids, attention_mask=flat_attention)
        step_cls = step_outputs.last_hidden_state[:, 0, :]  # [B*N, hidden]
        step_emb = step_cls.view(B, N, -1)  # [B, N, hidden]
        step_emb = self.dropout(step_emb)

        # ---------- Encode context + question ----------
        context_outputs = self.encoder(
            input_ids=context_encodings["input_ids"],
            attention_mask=context_encodings["attention_mask"]
        )
        context_vec = context_outputs.last_hidden_state[:, 0, :]  # [B, hidden]
        context_vec = context_vec.unsqueeze(1).repeat(1, N, 1)  # [B, N, hidden]

        # ---------- Combine context with reasoning steps ----------
        fused_input = step_emb + context_vec

        # ---------- LSTM processing ----------
        lstm_out, _ = self.lstm(fused_input)

        # ---------- Attention mechanism ----------
        attn_weights = torch.softmax(self.attn(lstm_out), dim=1)  # [B, N, 1]
        context_vec = (attn_weights * lstm_out).sum(dim=1, keepdim=True)  # [B, 1, 2*lstm_hidden]
        fused = lstm_out + context_vec

        # ---------- Classification ----------
        logits = self.classifier(fused)  # [B, N, num_classes]

        return logits
