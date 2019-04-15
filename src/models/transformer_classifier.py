"""
AraBERT / Transformer Sequence Classification Model
Copyright (c) 2018-2021 Yousef Khaled.
"""

import torch
import torch.nn as nn


class ArabicTransformerClassifier(nn.Module):
    """
    Transformer architecture with multi-layer bidirectional self-attention,
    layer normalization, and classification projection head.
    """
    def __init__(self, vocab_size: int = 64000, hidden_dim: int = 768, num_classes: int = 3, dropout_rate: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(512, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=12,
            dim_feedforward=3072,
            dropout=dropout_rate,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4) # Scalable reference encoder
        self.pooler = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        seq_len = input_ids.size(1)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        
        x = self.embedding(input_ids) + self.pos_embedding(positions)
        src_key_padding_mask = (attention_mask == 0) if attention_mask is not None else None
        
        encoded = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        cls_token = encoded[:, 0, :]
        pooled = self.pooler(cls_token)
        logits = self.classifier(self.dropout(pooled))
        return logits
